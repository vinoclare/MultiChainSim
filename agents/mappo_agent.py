# agents/mappo_agent.py
import torch
import numpy as np


class IndustrialAgent:
    def __init__(self, algorithm, algo_type="mappo", device="cuda", num_pad_tasks=30, profile_dim=None):
        self.alg = algorithm
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.algo_type = algo_type
        self.num_pad_tasks = num_pad_tasks
        self.profile_dim = profile_dim

    def _pad_task_obs(self, task_obs_np):
        num_tasks, feat_dim = task_obs_np.shape
        padded = np.zeros((self.num_pad_tasks, feat_dim), dtype=np.float32)
        padded[:num_tasks] = task_obs_np
        tensor = torch.tensor(padded, dtype=torch.float32, device=self.device).unsqueeze(0)
        return tensor

    def _pad_worker_obs(self, worker_obs_np):
        tensor = torch.tensor(worker_obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        return tensor

    def _pad_worker_profile(self, profile_np):
        tensor = torch.tensor(profile_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        return tensor

    def _get_valid_mask(self, task_obs_tensor):
        valid_flag = task_obs_tensor[0, :, -1]
        return (valid_flag > 0).float().unsqueeze(0)  # (1, T)

    def predict(self, task_obs_np, worker_obs_np, worker_profile_np, macro_hist_np=None):
        task_obs = self._pad_task_obs(task_obs_np)
        worker_obs = self._pad_worker_obs(worker_obs_np)
        profile = self._pad_worker_profile(worker_profile_np)
        valid_mask = self._get_valid_mask(task_obs)

        # WTOE 不需要在 predict 时强制用 macro_hist（评估一般走确定性 mean）
        mean = self.alg.predict(task_obs, worker_obs, profile, valid_mask)
        return mean.detach().cpu().numpy()[0]

    def sample(self, task_obs_np, worker_obs_np, worker_profile_np, macro_hist_np=None):
        task_obs = self._pad_task_obs(task_obs_np)
        worker_obs = self._pad_worker_obs(worker_obs_np)
        profile = self._pad_worker_profile(worker_profile_np)
        valid_mask = self._get_valid_mask(task_obs)

        # 关键改动：如果算法支持 macro_hist，就透传
        if macro_hist_np is not None:
            macro_hist = torch.tensor(macro_hist_np, dtype=torch.float32, device=self.device).unsqueeze(0)
            try:
                value, action, log_prob, entropy = self.alg.sample(
                    task_obs, worker_obs, profile, valid_mask, macro_hist=macro_hist
                )
            except TypeError:
                # 兼容旧算法签名
                value, action, log_prob, entropy = self.alg.sample(
                    task_obs, worker_obs, profile, valid_mask
                )
        else:
            value, action, log_prob, entropy = self.alg.sample(
                task_obs, worker_obs, profile, valid_mask
            )

        return (
            value[0].detach().cpu().numpy(),
            action[0].detach().cpu().numpy(),
            log_prob[0].detach().cpu().numpy(),
            entropy[0].detach().cpu().numpy()
        )

    def learn(self, *args, **kwargs):
        return self.alg.learn(*args, **kwargs)
