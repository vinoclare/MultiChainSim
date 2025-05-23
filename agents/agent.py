import torch
import numpy as np


class IndustrialAgent:
    def __init__(self, algorithm, algo_type="ppo", num_pad_tasks=10, global_context_dim=1, profile_dim= None):
        self.alg = algorithm
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.algo_type = algo_type
        self.num_pad_tasks = num_pad_tasks
        self.global_context_dim = global_context_dim
        self.profile_dim = profile_dim  # expected length of worker_profile features

        if algo_type == "sac":
            self.alg.sync_target(decay=0)

    def _pad_task_obs(self, task_obs_np):
        # task_obs_np: (n_tasks, feat_dim)
        num_tasks, feat_dim = task_obs_np.shape
        padded = np.zeros((self.num_pad_tasks, feat_dim), dtype=np.float32)
        padded[:num_tasks] = task_obs_np
        tensor = torch.tensor(padded, dtype=torch.float32, device=self.device).unsqueeze(0)
        return tensor  # (1, num_pad_tasks, feat_dim)

    def _pad_worker_obs(self, worker_obs_np):
        # worker_obs_np: (n_worker, feat_dim)
        n_worker, feat = worker_obs_np.shape
        tensor = torch.tensor(worker_obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        return tensor  # (1, n_worker, feat)

    def _pad_worker_profile(self, profile_np):
        # profile_np: (n_worker, profile_dim)
        tensor = torch.tensor(profile_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        return tensor  # (1, n_worker, profile_dim)

    def _get_global_context(self, global_context_np):
        # global_context_np: (global_context_dim,)
        tensor = torch.tensor(global_context_np, dtype=torch.float32, device=self.device).reshape(1, -1)
        return tensor  # (1, global_context_dim)

    def _get_valid_mask(self, task_obs_tensor):
        # task_obs_tensor: (1, num_pad_tasks, feat_dim), last dim's last element is valid_flag
        valid_flag = task_obs_tensor[0, :, -1]
        return (valid_flag > 0).float().unsqueeze(0)  # (1, num_pad_tasks)

    def predict(self,
                task_obs_np,
                worker_obs_np,
                worker_profile_np,
                global_context_np):
        # Prepare tensors
        task_obs = self._pad_task_obs(task_obs_np)
        worker_obs = self._pad_worker_obs(worker_obs_np)
        profile = self._pad_worker_profile(worker_profile_np)
        gctx = self._get_global_context(global_context_np)
        valid_mask = self._get_valid_mask(task_obs)

        # Call algorithm's predict (deterministic)
        mean = self.alg.predict(task_obs, worker_obs, profile, gctx, valid_mask)
        return mean.detach().cpu().numpy()[0]

    def sample(self,
               task_obs_np,
               worker_obs_np,
               worker_profile_np,
               global_context_np):
        # Prepare tensors
        task_obs = self._pad_task_obs(task_obs_np)
        worker_obs = self._pad_worker_obs(worker_obs_np)
        profile = self._pad_worker_profile(worker_profile_np)
        gctx = self._get_global_context(global_context_np)
        valid_mask = self._get_valid_mask(task_obs)

        # Sample from policy
        if self.algo_type == "ppo":
            value, action, log_prob, entropy = self.alg.sample(
                task_obs, worker_obs, profile, gctx, valid_mask)
            return (
                value[0].detach().cpu().numpy(),
                action[0].detach().cpu().numpy(),
                log_prob[0].detach().cpu().numpy(),
                entropy[0].detach().cpu().numpy()
            )
        else:
            # SAC
            action, log_prob = self.alg.sample(
                task_obs, worker_obs, profile, gctx, valid_mask)
            return (
                None,
                action[0].detach().cpu().numpy(),
                log_prob[0].detach().cpu().numpy(),
                None
            )

    def learn(self, *args, **kwargs):
        return self.alg.learn(*args, **kwargs)

    def sync_target(self, decay=None):
        if self.algo_type == "sac":
            self.alg.sync_target(decay)
