import torch
import torch.nn.functional as F
from collections import deque

from torch.distributions import Normal

from models.ppo_model import PPOIndustrialModel


class Distiller:
    def __init__(
        self,
        obs_spaces,
        global_context_dim,
        hidden_dim,
        act_dim,
        device="cuda",
        buffer_size=10000,
        sup_coef=1.0,
        neg_coef=1.0,
        margin=0.2
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = PPOIndustrialModel(
            task_input_dim=obs_spaces["task"],
            worker_load_input_dim=obs_spaces["worker_load"],
            worker_profile_input_dim=obs_spaces["worker_profile"],
            n_worker=obs_spaces["n_worker"],
            num_pad_tasks=obs_spaces["num_pad_tasks"],
            global_context_dim=global_context_dim,
            hidden_dim=hidden_dim
        ).to(self.device)
        self.buffer = deque(maxlen=buffer_size)

        self.act_dim = act_dim
        self.sup_coef = sup_coef
        self.neg_coef = neg_coef
        self.margin = margin

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)

    def collect(self, obs_dict, actions, pid):
        """
        收集一批数据，obs_dict 字典含 task_obs、worker_loads 等张量；
        actions: [B, n_worker, num_pad_tasks]
        pid:     [B]，对应每条轨迹来源的子策略 index
        """
        for i in range(actions.shape[0]):
            item = {
                "task_obs": obs_dict["task_obs"][i],
                "worker_loads": obs_dict["worker_loads"][i],
                "worker_profiles": obs_dict["worker_profile"][i],
                "global_context": obs_dict["global_context"][i],
                "valid_mask": obs_dict["valid_mask"][i],
                "action": actions[i],
                "pid": pid[i].item()
            }
            self.buffer.append(item)

    def bc_update(self, steps=300):
        """
        从 buffer 中取数据进行蒸馏训练，默认训练若干 step（不清空 buffer）
        """
        if len(self.buffer) < 32:
            return

        for _ in range(steps):
            batch = [self.buffer[i] for i in torch.randint(0, len(self.buffer), (32,))]
            obs_keys = ["task_obs", "worker_loads", "worker_profiles", "global_context", "valid_mask"]
            inputs = {k: torch.stack([torch.tensor(d[k]) for d in batch]).to(self.device)
                      for k in obs_keys}
            target_actions = torch.stack([torch.tensor(d["action"]) for d in batch]).to(self.device)
            pids = torch.tensor([d["pid"] for d in batch], device=self.device)

            # 分为正负样本
            pos_mask = (pids >= 0)
            neg_mask = (pids < 0)

            mean, _, _ = self.model(**inputs)

            pos_loss, neg_loss = 0.0, 0.0
            if pos_mask.any():
                a_pos = target_actions[pos_mask]
                mean_pos = mean[pos_mask]
                pos_loss = F.mse_loss(mean_pos, a_pos)

            if neg_mask.any():
                a_neg = target_actions[neg_mask]
                mean_neg = mean[neg_mask]
                # margin-based 对比蒸馏（压制负样本行为）
                dist = F.mse_loss(mean_neg, a_neg, reduction="none")  # [B, W, T]
                margin_loss = F.relu(self.margin - dist).mean()
                neg_loss = margin_loss

            loss = self.sup_coef * pos_loss + self.neg_coef * neg_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            return loss.item()

    def predict(self, obs_dict):
        """
        推理接口（用于部署主策略）
        """
        with torch.no_grad():
            inputs = {k: obs_dict[k].to(self.device) for k in [
                "task_obs", "worker_loads", "worker_profiles", "global_context", "valid_mask"]}
            mean, std, _ = self.model(**inputs)
            dist = Normal(mean, std)
            action = dist.sample()
            action = torch.clamp(action, 0, 1)
        return action
