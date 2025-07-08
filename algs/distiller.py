import torch
import torch.nn.functional as F
from collections import deque
import copy

import random
from torch.distributions import Normal, kl_divergence

from models.ppo_model import PPOIndustrialModel
from models.agent57_model import Agent57IndustrialModel


class Distiller:
    def __init__(
            self,
            obs_spaces,
            global_context_dim,
            hidden_dim,
            act_dim,
            K,
            loss_type="mse",
            neg_policy=False,
            device="cuda",
            buffer_size=10000,
            sup_coef=1.0,
            neg_coef=1.0,
            margin=3.0,
            std_t=0.2
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
        with torch.no_grad():
            self.model.log_std.fill_(-0.5)

        self.target_model = copy.deepcopy(self.model).eval()
        self.polyak_tau = 0.95

        self.act_dim = act_dim
        self.sup_coef = sup_coef
        self.neg_coef = neg_coef
        self.margin = margin
        self.var_coef = 0.005
        self.sigma_target = 0.8
        self.std_t = std_t

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)

        self.K = K
        self.pos_pid_max = K - 3
        self.loss_type = loss_type  # mse or kl
        self.neg_policy = neg_policy

        self.buffers = {pid: deque(maxlen=buffer_size) for pid in range(K)}

    def collect(self, obs_dict, actions, pid):
        """
        收集一批数据，obs_dict 字典含 task_obs、worker_loads 等张量；
        actions: [B, n_worker, num_pad_tasks]
        pid:     [B]，对应每条轨迹来源的子策略 index
        """
        obs_cpu = {k: v.detach().cpu() for k, v in obs_dict.items()}
        for i in range(actions.shape[0]):
            cur_pid = pid[i]
            self.buffers[cur_pid].append({
                "obs": {k: obs_cpu[k][i] for k in obs_cpu},
                "action": actions[i].detach().cpu()
            })

    def _sample_from_buffers(self, cur_pid: int, batch_size: int):
        buf = self.buffers[cur_pid]
        if len(buf) < batch_size:
            return random.sample(buf, len(buf))  # 不足则全取
        return random.sample(buf, batch_size)

    def bc_update(self, cur_pid: int, batch_size: int = 512, steps: int = 50):
        """对主策略进行若干步蒸馏更新"""
        if len(self.buffers[cur_pid]) < batch_size:
            return 0.0  # 数据不足

        total_loss = 0.0
        for _ in range(steps):
            batch = self._sample_from_buffers(cur_pid, batch_size)
            if len(batch) == 0:
                break

            # ---- 构造张量 ----
            obs_keys = ["task_obs", "worker_loads", "worker_profiles",
                        "global_context", "valid_mask"]
            obs_t = {k: torch.stack([torch.tensor(d["obs"][k])
                                     for d in batch]).to(self.device)
                     for k in obs_keys}
            target_actions = torch.stack([torch.tensor(d["action"])
                                          for d in batch]).to(self.device)

            # ---- 主策略前向 ----
            mean_s, std_s, _ = self.model(**obs_t)

            if self.neg_policy:
                is_pos_policy = cur_pid <= self.pos_pid_max
            else:
                is_pos_policy = True

            if is_pos_policy:
                if self.loss_type == "mse":
                    loss_pos = F.mse_loss(mean_s, target_actions)
                    loss = self.sup_coef * loss_pos
                else:  # "kl"
                    # teacher 视为 N(mean_t, σ=std_t) 的固定高斯
                    teacher_std = torch.full_like(mean_s, self.std_t)
                    dist_t = Normal(target_actions, teacher_std)
                    dist_s = Normal(mean_s, std_s.clamp(min=1e-3))
                    loss_kl = kl_divergence(dist_t, dist_s).mean()
                    loss = self.sup_coef * loss_kl
            else:
                dist_s = Normal(mean_s, std_s.clamp(min=1e-3))

                # ---- Softplus-log 概率惩罚 ----
                # log π(a⁻|s) 在多维动作上需按维求和
                logp_neg = dist_s.log_prob(target_actions).sum(-1)  # [B]
                loss_sp = F.softplus(logp_neg + self.margin).mean()  # margin≈2-4

                # ---- σ 正则，防止放大 σ 绕过约束 ----
                loss_var = (std_s - self.sigma_target).clamp(min=0).pow(2).mean()

                loss = self.neg_coef * loss_sp + self.var_coef * loss_var

            # ---- 反向更新 ----
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()

            total_loss += loss.item()

            with torch.no_grad():
                for p_targ, p_main in zip(self.target_model.parameters(), self.model.parameters()):
                    p_targ.data.mul_(self.polyak_tau).add_(p_main.data, alpha=1 - self.polyak_tau)

        return total_loss / steps

    @torch.no_grad()
    def predict(self, obs_dict):
        inputs = {k: v.to(self.device) for k, v in obs_dict.items()}
        mean, std, _ = self.model(**inputs)
        dist = Normal(mean, std)
        action = dist.sample().clamp(0, 1)
        return action


# 确定性动作
class Distiller2:
    def __init__(
            self,
            obs_spaces,
            global_context_dim,
            hidden_dim,
            act_dim,
            K,
            loss_type="mse",
            neg_policy=False,
            device="cuda",
            buffer_size=10000,
            sup_coef=1.0,
            neg_coef=1.0,
            margin=3.0,
            std_t=0.2
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
        with torch.no_grad():
            self.model.log_std.fill_(-0.5)

        self.target_model = copy.deepcopy(self.model).eval()
        self.polyak_tau = 0.95

        self.act_dim = act_dim
        self.sup_coef = sup_coef
        self.neg_coef = neg_coef
        self.margin = margin
        self.var_coef = 0.005
        self.sigma_target = 0.8
        self.std_t = std_t

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)

        self.K = K
        self.pos_pid_max = K - 3
        self.loss_type = loss_type  # mse or kl
        self.neg_policy = neg_policy

        self.buffers = {pid: deque(maxlen=buffer_size) for pid in range(K)}

    def collect(self, obs_dict, actions, pid):
        """
        收集一批数据，obs_dict 字典含 task_obs、worker_loads 等张量；
        actions: [B, n_worker, num_pad_tasks]
        pid:     [B]，对应每条轨迹来源的子策略 index
        """
        obs_cpu = {k: v.detach().cpu() for k, v in obs_dict.items()}
        for i in range(actions.shape[0]):
            cur_pid = pid[i]
            self.buffers[cur_pid].append({
                "obs": {k: obs_cpu[k][i] for k in obs_cpu},
                "action": actions[i].detach().cpu()
            })

    def _sample_from_buffers(self, cur_pid: int, batch_size: int):
        buf = self.buffers[cur_pid]
        if len(buf) < batch_size:
            return random.sample(buf, len(buf))  # 不足则全取
        return random.sample(buf, batch_size)

    def bc_update(self, cur_pid: int, batch_size: int = 512, steps: int = 50):
        """对主策略进行若干步蒸馏更新（直接生成动作）"""
        if len(self.buffers[cur_pid]) < batch_size:
            return 0.0  # 数据不足

        total_loss = 0.0
        for _ in range(steps):
            batch = self._sample_from_buffers(cur_pid, batch_size)
            if len(batch) == 0:
                break

            # ---- 构造张量 ----
            obs_keys = ["task_obs", "worker_loads", "worker_profiles",
                        "global_context", "valid_mask"]
            obs_t = {k: torch.stack([torch.tensor(d["obs"][k])
                                     for d in batch]).to(self.device)
                     for k in obs_keys}
            target_actions = torch.stack([torch.tensor(d["action"])
                                          for d in batch]).to(self.device)

            # ---- 主策略前向（直接输出动作） ----
            predicted_actions, _ = self.model.forward_direct(**obs_t)

            if self.neg_policy:
                is_pos_policy = cur_pid <= self.pos_pid_max
            else:
                is_pos_policy = True

            if is_pos_policy:
                loss = self.sup_coef * F.mse_loss(predicted_actions, target_actions)
            else:
                # 对于负子策略，依然保留 softplus-log 概率惩罚逻辑，可按需求调整
                diff = (predicted_actions - target_actions).pow(2).sum(dim=[1, 2])
                loss_sp = F.softplus(diff + self.margin).mean()
                loss = self.neg_coef * loss_sp

            # ---- 反向更新 ----
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / steps

    @torch.no_grad()
    def predict(self, obs_dict):
        inputs = {k: v.to(self.device) for k, v in obs_dict.items()}
        predicted_actions, _ = self.model.forward_direct(**inputs)
        action = predicted_actions.clamp(0, 1)
        return action


# 使用单头Agent57模型作为主策略
class Distiller3:
    def __init__(
            self,
            obs_spaces,
            global_context_dim,
            hidden_dim,
            act_dim,
            K,
            loss_type="mse",
            neg_policy=False,
            device="cuda",
            buffer_size=10000,
            sup_coef=1.0,
            neg_coef=1.0,
            margin=3.0,
            std_t=0.2
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = Agent57IndustrialModel(
            task_input_dim=obs_spaces["task"],
            worker_load_input_dim=obs_spaces["worker_load"],
            worker_profile_input_dim=obs_spaces["worker_profile"],
            n_worker=obs_spaces["n_worker"],
            num_pad_tasks=obs_spaces["num_pad_tasks"],
            global_context_dim=global_context_dim,
            hidden_dim=hidden_dim,
            K=1,
            neg_policy=False
        ).to(self.device)

        self.target_model = copy.deepcopy(self.model).eval()
        self.polyak_tau = 0.95

        self.act_dim = act_dim
        self.sup_coef = sup_coef
        self.neg_coef = neg_coef
        self.margin = margin
        self.var_coef = 0.005
        self.sigma_target = 0.8
        self.std_t = std_t

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)

        self.K = K
        self.pos_pid_max = K - 3
        self.loss_type = loss_type  # mse or kl
        self.neg_policy = neg_policy

        self.buffers = {pid: deque(maxlen=buffer_size) for pid in range(K)}

    def collect(self, obs_dict, actions, pid):
        """
        收集一批数据，obs_dict 字典含 task_obs、worker_loads 等张量；
        actions: [B, n_worker, num_pad_tasks]
        pid:     [B]，对应每条轨迹来源的子策略 index
        """
        obs_cpu = {k: v.detach().cpu() for k, v in obs_dict.items()}
        for i in range(actions.shape[0]):
            cur_pid = pid[i]
            self.buffers[cur_pid].append({
                "obs": {k: obs_cpu[k][i] for k in obs_cpu},
                "action": actions[i].detach().cpu()
            })

    def _sample_from_buffers(self, cur_pid: int, batch_size: int):
        buf = self.buffers[cur_pid]
        if len(buf) < batch_size:
            return random.sample(buf, len(buf))  # 不足则全取
        return random.sample(buf, batch_size)

    def bc_update(self, cur_pid: int, batch_size: int = 512, steps: int = 50):
        """对主策略进行若干步蒸馏更新"""
        if len(self.buffers[cur_pid]) < batch_size:
            return 0.0  # 数据不足

        total_loss = 0.0
        for _ in range(steps):
            batch = self._sample_from_buffers(cur_pid, batch_size)
            if len(batch) == 0:
                break

            # ---- 构造张量 ----
            obs_keys = ["task_obs", "worker_loads", "worker_profiles",
                        "global_context", "valid_mask"]
            obs_t = {k: torch.stack([torch.tensor(d["obs"][k])
                                     for d in batch]).to(self.device)
                     for k in obs_keys}
            obs_t["policy_id"] = 0
            target_actions = torch.stack([torch.tensor(d["action"])
                                          for d in batch]).to(self.device)

            # ---- 主策略前向 ----
            mean_s, std_s, _, _ = self.model(**obs_t)

            if self.neg_policy:
                is_pos_policy = cur_pid <= self.pos_pid_max
            else:
                is_pos_policy = True

            if is_pos_policy:
                if self.loss_type == "mse":
                    loss_pos = F.mse_loss(mean_s, target_actions)
                    loss = self.sup_coef * loss_pos
                else:  # "kl"
                    # teacher 视为 N(mean_t, σ=std_t) 的固定高斯
                    teacher_std = torch.full_like(mean_s, self.std_t)
                    dist_t = Normal(target_actions, teacher_std)
                    dist_s = Normal(mean_s, std_s.clamp(min=1e-3))
                    loss_kl = kl_divergence(dist_t, dist_s).mean()
                    loss = self.sup_coef * loss_kl
            else:
                dist_s = Normal(mean_s, std_s.clamp(min=1e-3))

                # ---- Softplus-log 概率惩罚 ----
                # log π(a⁻|s) 在多维动作上需按维求和
                logp_neg = dist_s.log_prob(target_actions).sum(-1)  # [B]
                loss_sp = F.softplus(logp_neg + self.margin).mean()  # margin≈2-4

                # ---- σ 正则，防止放大 σ 绕过约束 ----
                loss_var = (std_s - self.sigma_target).clamp(min=0).pow(2).mean()

                loss = self.neg_coef * loss_sp + self.var_coef * loss_var

            # ---- 反向更新 ----
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()

            total_loss += loss.item()

            with torch.no_grad():
                for p_targ, p_main in zip(self.target_model.parameters(), self.model.parameters()):
                    p_targ.data.mul_(self.polyak_tau).add_(p_main.data, alpha=1 - self.polyak_tau)

        return total_loss / steps

    @torch.no_grad()
    def predict(self, obs_dict):
        inputs = {k: v.to(self.device) for k, v in obs_dict.items()}
        inputs["policy_id"] = 0
        mean, std, _, _ = self.model(**inputs)
        dist = Normal(mean, std)
        action = dist.sample().clamp(0, 1)
        return action
