# agents/varibad_agent.py
import numpy as np
import torch
from torch.distributions import Normal


class VariBADIndustrialAgent:
    """
    VariBAD Agent 封装：负责
    1) 把 env 的 numpy obs -> torch tensor（含 padding 与 valid_mask）
    2) 维护 belief 状态（RNN hidden + 上一步动作/奖励/done）
    3) 调用 alg.sample(...) 得到 action/value/logp/entropy
    """

    def __init__(
        self,
        algorithm,
        device="cuda",
        num_pad_tasks=30,
        valid_index=-1,
        action_shape=None,  # e.g., (n_worker, act_dim_per_worker)；None 则直接返回扁平 action
    ):
        self.alg = algorithm
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.num_pad_tasks = num_pad_tasks
        self.valid_index = valid_index
        self.action_shape = action_shape

        # belief / prev-feedback 缓存
        self._h = None
        self._prev_action = None
        self._prev_reward = None
        self._prev_done = None

        # 尝试从 model 推断维度（要求你的 varibad_model 有这些属性）
        model = self.alg.model
        self._belief_hidden = getattr(getattr(model, "belief_rnn", None), "hidden_size", None)
        self._action_dim_flat = getattr(getattr(model, "policy_mean", None), "out_features", None)

    # ----------------------- padding & packing -----------------------

    def _pad_task_obs(self, task_obs_np: np.ndarray) -> torch.Tensor:
        """
        task_obs_np: [num_tasks, feat_dim]
        return: [1, num_pad_tasks, feat_dim]
        """
        num_tasks, feat_dim = task_obs_np.shape
        padded = np.zeros((self.num_pad_tasks, feat_dim), dtype=np.float32)
        padded[: min(num_tasks, self.num_pad_tasks)] = task_obs_np[: self.num_pad_tasks]
        return torch.tensor(padded, dtype=torch.float32, device=self.device).unsqueeze(0)

    def _pad_worker_loads(self, worker_loads_np: np.ndarray) -> torch.Tensor:
        """
        worker_loads_np: [n_worker, feat_dim]
        return: [1, n_worker, feat_dim]
        """
        return torch.tensor(worker_loads_np, dtype=torch.float32, device=self.device).unsqueeze(0)

    def _pad_worker_profile(self, profile_np: np.ndarray) -> torch.Tensor:
        """
        profile_np: [profile_dim] 或 [n_worker, profile_dim]（按你的 env 实际来）
        return: [1, ...]
        """
        return torch.tensor(profile_np, dtype=torch.float32, device=self.device).unsqueeze(0)

    def _get_valid_mask(self, task_obs_tensor: torch.Tensor) -> torch.Tensor:
        """
        task_obs_tensor: [1, num_pad_tasks, feat_dim]
        return: [1, num_pad_tasks]
        """
        valid_flag = task_obs_tensor[0, :, self.valid_index]
        return (valid_flag > 0).float().unsqueeze(0)

    def _build_raw_obs_vector(
        self,
        task_obs_tensor: torch.Tensor,
        worker_loads_tensor: torch.Tensor,
        profile_tensor: torch.Tensor,
        valid_mask_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """
        把结构化 obs 拼成一个扁平向量，给 varibad_model.encode_obs(raw_obs) 用
        return: [1, obs_dim]
        """
        # 注意：profile 可能是 [1, P] 或 [1, n_worker, P]
        flat_parts = [
            task_obs_tensor.reshape(1, -1),
            worker_loads_tensor.reshape(1, -1),
            profile_tensor.reshape(1, -1),
            valid_mask_tensor.reshape(1, -1),
        ]
        return torch.cat(flat_parts, dim=-1)

    # ----------------------- belief state -----------------------

    def reset_belief(self, batch_size: int = 1):
        """
        每个 episode 开始时调用一次（很关键）。
        """
        if self._belief_hidden is None or self._action_dim_flat is None:
            raise RuntimeError(
                "Cannot infer belief/action dims from model. "
                "Make sure model has `belief_rnn.hidden_size` and `policy_mean.out_features`."
            )

        self._h = torch.zeros((batch_size, self._belief_hidden), dtype=torch.float32, device=self.device)
        self._prev_action = torch.zeros((batch_size, self._action_dim_flat), dtype=torch.float32, device=self.device)
        self._prev_reward = torch.zeros((batch_size, 1), dtype=torch.float32, device=self.device)
        self._prev_done = torch.zeros((batch_size, 1), dtype=torch.float32, device=self.device)

    reset = reset_belief

    def set_prev_feedback(self, reward, done):
        """
        在 env.step(actions) 之后，把每层得到的 reward/done 喂回来，
        下一次 sample() 会用到（VariBAD belief 更新需要 r_t, done_{t-1}）。
        """
        if self._prev_reward is None or self._prev_done is None:
            # 如果忘了 reset_belief，就先补上
            self.reset_belief(batch_size=1)

        r = float(reward)
        d = float(done)
        self._prev_reward[:] = r
        self._prev_done[:] = d

    # ----------------------- acting -----------------------

    @torch.no_grad()
    def sample(
        self,
        task_obs_np: np.ndarray,
        worker_loads_np: np.ndarray,
        worker_profile_np: np.ndarray,
        reward_prev=None,
        done_prev=None,
        deterministic: bool = False,
        return_belief: bool = False,
    ):
        """
        - deterministic=True：用 mean 作为 action（评估时可用）
        - return_belief=True：额外返回 (z, mu, logvar, h) 方便你存 buffer 做 ELBO
        """
        if self._h is None:
            self.reset_belief(batch_size=1)

        # obs -> tensor（含 pad/mask）
        task_obs = self._pad_task_obs(task_obs_np)
        worker_loads = self._pad_worker_loads(worker_loads_np)
        profile = self._pad_worker_profile(worker_profile_np)
        valid_mask = self._get_valid_mask(task_obs)

        raw_obs = self._build_raw_obs_vector(task_obs, worker_loads, profile, valid_mask)

        # 选择本次用于 belief 更新的 (r_prev, done_prev)
        if reward_prev is None:
            r_prev = self._prev_reward
        else:
            r_prev = torch.tensor([[float(reward_prev)]], dtype=torch.float32, device=self.device)

        if done_prev is None:
            d_prev = self._prev_done
        else:
            d_prev = torch.tensor([[float(done_prev)]], dtype=torch.float32, device=self.device)

        # encode obs
        s_feat = self.alg.model.encode_obs(raw_obs)

        # 调用算法：belief update + policy/value
        # 约定 alg.sample 返回：
        # (action, logp, entropy, value, h_new, z, mu, logvar)
        out = self.alg.sample(s_feat, self._prev_action, r_prev, d_prev, self._h)
        action_flat, logp, entropy, value, h_new, z, mu, logvar = out

        # deterministic：用 mean 替换采样 action（belief 仍然按本步更新）
        if deterministic:
            x = self.alg.model.policy_fc(torch.cat([s_feat, z], dim=-1))
            mean = self.alg.model.policy_mean(x)
            # logstd 可能是 Parameter([action_dim])，这里 broadcast 到 batch
            logstd = self.alg.model.policy_logstd.view(1, -1).expand_as(mean)
            std = torch.exp(logstd)
            dist = Normal(mean, std)
            action_flat = mean
            logp = dist.log_prob(action_flat).sum(-1, keepdim=True)
            entropy = dist.entropy().sum(-1, keepdim=True)

        # 更新 belief & prev_action（供下一步用）
        self._h = h_new
        self._prev_action = action_flat
        # reward/done 默认不在这里更新（因为要等 env.step 之后才知道），靠 set_prev_feedback()

        # action reshape 回 env 需要的形状
        act_np_flat = action_flat[0].detach().cpu().numpy()
        if self.action_shape is not None:
            act_np = act_np_flat.reshape(self.action_shape)
        else:
            act_np = act_np_flat

        value_np = value[0].detach().cpu().numpy()
        logp_np = logp[0].detach().cpu().numpy()
        ent_np = entropy[0].detach().cpu().numpy()

        if not return_belief:
            return value_np, act_np, logp_np, ent_np

        belief_pack = {
            "z": z[0].detach().cpu().numpy(),
            "mu": mu[0].detach().cpu().numpy(),
            "logvar": logvar[0].detach().cpu().numpy(),
            "h": h_new[0].detach().cpu().numpy(),
        }
        return value_np, act_np, logp_np, ent_np, belief_pack
