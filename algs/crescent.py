import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from models.structure_encoder import StructureEncoder, StructureProjHead
from utils.contrastive_utils import compute_contrastive_loss


class CRESCENT:
    def __init__(self,
                 model,
                 clip_param=0.1,
                 value_loss_coef=0.5,
                 entropy_coef=0.01,
                 initial_lr=2.5e-4,
                 eps=1e-5,
                 max_grad_norm=0.5,
                 device="cuda",
                 inner_k=1,
                 use_clipped_value_loss=True,
                 norm_adv=True,
                 writer=None,
                 global_step_ref=None,
                 total_training_steps=None,
                 # ====== CReSCENT 相关参数 ======
                 macro_feat_dim=None,  # 必须传：macro_feat 的维度
                 repr_dim=64,
                 proj_dim=128,
                 lambda_ctr=0.1,
                 ctr_temperature=0.1,
                 ctr_time_window=1,
                 use_contrastive=True,
                 # ====== NEW: target encoder / EMA 参数 ======
                 ema_momentum=0.99):

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        self.clip_param = clip_param
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.initial_entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.norm_adv = norm_adv
        self.inner_k = inner_k

        # ====== 结构 encoder + 投影头 ======
        if macro_feat_dim is None:
            raise ValueError("CRESCENT 需要指定 macro_feat_dim 才能初始化结构编码器")

        self.macro_feat_dim = macro_feat_dim
        self.repr_dim = repr_dim
        self.proj_dim = proj_dim
        self.lambda_ctr = lambda_ctr
        self.ctr_temperature = ctr_temperature
        self.ctr_time_window = ctr_time_window
        self.use_contrastive = use_contrastive

        self.struct_encoder = StructureEncoder(
            input_dim=macro_feat_dim,
            repr_dim=repr_dim,
            hidden_dim=128,
            num_hidden_layers=1,
        ).to(self.device)

        self.struct_proj = StructureProjHead(
            repr_dim=repr_dim,
            proj_dim=proj_dim,
        ).to(self.device)

        self.ema_momentum = ema_momentum
        self.target_encoder = StructureEncoder(
            input_dim=macro_feat_dim,
            repr_dim=repr_dim,
            hidden_dim=128,
            num_hidden_layers=1,
        ).to(self.device)
        # 初始时参数与 online encoder 一致
        self.target_encoder.load_state_dict(self.struct_encoder.state_dict())
        # target encoder 不参与梯度更新
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # ====== 优化器：包含 policy + value + online encoder + proj 参数 ======
        params = list(self.model.parameters()) + \
                 list(self.struct_encoder.parameters()) + \
                 list(self.struct_proj.parameters())
        self.optimizer = optim.Adam(params, lr=initial_lr, eps=eps)

        self.writer = writer
        self.global_step_ref = global_step_ref
        self.total_training_steps = total_training_steps

    # ---------------------------------------------------------------------
    # 公共接口：采样 / 预测 / 值函数
    # ---------------------------------------------------------------------
    def sample(self, task_obs, worker_loads, worker_profile, valid_mask):
        task_obs = task_obs.to(self.device)
        worker_loads = worker_loads.to(self.device)
        worker_profile = worker_profile.to(self.device)
        valid_mask = valid_mask.to(self.device)

        action, log_prob, mean, std = self.model.act(
            task_obs, worker_loads, worker_profile, valid_mask)
        dist = Normal(mean, std)
        value = self.model.get_value(task_obs, worker_loads, worker_profile, valid_mask)
        return value, action, log_prob.sum(dim=2), dist.entropy().sum(dim=2)

    def predict(self, task_obs, worker_loads, worker_profile, valid_mask):
        task_obs = task_obs.to(self.device)
        worker_loads = worker_loads.to(self.device)
        worker_profile = worker_profile.to(self.device)
        valid_mask = valid_mask.to(self.device)
        mean, _ = self.model.forward_actor(task_obs, worker_loads, worker_profile, valid_mask)
        return mean

    def value(self, task_obs, worker_loads, worker_profile, valid_mask):
        task_obs = task_obs.to(self.device)
        worker_loads = worker_loads.to(self.device)
        worker_profile = worker_profile.to(self.device)
        valid_mask = valid_mask.to(self.device)
        return self.model.get_value(task_obs, worker_loads, worker_profile, valid_mask)

    # ---------------------------------------------------------------------
    # NEW: 给聚类 / intrinsic 用的接口（使用 EMA target encoder）
    # ---------------------------------------------------------------------
    def encode_macro_for_cluster(self, macro_feat):
        """
        使用 target encoder（EMA）把宏观结构特征映射到 z，用于聚类。
        输入:
          macro_feat: [T, macro_feat_dim] 或 [macro_feat_dim] 的 numpy / tensor
        返回:
          z: numpy.ndarray, shape [T, repr_dim]
        """
        # 转 tensor
        if isinstance(macro_feat, torch.Tensor):
            x = macro_feat.to(self.device, dtype=torch.float32)
        else:
            x = torch.as_tensor(macro_feat, device=self.device, dtype=torch.float32)
        if x.dim() == 1:
            x = x.unsqueeze(0)  # [1, F]

        self.target_encoder.eval()
        with torch.no_grad():
            z = self.target_encoder(x)  # [T, repr_dim]

        # 返回 numpy，方便在 run_crescent 里丢给 clusterer（它能接受 np 或 tensor）
        return z.detach().cpu().numpy()

    # ---------------------------------------------------------------------
    # NEW: EMA 更新 target encoder 参数
    # ---------------------------------------------------------------------
    def update_target_encoder(self):
        """
        使用 EMA 将 target_encoder 向 struct_encoder 缓慢对齐：
          θ_bar ← m * θ_bar + (1-m) * θ
        """
        m = self.ema_momentum
        with torch.no_grad():
            for p_t, p in zip(self.target_encoder.parameters(),
                              self.struct_encoder.parameters()):
                p_t.data.mul_(m).add_(p.data, alpha=1.0 - m)

    # ---------------------------------------------------------------------
    # 训练：HAPPO + 对比学习（online encoder）
    # ---------------------------------------------------------------------
    def learn(self,
              task_obs,
              worker_loads,
              worker_profile,
              valid_mask,
              macro_feat,  # [B, macro_feat_dim]
              episode_ids,  # [B]
              step_ids,  # [B]
              actions,
              values_old,
              returns,
              log_probs_old,  # [B, W]
              advantages,  # [B, W]
              current_steps,
              lr=None):

        device = self.device
        task_obs = task_obs.to(device)
        worker_loads = worker_loads.to(device)
        worker_profile = worker_profile.to(device)
        valid_mask = valid_mask.to(device)
        actions = actions.to(device)
        values_old = values_old.to(device)  # [B]
        returns = returns.to(device)  # [B]
        log_probs_old = log_probs_old.to(device)  # [B, W]
        advantages = advantages.to(device)  # [B, W]

        # macro_feat / ids 转 tensor
        macro_feat = torch.as_tensor(macro_feat, device=device, dtype=torch.float32)
        episode_ids = torch.as_tensor(episode_ids, device=device)
        step_ids = torch.as_tensor(step_ids, device=device)

        if self.norm_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        num_workers = actions.size(1)
        if self.total_training_steps is not None and self.total_training_steps > 0:
            frac = float(current_steps) / float(self.total_training_steps)
            frac = max(0.0, min(1.0, frac))
            entropy_coef = max(1e-3, self.initial_entropy_coef * (1.0 - frac))
        else:
            entropy_coef = self.entropy_coef

        # === 结构对比学习（online encoder） ===
        if self.use_contrastive:
            z = self.struct_encoder(macro_feat)  # [B, repr_dim]
            h = self.struct_proj(z)  # [B, proj_dim], 已 normalize
            ctr_loss_raw = compute_contrastive_loss(
                h,
                episode_ids=episode_ids,
                step_ids=step_ids,
                time_window=self.ctr_time_window,
                temperature=self.ctr_temperature,
            )
            ctr_loss = self.lambda_ctr * ctr_loss_raw
        else:
            ctr_loss_raw = torch.tensor(0.0, device=device)
            ctr_loss = torch.tensor(0.0, device=device)

        # 统计用
        action_loss_total = 0.0
        entropy_total = 0.0

        # === HAPPO 顺序更新 ===
        for w in range(num_workers):
            for inner_idx in range(self.inner_k):
                # 1) 前向当前网络 —— 注意：之前 worker 的更新已经让网络参数发生变化
                mean, std = self.model.forward_actor(task_obs,
                                                     worker_loads,
                                                     worker_profile,
                                                     valid_mask)  # [B, W, A]
                dist = Normal(mean, std)
                log_probs = dist.log_prob(actions).sum(dim=2)  # [B, W]
                entropy = dist.entropy().sum(dim=2)[:, w].mean()  # scalar

                # 2) 仅提取第 w 个 worker 的数据
                ratio = torch.exp(log_probs[:, w] - log_probs_old[:, w])  # [B]
                adv = advantages[:, w]
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv
                action_loss = -torch.min(surr1, surr2).mean()  # scalar

                # 3) critic（共享）损失
                values = self.model.get_value(task_obs,
                                              worker_loads,
                                              worker_profile,
                                              valid_mask).view(-1)  # [B]
                if self.use_clipped_value_loss:
                    value_pred_clipped = values_old + torch.clamp(
                        values - values_old, -self.clip_param, self.clip_param)
                    value_losses = (values - returns) ** 2
                    value_losses_clipped = (value_pred_clipped - returns) ** 2
                    value_loss = 0.5 * torch.max(value_losses,
                                                 value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (returns - values).pow(2).mean()

                # 4) 组合总 loss
                #    注意：对比损失只在第一个 worker 的第一次 inner_k 更新时加一次
                if self.use_contrastive and w == 0 and inner_idx == 0:
                    total_loss = (value_loss * self.value_loss_coef +
                                  action_loss -
                                  entropy_coef * entropy +
                                  ctr_loss)
                else:
                    total_loss = (value_loss * self.value_loss_coef +
                                  action_loss -
                                  entropy_coef * entropy)

                if lr:
                    for pg in self.optimizer.param_groups:
                        pg["lr"] = lr

                self.optimizer.zero_grad()
                total_loss.backward()
                # NEW: clip policy + encoder + proj 的梯度
                nn.utils.clip_grad_norm_(
                    list(self.model.parameters()) +
                    list(self.struct_encoder.parameters()) +
                    list(self.struct_proj.parameters()),
                    self.max_grad_norm
                )
                self.optimizer.step()

                # 5) 累积统计
                action_loss_total += action_loss.item()
                entropy_total += entropy.item()

        # === NEW: 每次 learn 结束后，EMA 更新 target encoder ===
        self.update_target_encoder()

        # 统计输出：取平均值便于对比 MAPPO
        denom = num_workers * max(1, self.inner_k)
        action_loss_avg = action_loss_total / denom
        entropy_avg = entropy_total / denom

        # 价值损失用最后一次 critic 计算的 value_loss
        # 返回对比损失数值方便外面写 TensorBoard / debug
        return value_loss.item(), action_loss_avg, entropy_avg, ctr_loss_raw.item()
