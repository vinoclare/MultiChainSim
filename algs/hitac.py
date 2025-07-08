import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter


class HiTAC(nn.Module):
    def __init__(
            self,
            local_kpi_dim: int,
            global_kpi_dim: int,
            policies_info_dim: int,
            num_layers: int,
            num_subpolicies: int,
            hidden_dim: int = 64,
            n_heads: int = 4,
            transformer_layers: int = 2,
            clip_param: float = 0.2,
            entropy_coef: float = 0.001,
            max_grad_norm: float = 0.5,
            device: str = "cuda",
            total_steps: int = 5000000,
            lr: float = 3e-4,
            ucb_lambda: float = 0.2,
            sticky_prob: float = 0.1,
            update_epochs: int = 10,
            train_temp: float = 0.5,
            distill_temp: float = 0.3,
            epsilon: float = 0.1,
            greedy_prob: float = 0.5,
            writer: SummaryWriter = None
    ):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.num_layers = num_layers
        self.num_subpolicies = num_subpolicies
        self.clip_param = clip_param
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.total_steps = total_steps
        self.update_epochs = update_epochs
        self.train_temp = train_temp
        self.distill_temp = distill_temp
        self.epsilon = epsilon
        self.greedy_prob = greedy_prob
        self.writer = writer

        # === 编码器 ===
        self.local_embed = nn.Linear(local_kpi_dim, hidden_dim)
        self.global_embed = nn.Linear(global_kpi_dim, hidden_dim)
        self.layer_pos = nn.Embedding(num_layers, hidden_dim)

        self.policies_encoder = nn.Sequential(
            nn.Linear(num_subpolicies * policies_info_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=n_heads,
                batch_first=True,
                dim_feedforward=4 * hidden_dim,
                norm_first=True
            ),
            num_layers=transformer_layers
        )

        # === 输出层：每层 → K个子策略的logits ===
        self.act_head = nn.Linear(hidden_dim, num_subpolicies)
        self.value_head = nn.Linear(hidden_dim, 1)

        self.logits_norm = nn.LayerNorm(num_subpolicies)

        self.value_loss_coef = 0.5
        self.reward_coef = 0.2
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        self.ucb_lambda = ucb_lambda
        self.sticky_prob = sticky_prob

        self.last_pid = torch.zeros(2, self.num_layers, dtype=torch.long, device=self.device)
        self.freq_counter = torch.zeros(2, self.num_layers, self.num_subpolicies, device=self.device)

        self.progress = 0.0

        # === PPO缓存 ===
        self.old_log_probs = None
        self.old_logits = None
        self.old_actions = None

    def forward(self, local_kpis, global_kpi, policies_info):
        local_kpis = local_kpis.to(self.device)
        global_kpi = global_kpi.to(self.device)
        policies_info = policies_info.to(self.device)

        B, L, K, D = policies_info.shape
        g_emb = self.global_embed(global_kpi).unsqueeze(1)  # (B, 1, d)
        l_emb = self.local_embed(local_kpis)  # (B, L, d)
        layer_ids = torch.arange(L, device=self.device)
        pos_vec = self.layer_pos(layer_ids).unsqueeze(0)

        policy_feat = policies_info.reshape(B, L, K * D)  # (B, L, K×6)
        policy_emb = self.policies_encoder(policy_feat)  # (B, L, d)

        l_emb = l_emb + pos_vec + policy_emb

        tokens = torch.cat([g_emb, l_emb], dim=1)  # (B, L+1, d)
        encoded = self.encoder(tokens)  # (B, L+1, d)
        h = encoded[:, 1:, :]  # 取出每层的表示 (B, L, d)
        logits = self.act_head(h)  # (B, L, K)
        logits = self.logits_norm(logits)
        values = self.value_head(h).squeeze(-1)  # (B, L)
        return logits, values

    def select(self, local_kpis, global_kpi, policies_info, step):
        """
        推理接口：为每一层选择子策略
        Return:
          pids: LongTensor[B, L]
        """
        if torch.rand(1).item() < self.epsilon:
            pids = torch.randint(0, self.num_subpolicies, (self.num_layers,), device=self.device)
            for l in range(self.num_layers):
                self.freq_counter[0, l, pids[l]] += 1
            dummy_logits = torch.zeros((1, self.num_layers, self.num_subpolicies), device=self.device)
            self.store_for_update(dummy_logits, pids.unsqueeze(0))
            return pids

        raw_logits, _ = self.forward(local_kpis, global_kpi, policies_info)  # (B, L, K)
        logits = raw_logits[0]

        # === 计算 UCB 奖励 ===
        avg_rewards = policies_info[0, :, :, 0]
        freq = self.freq_counter[0] + 1.0
        ucb_bonus = torch.sqrt(torch.log(torch.tensor(step + 1.0, device=self.device)) / freq)  # (L, K)
        reward_coef = max(5e-3, self.reward_coef * np.exp(-10 * self.progress))
        logits = logits + self.ucb_lambda * ucb_bonus + reward_coef * avg_rewards

        probs = F.softmax(logits / self.train_temp, dim=-1)
        dist = Categorical(probs)
        pids = dist.sample()  # (L)

        # 更新状态
        for l in range(self.num_layers):
            self.freq_counter[0, l, pids[l]] += 1

        self.last_pid[0] = pids.detach()
        self.store_for_update(raw_logits.detach(), pids.unsqueeze(0).detach())

        # log 每个子策略被选择的次数
        print(f"Step: {step}")
        print(f"Current Pids: {pids.detach().cpu().numpy().tolist()}")
        for l in range(self.num_layers):
            counts = [int(self.freq_counter[0, l, k].item()) for k in range(self.num_subpolicies)]
            print(f"  Layer {l}: {counts}")

        return pids

    def select_distill(self, local_kpis, global_kpi, policies_info, step):
        logits, _ = self.forward(local_kpis, global_kpi, policies_info)  # (B, L, K)
        logits = logits[0]

        # === 计算 UCB 奖励 ===
        avg_rewards = policies_info[0, :, :, 0]
        logits = logits + self.reward_coef * avg_rewards

        probs = F.softmax(logits / self.distill_temp, dim=-1)

        if torch.rand(1).item() < self.greedy_prob:
            pids = torch.argmax(probs, dim=-1)
        else:
            dist = Categorical(probs)
            pids = dist.sample()  # (L)

        # —— Sticky 保留机制 ——
        mask = torch.rand_like(pids.float()) < self.sticky_prob  # (L,)
        pids = torch.where(mask, self.last_pid[1].to(pids.device), pids)

        # 更新状态
        for l in range(self.num_layers):
            self.freq_counter[1, l, pids[l]] += 1

        self.last_pid[1] = pids.detach()

        # log 每个子策略被选择的次数
        print(f"Select times (Distill):")
        print(f"Current Pids: {pids.detach().cpu().numpy().tolist()}")
        for l in range(self.num_layers):
            counts = [int(self.freq_counter[1, l, k].item()) for k in range(self.num_subpolicies)]
            print(f"  Layer {l}: {counts}")

        return pids

    def store_for_update(self, logits, actions):
        """
        存储用于训练的logits和actions
        """
        self.old_logits = logits.detach()  # (B, L, K)
        self.old_log_probs = F.log_softmax(self.old_logits, dim=-1)
        self.old_actions = actions.detach()  # (B, L)

    def update(self, local_kpis, global_kpi, policies_info, returns, step):
        stats_list = []
        for _ in range(self.update_epochs):
            logits, values = self.forward(local_kpis, global_kpi, policies_info)  # (B, L, K)
            new_log_probs = F.log_softmax(logits, dim=-1)  # (B, L, K)

            B, L, K = logits.shape
            old_lp = self.old_log_probs.gather(-1, self.old_actions.unsqueeze(-1).to(self.device)).squeeze(-1)  # (B, L)
            new_lp = new_log_probs.gather(-1, self.old_actions.unsqueeze(-1)).squeeze(-1)  # (B, L)

            # === 计算 ratio + clipped loss ===
            ratio = torch.exp(new_lp - old_lp)  # (B, L)
            returns = returns.to(self.device)
            advantages = returns - values.detach()
            adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * adv
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(values, returns)

            entropy = - (new_log_probs * new_log_probs.exp()).sum(dim=-1).mean()
            self.progress = step / self.total_steps
            entropy_coef = max(1e-4, self.entropy_coef * (1 - self.progress))

            loss = policy_loss + self.value_loss_coef * value_loss - entropy_coef * entropy

            # === 更新 ===
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
            self.optimizer.step()

            stats = {
                "hitac/policy_loss": policy_loss.item(),
                "hitac/value_loss": value_loss.item(),
                "hitac/entropy": entropy.item(),
                "hitac/total_loss": loss.item()
            }
            stats_list.append(stats)

        avg_stats = {
            k: np.mean([s[k] for s in stats_list])
            for k in stats_list[0].keys()
        }
        return avg_stats
