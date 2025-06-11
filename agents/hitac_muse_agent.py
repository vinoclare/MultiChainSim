# agents/hitac_muse_agent.py
import torch
from algs.muse import MuSE
from algs.hitac import HiTAC
from algs.distiller import Distiller


class HiTACMuSEAgent:
    def __init__(
            self,
            muse_cfg,
            hitac_cfg,
            distill_cfg,
            obs_spaces,
            act_spaces,
            num_layers,
            global_context_dim,
            device="cuda",
            writer=None,
            total_training_steps=None
    ):
        self.device = torch.device(device)
        self.num_layers = num_layers
        self.K = muse_cfg["K"]

        # === 各层 MuSE ===
        self.muses = [
            MuSE(muse_cfg, obs_spaces[l], device=device, writer=writer, total_training_steps=total_training_steps)
            for l in range(self.num_layers)
        ]

        # === 各层 Distiller（主策略） ===
        self.distillers = [
            Distiller(
                obs_spaces=obs_spaces[lid],
                global_context_dim=global_context_dim,
                hidden_dim=muse_cfg["hidden_dim"],
                act_dim=act_spaces[lid],
                device=device,
                sup_coef=distill_cfg["sup_coef"],
                neg_coef=distill_cfg["neg_coef"],
                margin=distill_cfg["margin"]
            )
            for lid in range(self.num_layers)
        ]

        # === HiTAC ===
        self.hitac = HiTAC(
            local_kpi_dim=hitac_cfg["local_kpi_dim"],
            global_kpi_dim=hitac_cfg["global_kpi_dim"],
            num_layers=self.num_layers,
            num_subpolicies=self.K,
            hidden_dim=hitac_cfg["hidden_dim"],
            n_heads=hitac_cfg["n_heads"],
            transformer_layers=hitac_cfg["transformer_layers"],
            clip_param=hitac_cfg["clip_param"],
            entropy_coef=hitac_cfg["entropy_coef"],
            max_grad_norm=hitac_cfg["max_grad_norm"],
            device=device
        ).to(device)

    def select_subpolicies(self, local_kpis, global_kpi, greedy=False):
        """
        Args:
          local_kpis: Tensor (B, L, d_l)
          global_kpi: Tensor (B, d_g)
        Return:
          pid: Tensor (B, L)
        """
        return self.hitac.select(local_kpis, global_kpi, greedy=greedy)

    @torch.no_grad()
    def sample(self, obs_dicts, pids):
        """
        输入为各层 observation；返回各层动作等，pids 为当前策略选择。
        Return: Dict[layer_id → dict{values, actions, logp, ent, pid}]
        """
        output = {}

        for lid in range(self.num_layers):
            muse = self.muses[lid]
            obs = obs_dicts[lid]
            v_u, v_c, actions, logp, ent = muse.sample(
                obs["task_obs"], obs["worker_loads"],
                obs["worker_profile"], obs["global_context"], obs["valid_mask"], pids[:, lid]
            )
            output[lid] = {
                "v_u": v_u, "v_c": v_c, "actions": actions,
                "logp": logp, "ent": ent, "pid": pids[:, lid]
            }

        return output

    def muse_learn(self, layer_id, step, data):
        muse = self.muses[layer_id]
        v_loss, pi_loss, ent = muse.learn(
            data["pid"], data["task_obs"], data["worker_loads"],
            data["worker_profile"], data["global_context"], data["valid_mask"],
            data["actions"], data["returns"], data["logp_old"], data["advantages"], step
        )
        return {
            f"layer_{layer_id}/v_loss": v_loss,
            f"layer_{layer_id}/pi_loss": pi_loss,
            f"layer_{layer_id}/entropy": ent,
        }

    def distill_collect(self, layer_id, buffer_obs, buffer_actions, buffer_pid):
        self.distillers[layer_id].collect(buffer_obs, buffer_actions, buffer_pid)

    def distill_update(self, layer_id, steps=300):
        return self.distillers[layer_id].bc_update(steps)

    def main_policy_predict(self, layer_id, obs_dict):
        return self.distillers[layer_id].predict(obs_dict)

    def hitac_update(self, local_kpis, global_kpi, pid, advantage, lr=None):
        """
        存储 + PPO 更新 HiTAC
        """
        logits = self.hitac.forward(local_kpis, global_kpi)
        self.hitac.store_for_update(logits, pid, advantage)
        return self.hitac.update(local_kpis, global_kpi, pid, advantage, lr)
