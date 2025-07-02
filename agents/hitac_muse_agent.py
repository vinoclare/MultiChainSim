# agents/hitac_muse_agent.py
import torch
from torch.distributions import Categorical

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
        self.num_pos_subpolicies = self.K - 2 if distill_cfg["neg_policy"] else self.K
        self.distill_batch_size = distill_cfg["batch_size"]
        self.bc_steps = distill_cfg["bc_steps"]

        # === 各层 MuSE ===
        self.muses = [
            MuSE(muse_cfg, distill_cfg, obs_spaces[l], device=device, writer=writer, total_training_steps=total_training_steps)
            for l in range(self.num_layers)
        ]

        # === 各层 Distiller（主策略） ===
        self.distillers = [
            Distiller(
                obs_spaces=obs_spaces[lid],
                global_context_dim=global_context_dim,
                hidden_dim=distill_cfg["hidden_dim"],
                act_dim=act_spaces[lid],
                K=self.K,
                loss_type=distill_cfg["loss_type"],
                neg_policy=distill_cfg["neg_policy"],
                device=device,
                sup_coef=distill_cfg["sup_coef"],
                neg_coef=distill_cfg["neg_coef"],
                margin=distill_cfg["margin"],
                std_t=distill_cfg["std_t"]
            )
            for lid in range(self.num_layers)
        ]

        # === HiTAC ===
        self.hitac = HiTAC(
            local_kpi_dim=hitac_cfg["local_kpi_dim"]+self.num_pos_subpolicies,
            global_kpi_dim=hitac_cfg["global_kpi_dim"],
            policies_info_dim=hitac_cfg["policies_info_dim"],
            num_layers=self.num_layers,
            num_subpolicies=self.num_pos_subpolicies,
            hidden_dim=hitac_cfg["hidden_dim"],
            n_heads=hitac_cfg["n_heads"],
            transformer_layers=hitac_cfg["transformer_layers"],
            clip_param=hitac_cfg["clip_param"],
            entropy_coef=hitac_cfg["entropy_coef"],
            max_grad_norm=hitac_cfg["max_grad_norm"],
            device=device,
            total_steps=total_training_steps,
            lr=hitac_cfg["lr"],
            ucb_lambda=hitac_cfg["ucb_lambda"],
            sticky_prob=hitac_cfg["sticky_prob"],
            update_epochs=hitac_cfg["update_epochs"],
            temperature=hitac_cfg["temperature"],
            epsilon=hitac_cfg["epsilon"],
            greedy_prob=hitac_cfg["greedy_prob"],
            writer=writer
        ).to(device)

    def select_subpolicies(self, local_kpis, global_kpi, policies_info, step):
        return self.hitac.select(local_kpis, global_kpi, policies_info, step)

    def select_subpolicies_distill(self, local_kpis, global_kpi, policies_info, step):
        return self.hitac.select_distill(local_kpis, global_kpi, policies_info, step)

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
            v_u, v_c, actions, logp, ent, act_std = muse.sample(
                obs["task_obs"], obs["worker_loads"],
                obs["worker_profile"], obs["global_context"], obs["valid_mask"], pids[lid]
            )
            output[lid] = {
                "v_u": v_u, "v_c": v_c, "actions": actions,
                "logp": logp, "ent": ent, "act_std": act_std, "pid": pids[lid]
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

    def distill_update(self, layer_id, cur_pid):
        return self.distillers[layer_id].bc_update(cur_pid, self.distill_batch_size, self.bc_steps)

    def main_policy_predict(self, layer_id, obs_dict):
        return self.distillers[layer_id].predict(obs_dict)

    def hitac_update(self, local_kpis, global_kpi, policies_info, returns, step):
        return self.hitac.update(local_kpis, global_kpi, policies_info, returns, step)
