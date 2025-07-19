import torch


class FACMACAgent:
    def __init__(self, alg, device="cpu", expl_noise=0.1):
        self.alg = alg
        self.device = device
        self.expl_noise = expl_noise

    @torch.no_grad()
    def select_action(self, task_obs, load_obs, profile_obs, eval=False):
        t = lambda x: torch.tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)
        task, load, prof = t(task_obs), t(load_obs), t(profile_obs)

        model = self.alg.target if eval else self.alg.model
        act = model.get_actions(task, load, prof,
                                deterministic=eval,
                                noise_std=0.0 if eval else self.expl_noise)
        return act.squeeze(0).cpu().numpy()
