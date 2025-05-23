import torch
import torch.nn.functional as F
from torch.distributions import Normal
from copy import deepcopy


class SAC:
    def __init__(self,
                 model,
                 gamma=0.99,
                 tau=0.005,
                 alpha=0.2,
                 actor_lr=3e-4,
                 critic_lr=3e-4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.target_model = deepcopy(self.model).to(self.device)

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        self.actor_optimizer = torch.optim.Adam(
            self.model.get_actor_params(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(
            self.model.get_critic_params(), lr=critic_lr)

    def predict(self, task_obs, worker_obs):
        task_obs = task_obs.to(self.device)
        worker_obs = worker_obs.to(self.device)
        act_mean, _ = self.model.policy(task_obs, worker_obs)
        action = torch.tanh(act_mean)
        return action

    def sample(self, task_obs, worker_obs):
        task_obs = task_obs.to(self.device)
        worker_obs = worker_obs.to(self.device)

        act_mean, log_std = self.model.policy(task_obs, worker_obs)
        std = log_std.exp()
        normal = Normal(act_mean, std)

        x_t = normal.rsample()
        action = torch.tanh(x_t)

        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log((1 - action.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob

    def learn(self, task_obs, worker_obs, action, reward, next_task_obs, next_worker_obs, terminal):
        critic_loss = self._critic_learn(task_obs, worker_obs, action, reward,
                                         next_task_obs, next_worker_obs, terminal)
        actor_loss = self._actor_learn(task_obs, worker_obs)

        self.sync_target()
        return critic_loss.item(), actor_loss.item()

    def _critic_learn(self, task_obs, worker_obs, action, reward, next_task_obs, next_worker_obs, terminal):
        with torch.no_grad():
            next_action, next_logprob = self.sample(next_task_obs, next_worker_obs)
            q1_next, q2_next = self.target_model.value(next_task_obs, next_worker_obs, next_action)
            target_q = torch.min(q1_next, q2_next) - self.alpha * next_logprob
            target_q = reward + self.gamma * (1 - terminal) * target_q

        q1, q2 = self.model.value(task_obs, worker_obs, action)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        return critic_loss

    def _actor_learn(self, task_obs, worker_obs):
        act, log_pi = self.sample(task_obs, worker_obs)
        q1_pi, q2_pi = self.model.value(task_obs, worker_obs, act)
        min_q = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha * log_pi - min_q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        return actor_loss

    def sync_target(self, decay=None):
        if decay is None:
            decay = 1 - self.tau
        for param, target_param in zip(self.model.parameters(), self.target_model.parameters()):
            target_param.data.copy_((1 - decay) * param.data + decay * target_param.data)
