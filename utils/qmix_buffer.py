import random
import numpy as np


class QMixReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, transition):
        """
        transition: dict with keys:
            - task_obs: (n_agents, D1)
            - load_obs: (n_agents, D2)
            - profile_obs: (n_agents, D3)
            - state: (D_state,)
            - reward: float
            - next_task_obs, next_load_obs, next_profile_obs, next_state
            - done: bool
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        keys = batch[0].keys()
        out = {k: [] for k in keys}
        for transition in batch:
            for k in keys:
                out[k].append(transition[k])
        # convert to np.array
        for k in out:
            out[k] = np.array(out[k])
        return out

    def __len__(self):
        return len(self.buffer)
