from collections import namedtuple, deque
import numpy as np
import random

Experience = namedtuple(
    "Experience",
    field_names=["observation", "action", "reward", "next_observation", "done"],
)


class ReplayBuffer:
    def __init__(self, buffer_size: int):
        self.buffer = deque(maxlen=buffer_size)

    def append(self, experience: Experience):
        self.buffer.append(experience)

    def sample(self, sample_size: int):
        observations, actions, rewards, next_observations, dones = zip(*(random.sample(self.buffer, sample_size)))

        return (
            np.array(observations),
            np.array(actions),
            np.array(rewards, dtype=float),
            np.array(next_observations),
            np.array(dones, dtype=bool)
        )

    def __len__(self):
        return len(self.buffer)