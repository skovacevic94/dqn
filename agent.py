import copy

import numpy as np
import torch
from torch import nn

from qnet import QNet
from replay_buffer import ReplayBuffer, Experience


class Agent:
    def __init__(self, args):
        self.observation_space_dim = args['observation_space_dim']
        self.action_space_dim = args['action_space_dim']

        self.q_online = QNet(args['observation_space_dim'], args['hidden_dim'], args['action_space_dim'])
        self.q_target = copy.deepcopy(self.q_online)
        for p in self.q_target.parameters():
            p.requires_grad = False
        self.q_target.eval()

        self.optimizer = torch.optim.Adam(self.q_online.parameters(), lr=args['step_size'])
        self.loss_fn = nn.MSELoss()

        self.memory = ReplayBuffer(args['buffer_size'])
        self.BATCH_SIZE = args['batch_size']

        self.PARAM_UPDATE_PERIOD = args['param_update_period']
        self.LEARNING_PERIOD = args['learning_period']

        self.epsilon = args['epsilon_start']
        self.EPSILON_MIN = args['epsilon_min']
        self.EPSILON_DECAY_RATE = args['epsilon_decay_rate']
        self.GAMMA = args['discount_factor']

        self.current_step = 0

    def value(self, observations):
        with torch.no_grad():
            o = observations.reshape((-1, self.observation_space_dim))
            q_values = self.q_online(torch.tensor(o, dtype=torch.float)).detach().numpy()
            probs = self.policy(o)
            return np.sum(np.multiply(q_values, probs), axis=1)

    def policy(self, observations):
        with torch.no_grad():
            o = observations.reshape((-1, self.observation_space_dim))
            probs = np.ones((o.shape[0], self.action_space_dim), dtype=float) * (self.epsilon / self.action_space_dim)
            q_values = self.q_online(torch.tensor(o, dtype=torch.float)).detach().numpy()
            best_actions = np.argmax(q_values, axis=1)
            probs[range(probs.shape[0]), best_actions] += (1. - self.epsilon)
            return probs

    def act(self, observation):
        return np.random.choice(self.action_space_dim, p=self.policy(observation)[0])

    def step(self, experience: Experience):
        self.memory.append(experience)

        self.epsilon *= self.EPSILON_DECAY_RATE
        self.epsilon = max(self.EPSILON_MIN, self.epsilon)

        self.current_step += 1
        metadata = None
        if self.current_step % self.LEARNING_PERIOD == 0 and len(self.memory) > self.BATCH_SIZE:
            metadata = self.__learn()
        if self.current_step % self.PARAM_UPDATE_PERIOD == 0:
            self.q_target.load_state_dict(self.q_online.state_dict())

        return metadata

    def __learn(self) -> dict:
        observations, actions, rewards, next_observations, dones = self.memory.sample(self.BATCH_SIZE)
        observations = torch.tensor(observations, dtype=torch.float)
        actions = torch.LongTensor(actions).unsqueeze(-1)
        rewards = torch.tensor(rewards, dtype=torch.float)
        next_observations = torch.tensor(next_observations, dtype=torch.float)

        action_values = self.q_online(observations).gather(1, actions).squeeze(1)
        with torch.no_grad():
            next_state_values = self.q_target(next_observations).max(1)[0]
            next_state_values[dones] = 0.0
            next_state_values = next_state_values.detach()
            targets = rewards + self.GAMMA * next_state_values
        loss = self.loss_fn(action_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            'loss': loss
        }
    
    def save(self, checkpoint_filename):
        torch.save(self, checkpoint_filename)

    @staticmethod
    def load(checkpoint_filename):
        return torch.load(checkpoint_filename)
