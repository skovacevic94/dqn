import gym
from collections import deque
#import matplotlib.pyplot as plt
from agent import Agent


if __name__ == '__main__':
    checkpoint_file = './models/model.pkl'
    env = gym.make("LunarLander-v2")

    agent = Agent.load(checkpoint_file)

    values = deque(maxlen=1000)
    for _ in range(1000):
        observation = env.reset()
        action = agent.act(observation)
        done = False
        total_reward = 0
        while not done:
            env.render()
            observation, reward, done, info = env.step(action)
            value = agent.value(observation)
            values.append(value)
            action = agent.act(observation)
            total_reward += reward
            #plt.plot(values)
            #plt.clf()
        print(total_reward)
