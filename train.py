import gym
from agent import *
import tqdm
from collections import deque

from torch.utils.tensorboard import SummaryWriter

def sample_holdout_observations(env, sample_size):
    holdout_observations = np.zeros((sample_size, env.observation_space.shape[0]))
    for i in range(sample_size):
        holdout_observations[i, :] = env.reset()
    return holdout_observations

def evaluate_holdout_observations(agent: Agent, holdout_observations):
    state_values = agent.value(holdout_observations)
    return np.mean(state_values)

if __name__ == '__main__':
    env = gym.make("LunarLander-v2")
    checkpoint_file = "./models/model.pkl"
    agent_params = {
        "observation_space_dim": env.observation_space.shape[0],
        "hidden_dim": [64, 64],
        "action_space_dim": env.action_space.n,
        "buffer_size": int(1e6),
        "step_size": 5e-4,
        "batch_size": 64,
        "param_update_period": 1000,
        "learning_period": 5,
        "epsilon_start": 1.0,
        "epsilon_min": 0.01,
        "epsilon_decay_rate": 0.9995,
        "discount_factor": 0.99
    }

    writer = SummaryWriter()
    agent = Agent(agent_params)

    holdout_states = sample_holdout_observations(env, 128)

    global_step = 0
    scores_window = deque(maxlen=100)
    for episode in tqdm.tqdm(range(2000)):
        observation = env.reset()
        total_reward = 0
        while True:
            action = agent.act(observation)
            
            observation_, reward, done, info = env.step(action)
            
            experience = Experience(observation, action, reward, observation_, done)
            metadata = agent.step(experience)
            if metadata is not None:
                writer.add_scalar("loss", metadata['loss'], global_step)
            
            observation = observation_
            total_reward += reward
            global_step += 1
            if done:
                break
        
        average_holdout_value = evaluate_holdout_observations(agent, holdout_states)
        writer.add_scalar("avg_holdout_value", average_holdout_value, episode)
        scores_window.append(total_reward)
        if episode > 100:
            mean_100_score = np.mean(scores_window)
            writer.add_scalar("score", mean_100_score, episode)
            if mean_100_score > 200:
                break
    agent.save(checkpoint_file)
