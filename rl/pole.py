import pandas as pd
import gym
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam, SGD
from torch.distributions.categorical import Categorical
from plotnine import ggplot, aes, geom_line, stat_smooth, facet_wrap


def compute_return(rewards, gamma, normalize=True):
    returns = np.zeros(len(rewards), dtype=np.float32)
    discounted = 0
    i = len(rewards) - 1
    while i >= 0:
        discounted = gamma * discounted + rewards[i]
        returns[i] = discounted
        i -= 1
    if np.isnan(rewards).any():
        raise ValueError(f'NAN Returns: {returns}')
    if normalize:
        return (returns - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
    else:
        return returns


class ReinforceModel(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim, bias=False),
            #nn.Dropout(p=.6),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim, bias=False),
            nn.Softmax(dim=-1)
        )

    def forward(self, state_batch):
        return self.layers(state_batch)


class ReinforceAgent:
    def __init__(self, env, gamma=.99):
        self.model = ReinforceModel(env.observation_space.shape[0], env.action_space.n)
        self.gamma = gamma
        self.optimizer = Adam(self.model.parameters(), lr=.01)

    def act(self, state, train=True):
        if train:
            self.model.train()
        else:
            self.model.eval()
        state = torch.from_numpy(state).float()
        probs = self.model(state)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob

    def update(self, policy_history, rewards, train=True):
        returns = torch.from_numpy(compute_return(rewards, self.gamma)).float()
        self.model.train()
        loss = -(policy_history * returns).sum()
        if train:
            self.model.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss.item()


def main():
    env = gym.make('CartPole-v1')
    env = gym.wrappers.Monitor(env, '/tmp/cartpole', force=True)
    agent = ReinforceAgent(env)
    episode_records = {'episode': [], 'value': [], 'name': []}
    for episode in range(1100):
        train = episode < 1000
        observation = env.reset()
        rewards = []
        policy_history = []
        done = False
        max_step = 0
        for step in range(1000):
            action, log_prob = agent.act(observation, train=train)
            policy_history.append(log_prob.view(1))
            observation, reward, done, _ = env.step(action)
            rewards.append(reward)
            max_step = step
            if done:
                break
        policy_history = torch.cat(policy_history)
        rewards = np.array(rewards)
        loss = agent.update(policy_history, rewards, train=train)
        episode_records['value'].append(rewards.sum())
        episode_records['name'].append('reward')
        episode_records['episode'].append(episode)

        episode_records['value'].append(loss)
        episode_records['name'].append('loss')
        episode_records['episode'].append(episode)

        episode_records['value'].append(max_step)
        episode_records['name'].append('steps')
        episode_records['episode'].append(episode)

    # Finalize and plot stuff
    env.close()
    df = pd.DataFrame(episode_records)
    g = (
        ggplot(df) + aes(x='episode', y='value') + facet_wrap('name', scales='free', nrow=3)
        + geom_line() #+ stat_smooth(method='mavg')
    ).draw()
    g.show()
    input('Enter to exit')



if __name__ == '__main__':
    main()
