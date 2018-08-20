import matplotlib
matplotlib.use('Agg')
import pandas as pd
import gym
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam, SGD
from torch.distributions.categorical import Categorical
from plotnine import ggplot, aes, geom_line, stat_smooth, facet_wrap
from tensorboardX import SummaryWriter


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
    def __init__(self, obs_dim, action_dim, hidden_dim=100):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim, bias=False),
            nn.Softmax(dim=-1)
        )

    def forward(self, state_batch):
        return self.layers(state_batch)


class Baseline(nn.Module):
    def __init__(self, obs_dim, hidden_dim=100):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state_batch):
        return self.layers(state_batch)


class ReinforceAgent:
    def __init__(self, env, gamma=.99):
        self.model = ReinforceModel(env.observation_space.shape[0], env.action_space.n)
        self.baseline = Baseline(env.observation_space.shape[0])
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.baseline = self.baseline.cuda()
        self.gamma = gamma
        self.policy_optimizer = Adam(self.model.parameters(), lr=.01)
        self.baseline_optimizer = Adam(self.baseline.parameters(), lr=.01)

    def act(self, state, train=True):
        if train:
            self.baseline.train()
            self.model.train()
        else:
            self.baseline.eval()
            self.model.eval()
        state = torch.from_numpy(state).float()
        if torch.cuda.is_available():
            state = state.cuda()
        probs = self.model(state)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value_estimate = self.baseline(state)
        return action.item(), log_prob, value_estimate

    def update(self, policy_history, rewards, value_estimates, train=True, use_baseline=True):
        returns = torch.from_numpy(compute_return(rewards, self.gamma)).float()
        if torch.cuda.is_available():
            returns = returns.cuda()
        self.baseline.train()
        self.model.train()
        if use_baseline:
            value_diff = (returns - value_estimates)
        else:
            value_diff = returns
        value_diff_mean = value_diff.mean()
        loss = -(policy_history * value_diff).mean()
        if train:
            if use_baseline:
                self.baseline.zero_grad()
                value_diff_mean.backward(retain_graph=True)
                self.baseline_optimizer.step()

            self.model.zero_grad()
            loss.backward()
            self.policy_optimizer.step()

        return loss.item(), value_diff_mean.item()


def main():
    env = gym.make('CartPole-v1')
    # env = gym.wrappers.Monitor(env, '/tmp/cartpole', force=True)
    agent = ReinforceAgent(env)
    use_baseline = False
    writer = SummaryWriter()
    for episode in range(10000):
        observation = env.reset()
        rewards = []
        policy_history = []
        estimates = []
        done = False
        max_step = 0
        for step in range(1000):
            action, log_prob, value_estimate = agent.act(observation)
            estimates.append(value_estimate)
            policy_history.append(log_prob.view(1))
            observation, reward, done, _ = env.step(action)
            rewards.append(reward)
            max_step = step
            if done:
                break
        estimates = torch.cat(estimates)
        policy_history = torch.cat(policy_history)
        rewards = np.array(rewards)
        loss, v_diff = agent.update(policy_history, rewards, estimates, use_baseline=False)
        writer.add_scalar('reward', rewards.sum(), episode)
        writer.add_scalar('loss', loss, episode)
        writer.add_scalar('steps', max_step, episode)
        if use_baseline:
            writer.add_scalar('value_diff', v_diff, episode)
    env.close()


if __name__ == '__main__':
    main()
