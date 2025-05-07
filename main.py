import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import argparse
import matplotlib.pyplot as plt

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)

def train(env_name='LunarLander-v3', gamma=0.99, lr=0.01, max_episodes=1000, batch_size=5):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy = PolicyNetwork(state_dim, action_dim)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    batch_log_probs = []
    batch_returns = []
    episode_rewards = []

    for episode in range(1, max_episodes+1):
        state, _ = env.reset()
        rewards = []
        log_probs = []

        while True:
            state_tensor = torch.FloatTensor(state)
            action_probs = policy(state_tensor)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            log_probs.append(log_prob)
            rewards.append(reward)
            state = next_state

            if done:
                break

        # 計算 discounted return
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        batch_log_probs.extend(log_probs)
        batch_returns.extend(returns)
        episode_rewards.append(sum(rewards))  # <--- 儲存總回饋

        if (episode + 1) % batch_size == 0:
            log_probs_tensor = torch.stack(batch_log_probs)
            returns_tensor = torch.stack(batch_returns)

            policy_loss = (-log_probs_tensor * returns_tensor).mean()

            optimizer.zero_grad()
            policy_loss.backward()
            optimizer.step()

            batch_log_probs = []
            batch_returns = []

        if episode % (batch_size * 20) == 0:
            print(f'Episode {episode}, Total reward: {sum(rewards)}')

        os.makedirs('./checkpoints', exist_ok=True)
        if episode % 100 == 0:        
            torch.save(policy.state_dict(), f'./checkpoints/policy_model-{episode}.pth')

    # 繪製訓練曲線
    plt.figure(figsize=(12, 6))
    plt.plot(episode_rewards, label='Episode Reward')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Policy Gradient Training - Total Reward per Episode')
    plt.legend()
    plt.grid()
    plt.savefig('./checkpoints/training_curve.png')
    plt.show()

    env.close()
    return policy

def evaluate(policy, env_name='LunarLander-v3', render=False):
    env = gym.make(env_name, render_mode="human" if render else None)
    state, _ = env.reset()
    total_reward = 0
    while True:
        state_tensor = torch.FloatTensor(state)
        with torch.no_grad():
            action_probs = policy(state_tensor)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()

        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        state = next_state

        total_reward += reward

        if done:
            if total_reward >= 200:
                print("successful,\t", f"reward: {total_reward:.2f}")
            elif 50 <= total_reward:
                print("land,\t", f"reward: {total_reward:.2f}")
            else:
                print("fail,\t", f"reward: {total_reward:.2f}")
            state, _ = env.reset()
            total_reward = 0

def parse_args():
    parser = argparse.ArgumentParser(description='LunarLander Policy Gradient')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to a saved model')
    parser.add_argument('--batch', type=int, default=5, help='batch size')
    parser.add_argument('--render', action='store_true', help='Render evaluation')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    env_name = 'LunarLander-v3'

    if args.checkpoint:
        # 載入並 evaluate
        env = gym.make(env_name)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        policy = PolicyNetwork(state_dim, action_dim)
        policy.load_state_dict(torch.load(args.checkpoint))
        policy.eval()

        evaluate(policy, env_name, args.render)
    else:
        policy = train(env_name, args.gamma, args.lr, args.episodes, args.batch)
        print("Training complete. Saved to ./checkpoints/policy_model.pth")
