import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# Define the Q-Network
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Define the Trading Environment
class TradingEnv:
    def __init__(self, data, window_size, transaction_cost=0.001):
        self.data = data
        self.window_size = window_size
        self.transaction_cost = transaction_cost  # Cost as a fraction of trade value
        self.action_space = [-1, 0, 1]  # Short, Neutral, Long
        self.current_step = window_size
        self.done = False
        self.position = 0  # Current position: -1 (short), 0 (neutral), 1 (long)

    def reset(self):
        self.current_step = self.window_size
        self.done = False
        self.position = 0
        return self._get_state()

    def _get_state(self):
        return self.data[self.current_step - self.window_size : self.current_step]

    def step(self, action):
        price_now = self.data[self.current_step]
        price_next = self.data[self.current_step + 1]

        # Reward based on price change and position
        reward = self.action_space[action] * (price_next - price_now)

        # Apply transaction costs if there's a position change
        if self.position != self.action_space[action]:
            reward -= abs(self.action_space[action] - self.position) * self.transaction_cost * price_now

        # Update position and step
        self.position = self.action_space[action]
        self.current_step += 1
        self.done = self.current_step >= len(self.data) - 1

        return self._get_state(), reward, self.done

# DQN Agent
class DQNAgent:
    def __init__(self, state_dim, action_dim, gamma=0.99, lr=1e-4, batch_size=32, memory_size=10000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size

        # Q-Networks
        self.policy_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer and Replay Buffer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = deque(maxlen=memory_size)

    def act(self, state, epsilon):
        if random.random() < epsilon:
            return random.choice(range(self.action_dim))
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        # Sample a batch
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # Q-Learning Target
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            targets = rewards + self.gamma * next_q_values * (1 - dones)

        # Compute Loss and Backpropagate
        loss = nn.MSELoss()(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

# Training Loop
def train_trading_agent(data, episodes=100, window_size=30, epsilon_decay=0.995, min_epsilon=0.1, transaction_cost=0.001):
    env = TradingEnv(data, window_size, transaction_cost)
    agent = DQNAgent(state_dim=window_size, action_dim=3)
    epsilon = 1.0

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0

        while True:
            action = agent.act(state, epsilon)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.train()
            state = next_state
            total_reward += reward

            if done:
                break

        # Decay epsilon and update target network
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        agent.update_target_network()

        print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward:.2f}")

# Example usage
if __name__ == "__main__":
    # Dummy price data
    np.random.seed(42)
    price_data = np.cumsum(np.random.randn(1000)) + 100  # Simulated price data
    train_trading_agent(price_data)
