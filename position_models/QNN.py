import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

# Define Q-learning agent
class QLearningAgent:
    def __init__(self, state_size, action_size, hidden_layers, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=2000)

        # Q-Network
        self.q_network = FlexibleNeuralNet(state_size, action_size, hidden_layers)
        self.target_network = FlexibleNeuralNet(state_size, action_size, hidden_layers)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # Random action
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return torch.argmax(q_values).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
            target = reward
            if not done:
                target += self.gamma * torch.max(self.target_network(next_state_tensor)).item()
            
            q_values = self.q_network(state_tensor)
            target_f = q_values.clone().detach()
            target_f[0][action] = target

            # Update Q-network
            self.optimizer.zero_grad()
            loss = self.criterion(q_values, target_f)
            loss.backward()
            self.optimizer.step()

        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

# Example usage
def main():
    state_size = 4  # Example state size
    action_size = 2  # Example action space
    agent = QLearningAgent(state_size, action_size, hidden_layers=[64, 64])

    episodes = 1000
    batch_size = 32

    for e in range(episodes):
        state = np.random.rand(state_size)  # Replace with environment reset
        total_reward = 0

        for time in range(200):  # Replace with environment max steps
            action = agent.act(state)
            next_state = np.random.rand(state_size)  # Replace with environment step
            reward = random.random()  # Replace with environment reward
            done = random.choice([True, False])  # Replace with environment done
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if done:
                agent.update_target_network()
                print(f"Episode {e+1}/{episodes}, Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")
                break

        agent.replay(batch_size)

if __name__ == "__main__":
    main()
