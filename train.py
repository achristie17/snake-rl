import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from snake.snake_game import Snake
import os
import time

# Define a simple policy network
class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = F.relu(self.fc(state))
        x = F.relu(self.fc2(x))
        action_probs = F.softmax(self.fc3(x), dim=-1)
        return action_probs

# Function to calculate discounted rewards
def calculate_discounted_rewards(rewards, gamma=1):
    discounted_rewards = []
    running_add = 0
    for r in reversed(rewards):
        running_add = running_add * gamma + r
        discounted_rewards.insert(0, running_add)
    return discounted_rewards

# Function to choose actions based on policy probabilities
def choose_action(policy_net, state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    action_probs = policy_net(state)
    action = np.random.choice(np.arange(len(action_probs.squeeze())), p=action_probs.squeeze().detach().numpy())
    return action

# Function to train the policy network using REINFORCE
def train_policy_network(policy_net, optimizer, states, actions, rewards):
    optimizer.zero_grad()

    # Convert lists to PyTorch tensors
    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.int64)
    discounted_rewards = calculate_discounted_rewards(rewards)
    discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)

    # Forward pass
    action_probs = policy_net(states)
    selected_action_probs = action_probs.gather(1, actions.view(-1, 1))

    # Calculate loss using REINFORCE objective
    loss = -torch.sum(torch.log(selected_action_probs) * discounted_rewards)

    # Backward pass
    loss.backward()
    optimizer.step()

# Main training loop
def train_policy_gradient(num_episodes=5000, max_steps=1000, board_size=10, model_path='policy_model.pth'):
    game = Snake(board_size)
    state = game.get_state()
    state_size = 8
    action_size = 4
    returns = []

    policy_net = PolicyNetwork(state_size, action_size)
    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)

    max_reward = 0
    for episode in range(num_episodes):
        game = Snake(board_size)
        state = game.get_state()
        states, actions, rewards = list(), list(), list()

        for step in range(max_steps):
            # Choose action
            action = choose_action(policy_net, state)

            # Take action and observe next state and reward
            next_state, reward, done = game.step(action)
            # if (episode > 4500):
            #     print(game)
            #     time.sleep(0.1)

            # Store state, action, and reward
            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state

            if done:
                break

        # Train the policy network
        train_policy_network(policy_net, optimizer, states, actions, rewards)

        # Print episode details
        total_reward = sum(rewards)
        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}")
        returns.append(total_reward)
        if total_reward > max_reward:
            torch.save(policy_net, model_path)
            max_reward = total_reward

    return returns