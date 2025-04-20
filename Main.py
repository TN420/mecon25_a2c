import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque

# ================================
# Configuration Constants
# ================================

# Environment Constants
TOTAL_PRBS = 100
URLLC_QUOTA = 30
EMBB_QUOTA = 70

# Traffic Types
TRAFFIC_TYPES = ['URLLC', 'eMBB']

# A2C Hyperparameters
GAMMA = 0.95
LR = 0.001
BATCH_SIZE = 16
MEMORY_SIZE = 5000
TARGET_UPDATE = 10
EPISODES = 300
STEPS_PER_EPISODE = 50
EPSILON_START = 0.9
EPSILON_END = 0.05
EPSILON_DECAY = 100

# ================================
# Environment
# ================================

class RANEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.urllc_usage = 0
        self.embb_usage = 0
        self.total_prbs = TOTAL_PRBS
        self.state = self._get_state()
        return self.state

    def _get_state(self):
        return np.array([
            self.urllc_usage / TOTAL_PRBS,
            self.embb_usage / TOTAL_PRBS,
            1.0,
            1.0
        ], dtype=np.float32)

    def step(self, action, traffic_type):
        reward = 0
        done = False
        admitted = False
        blocked = False
        sla_violated = False

        if traffic_type == 'URLLC':
            request_prbs = np.random.randint(1, 5)
        else:
            request_prbs = np.random.randint(5, 50)

        if action == 1:
            if traffic_type == 'URLLC':
                if self.urllc_usage + request_prbs <= URLLC_QUOTA:
                    self.urllc_usage += request_prbs
                    reward = 1
                    admitted = True
                else:
                    reward = -1
                    blocked = True
                    sla_violated = True
            elif traffic_type == 'eMBB':
                if self.embb_usage + request_prbs <= EMBB_QUOTA:
                    self.embb_usage += request_prbs
                    reward = 1
                    admitted = True
                elif self.urllc_usage + self.embb_usage + request_prbs <= TOTAL_PRBS:
                    self.embb_usage += request_prbs
                    reward = -0.5
                    admitted = True
                    sla_violated = True
                else:
                    reward = -1
                    blocked = True
                    sla_violated = True
        else:
            reward = 0.1 if not admitted else -0.2

        next_state = self._get_state()
        return next_state, reward, done, admitted, blocked, sla_violated, traffic_type

# ================================
# A2C Networks
# ================================

class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, 16)
        self.fc2 = nn.Linear(16, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.softmax(self.fc2(x), dim=-1)

class Critic(nn.Module):
    def __init__(self, state_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# ================================
# Utility Functions
# ================================

def smooth(data, window=10):
    return np.convolve(data, np.ones(window)/window, mode='valid')

# ================================
# Training Loop
# ================================

def train_a2c(episodes=EPISODES):
    env = RANEnv()
    state_size = len(env.reset())
    action_size = 2

    actor = Actor(state_size, action_size)
    critic = Critic(state_size)
    optimizer = optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=LR)

    reward_history = []
    urllc_block_history = []
    embb_block_history = []
    urllc_sla_pres = []
    embb_sla_pres = []
    urllc_usage_hist = []
    embb_usage_hist = []
    actor_losses = []
    critic_losses = []

    for episode in range(episodes):
        state = env.reset()
        episode_return = 0
        urllc_blocks = 0
        embb_blocks = 0
        urllc_sla_preserved = 0
        embb_sla_preserved = 0
        urllc_total_requests = 0
        embb_total_requests = 0

        for t in range(STEPS_PER_EPISODE):
            traffic_type = random.choice(TRAFFIC_TYPES)
            if traffic_type == 'URLLC':
                urllc_total_requests += 1
            else:
                embb_total_requests += 1

            state_tensor = torch.tensor(state).float().unsqueeze(0)
            action_probs = actor(state_tensor)
            action = np.random.choice(action_size, p=action_probs.detach().numpy().squeeze())

            next_state, reward, done, admitted, blocked, sla_violated, t_type = env.step(action, traffic_type)

            if blocked:
                if t_type == 'URLLC': urllc_blocks += 1
                if t_type == 'eMBB': embb_blocks += 1
            if not sla_violated:
                if t_type == 'URLLC': urllc_sla_preserved += 1
                if t_type == 'eMBB': embb_sla_preserved += 1

            state_value = critic(state_tensor)
            next_state_value = critic(torch.tensor(next_state).float().unsqueeze(0))
            advantage = reward + GAMMA * next_state_value - state_value

            actor_loss = -torch.log(action_probs.squeeze(0)[action]) * advantage.detach()
            critic_loss = advantage.pow(2)

            optimizer.zero_grad()
            (actor_loss + critic_loss).mean().backward()
            optimizer.step()

            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())

            state = next_state
            episode_return += reward

        reward_history.append(episode_return)
        urllc_block_history.append(urllc_blocks)
        embb_block_history.append(embb_blocks)
        urllc_sla_pres.append(urllc_sla_preserved / urllc_total_requests if urllc_total_requests > 0 else 0)
        embb_sla_pres.append(embb_sla_preserved / embb_total_requests if embb_total_requests > 0 else 0)
        urllc_usage_hist.append(env.urllc_usage)
        embb_usage_hist.append(env.embb_usage)

        print(f"Episode {episode+1}: Episode Return = {episode_return:.2f}, URLLC SLA Ratio = {urllc_sla_pres[-1]:.2f}, eMBB SLA Ratio = {embb_sla_pres[-1]:.2f}")

    # Plot performance
    fig, axs = plt.subplots(3, 2, figsize=(14, 10))

    axs[0, 0].plot(smooth(reward_history))
    axs[0, 0].set_title("Smoothed Episode Return")

    axs[0, 1].plot(smooth(urllc_block_history), label="URLLC")
    axs[0, 1].plot(smooth(embb_block_history), label="eMBB")
    axs[0, 1].set_title("Smoothed Block Rate")
    axs[0, 1].legend()

    axs[1, 0].plot(smooth(urllc_sla_pres), label="URLLC")
    axs[1, 0].plot(smooth(embb_sla_pres), label="eMBB")
    axs[1, 0].set_ylim(0, 1.05)
    axs[1, 0].set_title("Smoothed SLA Preservation Ratio")
    axs[1, 0].legend()

    axs[1, 1].axis('off')

    axs[2, 0].plot(smooth(urllc_usage_hist))
    axs[2, 0].set_title("Smoothed URLLC PRB Usage")

    axs[2, 1].plot(smooth(embb_usage_hist))
    axs[2, 1].set_title("Smoothed eMBB PRB Usage")

    for ax in axs.flat:
        ax.set_xlabel("Episode")
        ax.grid(True)

    plt.tight_layout()
    plt.savefig("a2c_performance_plots.png")
    plt.close()

    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.plot(actor_losses, label="Actor Loss", alpha=0.7)
    plt.plot(critic_losses, label="Critic Loss", alpha=0.7)
    plt.axhline(0, color='black', linewidth=0.8)
    plt.title("Actor and Critic Losses Over Time")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("a2c_losses_plot.png")
    plt.close()

# ================================
# Entry Point
# ================================

if __name__ == "__main__":
    train_a2c()
