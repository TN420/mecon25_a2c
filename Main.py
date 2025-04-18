import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque

# Environment Constants
TOTAL_PRBS = 100
URLLC_QUOTA = 30
EMBB_QUOTA = 70

# Traffic Types
TRAFFIC_TYPES = ['URLLC', 'eMBB']

# DQN Hyperparameters
GAMMA = 0.95
LR = 0.001
BATCH_SIZE = 32
MEMORY_SIZE = 10000
TARGET_UPDATE = 10

# Environment
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
        # State now includes SLA preservation ratios
        return np.array([
            self.urllc_usage / TOTAL_PRBS,
            self.embb_usage / TOTAL_PRBS,
            1.0,  # Default SLA preservation for URLLC
            1.0   # Default SLA preservation for eMBB
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

# DQN Network
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state = map(np.array, zip(*batch))
        return state, action, reward, next_state

    def __len__(self):
        return len(self.buffer)

# Moving average smoothing
def smooth(data, window=10):
    return np.convolve(data, np.ones(window)/window, mode='valid')

# Q-value Heatmap Plotting
def plot_q_value_heatmaps(policy_net):
    usage_levels = np.linspace(0, 1, 50)
    q_vals_action0 = np.zeros((50, 50))
    q_vals_action1 = np.zeros((50, 50))

    for i, urllc_norm in enumerate(usage_levels):
        for j, embb_norm in enumerate(usage_levels):
            # Pass 4-dimensional state to match the expected input size
            state = torch.tensor([urllc_norm, embb_norm, 1.0, 1.0], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_values = policy_net(state)
                q_vals_action0[i, j] = q_values[0, 0].item()
                q_vals_action1[i, j] = q_values[0, 1].item()

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(q_vals_action0, xticklabels=False, yticklabels=False, ax=axs[0], cmap='coolwarm')
    axs[0].set_title("Q-values for Action 0 (Reject)")

    sns.heatmap(q_vals_action1, xticklabels=False, yticklabels=False, ax=axs[1], cmap='coolwarm')
    axs[1].set_title("Q-values for Action 1 (Admit)")

    for ax in axs:
        ax.set_xlabel("eMBB Usage (normalized)")
        ax.set_ylabel("URLLC Usage (normalized)")

    plt.tight_layout()
    plt.savefig("q_value_heatmaps.png")
    plt.show()

# Training Loop
def train_dqn(episodes=200):
    env = RANEnv()
    state_size = len(env.reset())
    action_size = 2

    policy_net = DQN(state_size, action_size)
    target_net = DQN(state_size, action_size)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayBuffer(MEMORY_SIZE)

    reward_history = []
    urllc_block_history = []
    embb_block_history = []
    urllc_sla_pres = []
    embb_sla_pres = []
    urllc_usage_hist = []
    embb_usage_hist = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        urllc_blocks = 0
        embb_blocks = 0
        urllc_sla_preserved = 0
        embb_sla_preserved = 0
        urllc_total_requests = 0
        embb_total_requests = 0

        for t in range(100):
            traffic_type = random.choice(TRAFFIC_TYPES)
            if traffic_type == 'URLLC':
                urllc_total_requests += 1
            else:
                embb_total_requests += 1

            epsilon = max(0.05, 0.9 - episode / 200)
            if random.random() < epsilon:
                action = random.randint(0, 1)
            else:
                with torch.no_grad():
                    q_vals = policy_net(torch.tensor(state).float().unsqueeze(0))
                    action = q_vals.argmax().item()

            next_state, reward, done, admitted, blocked, sla_violated, t_type = env.step(action, traffic_type)

            if blocked:
                if t_type == 'URLLC': urllc_blocks += 1
                if t_type == 'eMBB': embb_blocks += 1
            if not sla_violated:
                if t_type == 'URLLC': urllc_sla_preserved += 1
                if t_type == 'eMBB': embb_sla_preserved += 1

            memory.push(state, action, reward, next_state)
            state = next_state
            total_reward += reward

            if len(memory) >= BATCH_SIZE:
                s, a, r, ns = memory.sample(BATCH_SIZE)
                s = torch.tensor(s).float()
                a = torch.tensor(a).long()
                r = torch.tensor(r).float()
                ns = torch.tensor(ns).float()

                q_values = policy_net(s).gather(1, a.unsqueeze(1)).squeeze()
                next_q_values = target_net(ns).max(1)[0].detach()
                expected_q = r + GAMMA * next_q_values

                loss = nn.MSELoss()(q_values, expected_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        reward_history.append(total_reward)
        urllc_block_history.append(urllc_blocks)
        embb_block_history.append(embb_blocks)
        urllc_sla_pres.append(urllc_sla_preserved / urllc_total_requests if urllc_total_requests > 0 else 0)
        embb_sla_pres.append(embb_sla_preserved / embb_total_requests if embb_total_requests > 0 else 0)
        urllc_usage_hist.append(env.urllc_usage)
        embb_usage_hist.append(env.embb_usage)

        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        print(f"Episode {episode+1}: Total Reward = {total_reward:.2f}, URLLC SLA Ratio = {urllc_sla_pres[-1]:.2f}, eMBB SLA Ratio = {embb_sla_pres[-1]:.2f}")

    # Plotting
    fig, axs = plt.subplots(3, 2, figsize=(14, 10))

    axs[0, 0].plot(smooth(reward_history))
    axs[0, 0].set_title("Smoothed Total Reward")

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
    plt.savefig("performance_plots.png")
    plt.show()

    # Plot Q-value heatmaps
    plot_q_value_heatmaps(policy_net)

if __name__ == "__main__":
    train_dqn(episodes=1000)
