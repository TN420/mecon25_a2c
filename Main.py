import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns

# Constants
TOTAL_PRBS = 100
URLLC_QUOTA = 30
EMBB_QUOTA = 70
TRAFFIC_TYPES = ['URLLC', 'eMBB']
GAMMA = 0.99
LR = 0.001

# Environment class
class RANEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.urllc_usage = 0
        self.embb_usage = 0
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

# Actor-Critic networks
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.fc(x)

class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.fc(x)

# Helper function
def smooth_curve(data, window_size=10):
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Training
def train_a2c(episodes=500, max_steps=100):
    env = RANEnv()
    state_dim = len(env.reset())
    action_dim = 2

    actor = Actor(state_dim, action_dim)
    critic = Critic(state_dim)
    actor_optim = optim.Adam(actor.parameters(), lr=LR)
    critic_optim = optim.Adam(critic.parameters(), lr=LR)

    actor_losses, critic_losses = [], []
    episode_returns = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0

        for t in range(max_steps):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            probs = actor(state_tensor)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample().item()

            traffic_type = random.choice(TRAFFIC_TYPES)
            next_state, reward, done, _, _, _, _ = env.step(action, traffic_type)

            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
            reward_tensor = torch.tensor([reward], dtype=torch.float32)

            value = critic(state_tensor)
            next_value = critic(next_state_tensor)
            advantage = reward_tensor + GAMMA * next_value - value

            actor_loss = -dist.log_prob(torch.tensor(action)) * advantage.detach()
            critic_loss = advantage.pow(2)

            actor_optim.zero_grad()
            actor_loss.backward()
            actor_optim.step()

            critic_optim.zero_grad()
            critic_loss.backward()
            critic_optim.step()

            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
            state = next_state
            total_reward += reward

        episode_returns.append(total_reward)
        print(f"Episode {episode + 1}, Return: {total_reward:.2f}")

    # Plot losses
    plt.figure(figsize=(12, 5))
    plt.plot(smooth_curve(actor_losses), label='Actor Loss')
    plt.plot(smooth_curve(critic_losses), label='Critic Loss')
    plt.title('Smoothed Actor and Critic Loss')
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig("a2c_losses.png")
    plt.show()

    # Plot returns
    plt.figure(figsize=(12, 5))
    plt.plot(smooth_curve(episode_returns, window_size=20), label='Smoothed Episode Return')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.title('Smoothed Episode Return Over Time')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("a2c_episode_returns.png")
    plt.show()

if __name__ == "__main__":
    train_a2c(episodes=1000, max_steps=100)
