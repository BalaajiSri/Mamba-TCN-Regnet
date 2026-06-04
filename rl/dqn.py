from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: float


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def push(self, transition: Transition):
        self.buffer.append(transition)

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states = np.stack([item.state for item in batch]).astype(np.float32)
        actions = np.asarray([item.action for item in batch], dtype=np.int64)
        rewards = np.asarray([item.reward for item in batch], dtype=np.float32)
        next_states = np.stack([item.next_state for item in batch]).astype(np.float32)
        dones = np.asarray([item.done for item in batch], dtype=np.float32)
        return states, actions, rewards, next_states, dones


class DuelingQNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int = 3, dueling: bool = True):
        super().__init__()
        self.dueling = dueling
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        if dueling:
            self.value_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1),
            )
            self.advantage_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, action_dim),
            )
        else:
            self.q_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, action_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(x)
        if not self.dueling:
            return self.q_head(hidden)
        value = self.value_head(hidden)
        advantage = self.advantage_head(hidden)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)


class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        lr: float,
        gamma: float,
        tau: float,
        double_dqn: bool = True,
        dueling: bool = True,
        device: str = "cpu",
    ):
        self.gamma = gamma
        self.tau = tau
        self.double_dqn = double_dqn
        self.device = torch.device(device)
        self.policy_net = DuelingQNetwork(state_dim=state_dim, hidden_dim=hidden_dim, dueling=dueling).to(self.device)
        self.target_net = DuelingQNetwork(state_dim=state_dim, hidden_dim=hidden_dim, dueling=dueling).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)

    def act(self, state: np.ndarray, epsilon: float) -> int:
        if random.random() < epsilon:
            return random.randrange(3)
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return int(q_values.argmax(dim=-1).item())

    def update(self, batch, loss_fn=nn.SmoothL1Loss()):
        states, actions, rewards, next_states, dones = batch
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(-1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(-1)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(-1)

        q_values = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            if self.double_dqn:
                next_actions = self.policy_net(next_states).argmax(dim=-1, keepdim=True)
                next_q = self.target_net(next_states).gather(1, next_actions)
            else:
                next_q = self.target_net(next_states).max(dim=-1, keepdim=True).values
            target = rewards + (1.0 - dones) * self.gamma * next_q

        loss = loss_fn(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update()
        return float(loss.item())

    def soft_update(self):
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)

