import torch
import random
import numpy as np
from .DQN import *
import torch.optim as optim
from collections import deque


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=100000)
        self.gamma = 0.95  # 折扣率
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = DQN(state_size, action_size).float()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state)
        q_values = self.model(state)
        return torch.argmax(q_values).item()

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        return zip(*batch)

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            batch_size = len(self.memory)

        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.sample(batch_size)
        # 将数据转换为tensor
        state_batch = torch.tensor(np.array(state_batch), dtype=torch.float)
        action_batch = torch.tensor(action_batch).unsqueeze(1)
        reward_batch = torch.tensor(reward_batch, dtype=torch.float)
        next_state_batch = torch.tensor(np.array(next_state_batch), dtype=torch.float)
        done_batch = torch.tensor(np.float32(done_batch))
        q_values = self.model(state_batch).gather(dim=1, index=action_batch)  # 计算当前状态(s_t,a)对应的Q(s_t, a)
        next_q_values = self.model(next_state_batch).max(1)[0].detach()  # 计算下一时刻的状态(s_t_,a)对应的Q值
        # 计算期望的Q值，对于终止状态，此时done_batch[0]=1, 对应的expected_q_value等于reward
        expected_q_values = reward_batch + self.gamma * next_q_values * (1 - done_batch)
        loss = nn.MSELoss()(q_values, expected_q_values.unsqueeze(1))  # 计算均方根损失
        # 优化更新模型
        self.optimizer.zero_grad()
        loss.backward()
        # clip防止梯度爆炸
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
