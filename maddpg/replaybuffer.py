from collections import deque
import random
import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=int(capacity))

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        sample = map(np.array, zip(*batch))
        return tuple(map(torch.FloatTensor, sample))

    def __len__(self):
        return len(self.buffer)
