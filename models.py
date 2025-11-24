import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, obs_size, act_size):
        super(Actor, self).__init__()
        self.input_layer = nn.Linear(obs_size, 128)
        self.linear_1 = nn.Linear(128, 256)
        self.output_layer = nn.Linear(256, act_size)
        self.activation = nn.ReLU()
        self.final_activation = nn.Tanh()

    def forward(self, x):
        x = self.activation(self.input_layer(x))
        x = self.activation(self.linear_1(x))
        return self.final_activation(self.output_layer(x))

class Critic(nn.Module):
    def __init__(self, obs_size, act_size):
        super(Critic, self).__init__()
        self.input_layer = nn.Linear(obs_size + act_size, 128)
        self.linear_1 = nn.Linear(128, 256)
        self.output_layer = nn.Linear(256, 1)
        self.activation = nn.ReLU()

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=1)
        x = self.activation(self.input_layer(x))
        x = self.activation(self.linear_1(x))
        return self.output_layer(x)
