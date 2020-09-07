import torch
import torch.nn as nn
import torch.nn.functional as F


class Policy(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super(Policy, self).__init__()
        # two fully connected layer
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.means = nn.Linear(hidden_size, action_size)
        self.std_devs = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        means = self.means(x).squeeze(0)
        # exp because std_devs must be nonnegative
        std_devs = torch.exp(self.std_devs(x)).squeeze(0)
        action_params = torch.cat([means, std_devs])
        if torch.isnan(action_params).any():
            print('Oops in Policy')
        return action_params
