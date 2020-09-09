import torch
import torch.nn as nn
import torch.nn.functional as F
import utils


class Policy(nn.Module):
    def __init__(self, state_size, action_size, hidden_sizes):
        super(Policy, self).__init__()
        # two fully connected layers
        self.fc1 = nn.Linear(state_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        # separate heads for mean and std_dev
        self.means_layer = nn.Linear(hidden_sizes[1], action_size)
        self.std_devs_layer = nn.Linear(hidden_sizes[1], action_size)

    def forward(self, x):
        """Return the means in the first half and std_devs in the second
        representing statistics of a Normal distribution
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        means = self.means_layer(x).squeeze(0)
        # exp because std_devs must be nonnegative - but clip to keep
        # it from getting too big
        std_devs_raw = torch.clamp(self.std_devs_layer(x), 0, 1)
        std_devs = torch.exp(std_devs_raw).squeeze(0)
        action_params = torch.cat([means, std_devs])
        if torch.isnan(action_params).any() or utils.torch_isinf_any(action_params):
            print('Oops in Policy')
        return action_params
