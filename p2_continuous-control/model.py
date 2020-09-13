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
        """Return the means in the first half of the row and std_devs in the second
        half representing statistics of a Normal distribution. Agent index is the
        row number.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        means = self.means_layer(x)
        means_tanh = F.tanh(means)

        # exp because std_devs must be nonnegative - but clip to keep
        # it from getting too big
        # std_devs = torch.exp(self.std_devs_layer(x)).squeeze(0)
        # std_devs_clipped = torch.clamp(std_devs, 0.0, 1.0)
        std_devs = self.std_devs_layer(x)
        std_devs_sigmoid = F.sigmoid(std_devs)

        action_params = torch.cat([means_tanh, std_devs_sigmoid], dim=-1)
        if torch.isnan(action_params).any() or utils.torch_isinf_any(action_params):
            print('Oops in Policy')
        return action_params
