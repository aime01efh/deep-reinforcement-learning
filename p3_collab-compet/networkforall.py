# import torch
import torch.nn as nn
import torch.nn.functional as F

# import numpy as np


# def hidden_init(layer):
#     """Return initializing values for the given layer. Not currently used."""
#     fan_in = layer.weight.data.size()[0]
#     lim = 1.0 / np.sqrt(fan_in)
#     return (-lim, lim)


class Network(nn.Module):
    """A neural network with two fully-connected hidden layers to be used for both
    actors and critics. As actor, this NN will return output_dim actions that are
    restricted to the interval [-1.0, 1.0] due to a tanh final activation.
    As critic, will return output_dim Q values.
    """

    def __init__(
        self, input_dim, hidden_in_dim, hidden_out_dim, output_dim, dropout, actor=False
    ):
        super(Network, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_in_dim)
        self.fc2 = nn.Linear(hidden_in_dim, hidden_out_dim)
        self.fc2bis = nn.Linear(hidden_out_dim, hidden_out_dim)  # for critic
        self.fc2ter = nn.Linear(hidden_out_dim, hidden_out_dim)  # for critic
        self.fc3 = nn.Linear(hidden_out_dim, output_dim)
        self.nonlin = F.relu  # leaky_relu?
        self.dropout = nn.Dropout(dropout)
        self.actor = actor
        # self.reset_parameters()

    # def reset_parameters(self):
    #     """Manually reset linear layer parameters. Not currently used.
    #     """
    #     self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
    #     self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
    #     self.fc3.weight.data.uniform_(
    #         -1e-3, 1e-3,
    #     )

    def forward(self, x):
        if self.actor:
            h1 = self.nonlin(self.fc1(x))
            h1 = self.dropout(h1)
            h2 = self.nonlin(self.fc2(h1))
            h2 = self.dropout(h2)
            h3 = self.fc3(h2)

            output_actions = F.tanh(h3)
            return output_actions

        else:
            # critic network simply outputs a number
            h1 = self.nonlin(self.fc1(x))
            h1 = self.dropout(h1)
            h2 = self.nonlin(self.fc2(h1))
            h2 = self.dropout(h2)
            # h2bis = self.nonlin(self.fc2bis(h2))
            # h2bis = self.dropout(h2bis)
            # h2ter = self.nonlin(self.fc2ter(h2bis))
            # h2ter = self.dropout(h2ter)
            h3 = self.fc3(h2)
            return h3
