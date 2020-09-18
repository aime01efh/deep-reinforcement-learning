# individual network settings for each actor + critic pair
# see networkforall for details

from collections import namedtuple
from networkforall import Network
from utilities import hard_update  # , gumbel_softmax, onehot_from_logits
from torch.optim import Adam

# import torch
# import numpy as np


# add OU noise for exploration
from OUNoise import OUNoise

NNParams = namedtuple(
    "NNParams",
    [
        "in_actor",
        "hidden_in_actor",
        "hidden_out_actor",
        "out_actor",
        "in_critic",
        "hidden_in_critic",
        "hidden_out_critic",
        "lr_actor",
        "lr_critic",
    ],
)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"


class DDPGAgent:
    def __init__(self, p: NNParams):
        self.actor = Network(
            p.in_actor, p.hidden_in_actor, p.hidden_out_actor, p.out_actor, actor=True
        ).to(device)
        self.critic = Network(
            p.in_critic, p.hidden_in_critic, p.hidden_out_critic, 1
        ).to(device)
        self.target_actor = Network(
            p.in_actor, p.hidden_in_actor, p.hidden_out_actor, p.out_actor, actor=True
        ).to(device)
        self.target_critic = Network(
            p.in_critic, p.hidden_in_critic, p.hidden_out_critic, 1
        ).to(device)

        self.noise = OUNoise(p.out_actor, scale=1.0)

        # initialize targets same as original networks
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=p.lr_actor)
        self.critic_optimizer = Adam(
            self.critic.parameters(), lr=p.lr_critic, weight_decay=1.0e-5
        )

    def act(self, obs, noise=0.0):
        obs = obs.to(device)
        action = self.actor(obs) + noise * self.noise.noise()
        return action

    def target_act(self, obs, noise=0.0):
        obs = obs.to(device)
        action = self.target_actor(obs) + noise * self.noise.noise()
        return action
