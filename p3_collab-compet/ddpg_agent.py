# individual network settings for each actor + critic pair
# see networkforall for details

from typing import NamedTuple
from networkforall import Network
from utilities import hard_update  # , gumbel_softmax, onehot_from_logits
from torch.optim import Adam

# import torch
# import numpy as np

# add OU noise for exploration
from OUNoise import OUNoise

# WEIGHT_DECAY = 1e-5
WEIGHT_DECAY = 0


class NNParams(NamedTuple):
    in_actor: int
    hidden_in_actor: int
    hidden_out_actor: int
    out_actor: int
    in_critic: int
    hidden_in_critic: int
    hidden_out_critic: int
    lr_actor: float
    lr_critic: float
    dropout: float


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"


class DDPGAgent:
    """Deep Deterministic Policy Gradient agent

    Manages four separate neural networks: working and target actor networks,
    and working and target critic networks, as well as OU noise generation.
    """
    def __init__(self, p: NNParams):
        self.actor = Network(
            p.in_actor,
            p.hidden_in_actor,
            p.hidden_out_actor,
            p.out_actor,
            p.dropout,
            actor=True,
        ).to(device)
        self.critic = Network(
            p.in_critic, p.hidden_in_critic, p.hidden_out_critic, 1, p.dropout
        ).to(device)
        self.target_actor = Network(
            p.in_actor,
            p.hidden_in_actor,
            p.hidden_out_actor,
            p.out_actor,
            p.dropout,
            actor=True,
        ).to(device)
        self.target_critic = Network(
            p.in_critic, p.hidden_in_critic, p.hidden_out_critic, 1, p.dropout
        ).to(device)

        self.noise = OUNoise(p.out_actor, scale=1.0)

        # initialize targets same as original networks
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=p.lr_actor)
        self.critic_optimizer = Adam(
            self.critic.parameters(), lr=p.lr_critic, weight_decay=WEIGHT_DECAY
        )

    def act(self, obs, noise=0.0):
        """Select an action from the working policy based on the given state, with OU noise
        added that is scaled by the value of the "noise" parameter
        """
        obs = obs.to(device)
        action = self.actor(obs) + noise * self.noise.noise()
        return action

    def target_act(self, obs, noise=0.0):
        """Select an action from the target policy based on the given state, with OU noise
        added that is scaled by the value of the "noise" parameter
        """
        obs = obs.to(device)
        action = self.target_actor(obs) + noise * self.noise.noise()
        return action
