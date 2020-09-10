import torch

# import numpy as np
import random
import model
from utils import device

# import torch.nn.functional as F

MIN_ACTION = -1.0
MAX_ACTION = 1.0

# To avoid numerical problems when action is very unlikely in the new policy
MIN_ACTION_PROB = torch.FloatTensor([1e-6])


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, hidden_sizes, seed):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            hidden_sizes (list of int): number of units in the two hidden layers
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        self.policy = model.Policy(state_size, action_size, hidden_sizes).to(device)

    @property
    def parameters(self):
        """The trainable parameters of the model"""
        return self.policy.parameters()

    def _action_params_to_normals(self, action_params):
        """From the NN model output which has means in the first half
        and std_devs in the second half, return the corresponding
        Normal distributions
        """
        num_agents = action_params.shape[0] // 2
        means = action_params[:num_agents, :]
        std_devs = action_params[num_agents:]

        # get normal distributions with these parameters
        dist = torch.distributions.Normal(means, std_devs)
        return dist

    def _actions_to_log_prob(self, dists, actions):
        """Given actions that have been sampled from the Normal distributions
        created by _action_params_to_normals, return the log of the
        probability densities of those actions, summed across the set of actions
        of each parallel agent
        """
        # Remember that log_prob returns probability densities which can be > 1
        log_probs_2d = dists.log_prob(actions)
        log_probs = torch.sum(log_probs_2d, dim=-1)
        return log_probs

    def act(self, state):
        """Return sampled actions and their probability for the given state
        per current policy
        """
        # Run the state through the policy to get the means and and std_devs
        # of Normal distributions to be sampled for each action
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.policy.eval()
        with torch.no_grad():
            action_params = self.policy(state)

        # Create Normal distributions, sample actions, get log probabilities
        dists = self._action_params_to_normals(action_params)
        actions = torch.clamp(dists.sample(), MIN_ACTION, MAX_ACTION)
        # Get the sum log_prob of actions within each individual agent
        log_prob = self._actions_to_log_prob(dists, actions)

        actions_np = actions.squeeze().cpu().detach().numpy()
        action_probs = torch.max(torch.exp(log_prob.squeeze()), MIN_ACTION_PROB)
        return (actions_np, action_probs)

    def states_actions_to_prob(self, states, actions, train=True):
        """Return probability density of the given sampled action in the
        current policy, as well as the Normal distributions of the policy
        """
        self.policy.train(train)
        action_params = self.policy(states)
        dists = self._action_params_to_normals(action_params)
        # Sum log_prob of actions within each individual agent
        log_prob = self._actions_to_log_prob(dists, actions)

        prob_clipped = torch.max(torch.exp(log_prob.squeeze()), MIN_ACTION_PROB)
        return prob_clipped, dists

    def save(self, filename):
        torch.save(self.policy.state_dict(), filename)

    def load(self, filename):
        self.policy.load_state_dict(torch.load(filename))
