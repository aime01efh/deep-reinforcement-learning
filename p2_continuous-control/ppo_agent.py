import torch
import numpy as np
import random
import model
from utils import device

# import torch.nn.functional as F

MIN_ACTION = -1.0
MAX_ACTION = 1.0


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        self.policy = model.Policy(state_size, action_size).to(device)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    @property
    def parameters(self):
        return self.policy.parameters()

    def _action_params_to_normals(self, action_params):
        # action_params:
        #  [a1_mean, a2_mean, ..., a1_std_dev, a2_std_dev, ...] of a normal distrib
        # slice to get vectors of means and stddevs
        num_parallel_acts = action_params.shape[0] // 2
        means = action_params[:num_parallel_acts, :]
        std_devs = action_params[num_parallel_acts:]

        # get normal distributions with these parameters and sample them
        dist = torch.distributions.Normal(means, std_devs)
        return dist

    def act(self, state):  # TODO train=True?
        """Returns actions and probability for given state per current policy

        Params
        ======
            state (array_like): current state
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.policy.eval()
        with torch.no_grad():
            action_params = self.policy(state)
        # self.policy.train(train)  TODO ok?

        dists = self._action_params_to_normals(action_params)
        actions = torch.clamp(dists.sample(), MIN_ACTION, MAX_ACTION)
        # Sum log_prob of actions within each individual agent
        log_prob = torch.sum(dists.log_prob(actions), dim=1)

        # return (actions.squeeze().cpu().detach().numpy(),
        #         np.exp(log_prob.squeeze().cpu().detach().numpy()))
        return (actions.squeeze().cpu().detach().numpy(),
                torch.exp(log_prob.squeeze()))

    def states_actions_to_prob(self, states, actions):
        """convert states to probability, passing through the policy
        """
        states = torch.tensor(states, dtype=torch.float, device=device)
        action_params = self.policy(states)
        dists = self._action_params_to_normals(action_params)
        # Sum log_prob of actions within each individual agent
        log_prob = torch.sum(dists.log_prob(actions), dim=1)

        return torch.exp(log_prob.squeeze())

    def save(self, filename):
        torch.save(self.policy.state_dict(), filename)
    
    def load(self, filename):
        self.policy.load_state_dict(torch.load(filename))
