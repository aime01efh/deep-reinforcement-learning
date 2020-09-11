import torch
import numpy as np


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def collect_trajectories(env, agent, nrand=5):
    """Collect parallel trajectories for the independent agents
    and return lists of states, rewards, probs, and actions
    """
    # number of parallel instances
    brain_name = env.brain_names[0]
    env_info = env.reset(train_mode=True)[brain_name]
    # num_agents = len(env_info.agents)

    # initialize returning lists and start the game!
    state_list = []
    reward_list = []
    prob_list = []
    action_list = []

    while True:
        state = env_info.vector_observations
        action, action_prob = agent.act(state)

        # advance the game
        env_info = env.step(action)[brain_name]

        # Append each trajectory step to the output lists for all agents
        state_list.append(state)
        reward_list.append(env_info.rewards)
        prob_list.append(action_prob)
        action_list.append(action)

        # stop if any of the trajectories is done
        # we want all the lists to be rectangular
        if np.any(env_info.local_done):
            break

    # return pi_theta, states, actions, rewards, probability
    return prob_list, state_list, action_list, reward_list


def torch_isinf_any(x):
    """Return True if any element of x is inf or -inf.
    Adapted from https://github.com/pytorch/pytorch/issues/9132
    because the version of torch I'm using doesn't have torch.isinf() yet.
    """
    return x.eq(float('inf')).any() or x.eq(float('-inf')).any()
