import torch
import numpy as np


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def collect_trajectories(env, agent, tmax=200, nrand=5):
    """Collect parallel trajectories for the independent agents
    and return lists of states, rewards, probs, and actions
    """
    # number of parallel instances
    brain_name = env.brain_names[0]
    env_info = env.reset(train_mode=True)[brain_name]
    num_agents = len(env_info.agents)

    # initialize returning lists and start the game!
    state_list = []
    reward_list = []
    prob_list = []
    action_list = []

    for t in range(tmax):
        state = env_info.vector_observations
        action, action_prob = agent.act(state)

        # advance the game
        env_info = env.step(action)[brain_name]

        # Append each trajectory step to the output lists for all agents
        for agent_idx in range(num_agents):
            # store the result
            state_list.append(state[agent_idx])
            reward_list.append(env_info.rewards[agent_idx])
            prob_list.append(action_prob[agent_idx])
            action_list.append(action[agent_idx])

        # stop if any of the trajectories is done
        # we want all the lists to be rectangular
        if np.any(env_info.local_done):
            break

    # return pi_theta, states, actions, rewards, probability
    return prob_list, state_list, action_list, reward_list
