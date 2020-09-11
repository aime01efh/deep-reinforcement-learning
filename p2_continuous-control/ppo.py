from collections import deque
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import utils


NUM_EPISODES = 10000
DISCOUNT_RATE = .99
EPSILON = 0.1
BETA = .01
# SDG_epoch is number of times to reuse trajectories; 1=REINFORCE
SGD_EPOCH = 6
LEARN_RATE = 1e-4


def train_ppo(env, agent, num_episodes=NUM_EPISODES,
              epsilon=EPSILON, discount_rate=DISCOUNT_RATE,
              beta=BETA, num_sgd_epoch=SGD_EPOCH,
              learn_rate=LEARN_RATE, report_every=20,
              score_goal=30.0, progressbar=True):
    """Train a PPO agent. Return a list of rewards
    averaged across the parallel evironments.
    """
    # last 100 scores
    scores_window = deque(maxlen=100)

    optimizer = optim.Adam(agent.parameters, lr=learn_rate)
    mean_rewards_per_episode = []
    # brain_name = env.brain_names[0]
    # env_info = env.reset(train_mode=True)[brain_name]
    # num_agents = len(env_info.agents)

    range_iter = range(num_episodes)
    if progressbar:
        range_iter = tqdm(range_iter)

    for episode_idx in range_iter:
        # Gather trajectories from all parallel agents
        prob_list, states, actions, rewards = \
            utils.collect_trajectories(env, agent)

        # Sum all rewards in the episode for each individual agent,
        # then average this per-agent total reward across all agents
        avg_episode_reward = np.array(rewards).sum(axis=0).mean()

        # Optimize policy weights via gradient ascent
        for _ in range(num_sgd_epoch):
            L = -clipped_surrogate(agent, prob_list, states, actions, rewards,
                                   epsilon=epsilon, beta=beta)

            # print('L =', L.detach().numpy())
            optimizer.zero_grad()
            L.backward()
            optimizer.step()
            del L

        # the clipping parameter reduces as time goes on
        epsilon *= 0.999

        # this reduces exploration in later runs
        beta *= 0.995

        # save the average reward of the parallel environments for this episode
        mean_rewards_per_episode.append(avg_episode_reward)
        scores_window.append(avg_episode_reward)

        if abs(np.mean(avg_episode_reward)) < 1e-6:
            print('Score is 0')

        # display some progress every 20 iterations
        if (episode_idx + 1) % report_every == 0:
            print("Episode: {0:d}, score: {1:f}, window mean: {2:f}"
                  .format(episode_idx+1, avg_episode_reward,
                          np.mean(scores_window)))

        if np.mean(scores_window) >= score_goal:
            print('\nEnvironment solved in {:d} episodes!\t'
                  'Average Score: {:.2f}'
                  .format(episode_idx-100, np.mean(scores_window)))
            break

    return mean_rewards_per_episode


def clipped_surrogate(agent, old_probs, states, actions, rewards,
                      discount=0.995, epsilon=0.1, beta=0.01):
    """Return the PPO surrogate loss function using a Monte Carlo policy gradient
    """
    discount = discount**np.arange(len(rewards))
    rewards_np = np.asarray(rewards) * discount[:, np.newaxis]

    # convert rewards to future rewards
    rewards_future = rewards_np[::-1].cumsum(axis=0)[::-1]

    mean = np.mean(rewards_future, axis=1)
    std_dev = np.std(rewards_future, axis=1) + 1.0e-10

    # batch normalization of rewards, accelerates training
    rewards_normalized = ((rewards_future - mean[:, np.newaxis])
                          / std_dev[:, np.newaxis])

    states_t = torch.tensor(states, dtype=torch.float, device=utils.device)
    actions_t = torch.tensor(actions, dtype=torch.float, device=utils.device)
    old_probs_t = torch.stack(old_probs, dim=0)

    rewards_t = torch.tensor(rewards_normalized, dtype=torch.float,
                             device=utils.device)
    # "rewards" is now future rewards, normalized, as torch tensor

    # Get probabilities of the sampled actions in the current policy
    new_probs, action_prob_dists = agent.states_actions_to_prob(states_t, actions_t)

    ratio = new_probs / old_probs_t
    if torch.isnan(ratio).any():
        print('Oops in ratio')
    clipped_ratio = torch.clamp(ratio, 1-epsilon, 1+epsilon)
    clipped_surrogate_val = torch.min(ratio * rewards_t, clipped_ratio * rewards_t)

    # include a regularization term
    # this entropy is for binary action:
    #   entropy = (-(new_probs*torch.log(old_probs+1.e-10)
    #              + (1.0-new_probs)*torch.log(1.0-old_probs+1.e-10)))
    entropy = action_prob_dists.entropy().mean()

    # take 1/T * sigma(...)
    if (
        torch.isnan(clipped_surrogate_val).any()
        or torch.isnan(beta * entropy).any()
        or utils.torch_isinf_any(clipped_surrogate_val)
        or utils.torch_isinf_any(beta * entropy)
    ):
        print("Oops in L")
    return torch.mean(clipped_surrogate_val + beta*entropy)
