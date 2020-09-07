from collections import deque
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import utils


NUM_EPISODES = 500
DISCOUNT_RATE = .99
EPSILON = 0.1
BETA = .01
TMAX = 320
# SDG_epoch is number of times to reuse trajectories; 1=REINFORCE
SGD_EPOCH = 4
LEARN_RATE = 1e-4


def train_ppo(env, agent, num_episodes=NUM_EPISODES,
              epsilon=EPSILON, discount_rate=DISCOUNT_RATE,
              beta=BETA, tmax=TMAX, sgd_epoch=SGD_EPOCH,
              learn_rate=LEARN_RATE, report_every=20,
              score_goal=30.0, progressbar=True):
    """Train a PPO agent. Return a list of rewards
    averaged across the parallel evironments.
    """
    # last 100 scores
    scores_window = deque(maxlen=100)

    optimizer = optim.Adam(agent.parameters, lr=learn_rate)
    mean_rewards_per_episode = []
    brain_name = env.brain_names[0]
    env_info = env.reset(train_mode=True)[brain_name]
    num_agents = len(env_info.agents)

    range_iter = range(num_episodes)
    if progressbar:
        range_iter = tqdm(range_iter)

    for episode_idx in range_iter:
        # collect trajectories
        prob_list, states, actions, rewards = \
            utils.collect_trajectories(env, agent, tmax=tmax)

        total_episode_rewards = np.sum(rewards)
        avg_episode_reward_per_agent = total_episode_rewards / num_agents

        # gradient ascent step
        for _ in range(sgd_epoch):
            L = -clipped_surrogate(agent, prob_list, states, actions, rewards,
                                   epsilon=epsilon, beta=beta)

            optimizer.zero_grad()
            L.backward()
            optimizer.step()
            del L

        # the clipping parameter reduces as time goes on
        epsilon *= 0.999

        # this reduces exploration in later runs
        beta *= 0.995

        # save the average reward of the parallel environments for this episode
        mean_rewards_per_episode.append(avg_episode_reward_per_agent)
        scores_window.append(avg_episode_reward_per_agent)

        if abs(np.mean(avg_episode_reward_per_agent)) < 1e-6:
            print('Score is 0')

        # display some progress every 20 iterations
        if (episode_idx + 1) % report_every == 0:
            print("Episode: {0:d}, score: {1:f}, window mean: {2:f}"
                  .format(episode_idx+1, np.mean(avg_episode_reward_per_agent),
                          np.mean(scores_window)))

        if np.mean(scores_window) >= score_goal:
            print('\nEnvironment solved in {:d} episodes!\t'
                  'Average Score: {:.2f}'
                  .format(episode_idx-100, np.mean(scores_window)))
            break

    return mean_rewards_per_episode


def clipped_surrogate(agent, old_probs, states, actions, rewards,
                      discount=0.995, epsilon=0.1, beta=0.01):
    discount = discount**np.arange(len(rewards))
    rewards = np.asarray(rewards)*discount[:, np.newaxis]

    # convert rewards to future rewards
    rewards_future = rewards[::-1].cumsum(axis=0)[::-1]

    mean = np.mean(rewards_future, axis=1)
    std = np.std(rewards_future, axis=1) + 1.0e-10

    # batch normalization of rewards, accelerates training
    rewards_normalized = (rewards_future - mean[:, np.newaxis])/std[:, np.newaxis]

    actions = torch.tensor(actions, dtype=torch.float, device=utils.device)
    old_probs = torch.tensor(old_probs, dtype=torch.float, device=utils.device)

    rewards = torch.tensor(rewards_normalized, dtype=torch.float,
                           device=utils.device)
    # "rewards" is now future rewards, normalized, as torch tensor

    # convert states to policy (or probability)
    new_probs, action_prob_dists = agent.states_actions_to_prob(states, actions)

    ratio = new_probs / old_probs
    clipped_ratio = torch.clamp(ratio, 1-epsilon, 1+epsilon)
    clipped_surrogate_val = torch.min(ratio*rewards, clipped_ratio*rewards)

    # include a regularization term
    # this steers new_policy towards 0.5
    # prevents policy to become exactly 0 or 1 helps exploration
    # add in 1.e-10 to avoid log(0) which gives nan
    # this entropy is for binary action
    # entropy = (-(new_probs*torch.log(old_probs+1.e-10)
    #            + (1.0-new_probs)*torch.log(1.0-old_probs+1.e-10)))
    entropy = action_prob_dists.entropy().mean()

    # take 1/T * sigma(...)
    if torch.isnan(clipped_surrogate_val).any():
        print("Oops in L-1")
    if torch.isnan(beta*entropy).any():
        print("Oops in L-2")
    return torch.mean(clipped_surrogate_val + beta*entropy)
