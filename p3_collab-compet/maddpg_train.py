from maddpg_agent import MADDPG_Agent
import os
from collections import deque

from utilities import transpose_to_tensor
from buffer import ReplayBuffer
import torch
import numpy as np
from tensorboardX import SummaryWriter
from tqdm import tqdm


# Default hyperparameters - some set per suggestions from
#   https://knowledge.udacity.com/questions/315134
NUM_EPISODES = 3000
LEARN_RATE = 1e-3
BATCHSIZE = 1024
EPISODE_LENGTH = 80
EPISODES_PER_UPDATE = 2
DISCOUNT_FACTOR = 0.99
TAU = 0.001
OU_NOISE = 1.0
NOISE_REDUCTION = 0.9999

# Default neural network sizes
HIDDEN_IN_ACTOR = 128
HIDDEN_OUT_ACTOR = 64
HIDDEN_IN_CRITIC = 256
HIDDEN_OUT_CRITIC = 128
LR_ACTOR = 1.0e-4
LR_CRITIC = 3.0e-4

GOAL_WINDOW_LEN = 100
REPLAY_BUFFER_LEN = 100000

MIN_ACTION = -1.0
MAX_ACTION = 1.0


def seeding(random_seed=1):
    """Set the PRNG seeds for numpy and torch"""
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)


def get_train_obs(env_info):
    """Retrieve state from env_info and return (obs, obs_full) in which obs is a list
    of per-agent observations and obs_full has observations from all agents combined
    """
    raw_obs = env_info.vector_observations.reshape(1, -1).squeeze()
    obs_full = [[raw_obs]]
    # obs = [[raw_obs, raw_obs]]  # Both agents get same full observation set
    obs = [[x for x in env_info.vector_observations]]  # Each agent gets only own obs
    return obs, obs_full


def train_maddpg(
    env,
    main_agent: MADDPG_Agent,
    num_episodes=NUM_EPISODES,
    batchsize=BATCHSIZE,
    episode_length=EPISODE_LENGTH,
    episodes_per_update=EPISODES_PER_UPDATE,
    score_history=None,
    ou_noise=OU_NOISE,
    noise_reduction=NOISE_REDUCTION,
    random_seed=237,
    save_interval=1000,
    report_every=100,
    score_goal=0.5,
    progressbar=True,
):
    """Perform MADDPG agent training
    """
    seeding(random_seed)
    if score_history is None:
        score_history = []

    # keep track of last 100 scores
    scores_window = deque(maxlen=GOAL_WINDOW_LEN)

    brain_name = env.brain_names[0]
    env_info = env.reset(train_mode=True)[brain_name]
    num_agents = len(env_info.agents)

    log_path = os.getcwd() + "/log"
    model_dir = os.getcwd() + "/model_dir"
    os.makedirs(model_dir, exist_ok=True)

    buffer = ReplayBuffer(REPLAY_BUFFER_LEN)
    logger = SummaryWriter(log_dir=log_path)

    agent_rewards = []
    for _ in range(num_agents):
        agent_rewards.append(deque(maxlen=GOAL_WINDOW_LEN))

    range_iter = range(num_episodes)
    if progressbar:
        range_iter = tqdm(range_iter)

    for episode_idx in range_iter:
        scores = run_one_episode(
            env,
            main_agent,
            ou_noise,
            num_agents,
            episode_length,
            buffer,
            noise_reduction,
        )

        episode_score = np.max(scores)
        scores_window.append(episode_score)
        score_history.append(episode_score)

        # update agents once after every episode_per_update
        if len(buffer) > batchsize and episode_idx % episodes_per_update == 0:
            samples = buffer.sample(batchsize)
            # samples is a 7-element list: sample transitions from the replay
            # buffer, transposed
            main_agent.update(samples, logger)
            # soft update the target network towards the actual networks
            main_agent.update_targets()

        for i in range(num_agents):
            agent_rewards[i].append(scores[i])

        # Reporting
        logger.add_scalar("scores/episode_score", episode_score, episode_idx)
        if (episode_idx + 1) % report_every == 0 or episode_idx == num_episodes - 1:
            log_episode(
                logger, scores_window, episode_idx, agent_rewards, episode_score,
            )

        # Saving model
        if (episode_idx + 1) % save_interval == 0 or episode_idx == num_episodes - 1:
            save_dict_list = []
            save_model(main_agent, save_dict_list, model_dir, episode_idx)

        # See if we're good enough to stop
        if np.mean(scores_window) >= score_goal:
            print(
                "\nEnvironment solved in {:d} episodes!\t"
                "Average Score: {:.2f}".format(
                    episode_idx - GOAL_WINDOW_LEN, np.mean(scores_window)
                )
            )
            break

    env.close()
    logger.close()

    return score_history


def save_model(main_agent, save_dict_list, model_dir, episode_idx):
    """Save current model parameters"""
    for agent in main_agent.maddpg_agent:
        save_dict = {
            "actor_params": agent.actor.state_dict(),
            "actor_optim_params": agent.actor_optimizer.state_dict(),
            "critic_params": agent.critic.state_dict(),
            "critic_optim_params": agent.critic_optimizer.state_dict(),
        }
        save_dict_list.append(save_dict)

    torch.save(
        save_dict_list, os.path.join(model_dir, "episode-{}.pt".format(episode_idx)),
    )


def log_episode(logger, scores_window, episode_idx, agent_rewards, episode_score):
    logger.add_scalar("scores/mean_window_score", np.mean(scores_window), episode_idx)
    """Log metrics from current episode"""
    for a_i, ag_rewards in enumerate(agent_rewards):
        logger.add_scalar(
            "agent%i/rewards/mean_window_reward" % a_i,
            np.mean(ag_rewards),
            episode_idx,
        )

    print(
        "Episode: {0:d}, score: {1:f}, window mean: {2:f}".format(
            episode_idx + 1, episode_score, np.mean(scores_window)
        )
    )


def run_one_episode(
    env, main_agent, ou_noise, num_agents, episode_length, buffer, noise_reduction,
):
    """Run one episode, adding steps to the replay buffer"""

    brain_name = env.brain_names[0]

    env_info = env.reset(train_mode=True)[brain_name]
    main_agent.reset_episode()
    noise_scale = ou_noise

    scores = np.zeros(num_agents)
    # obs_full: observations as returned from env_info
    # obs: per-agent observations, each just a copy of obs_full
    obs, obs_full = get_train_obs(env_info)

    obs_t = transpose_to_tensor(obs)

    # Run a trajectory and add steps to the replay buffer
    for _ in range(episode_length):
        actions_list = [x.squeeze(0) for x in main_agent.act(obs_t, noise_scale)]
        actions = torch.stack(actions_list).unsqueeze(0).detach().numpy()
        actions = np.clip(actions, MIN_ACTION, MAX_ACTION)
        env_info = env.step(actions.squeeze(0))[brain_name]
        next_obs, next_obs_full = get_train_obs(env_info)
        rewards_2d = [env_info.rewards]
        dones = env_info.local_done
        dones_2d = [dones]
        scores += env_info.rewards

        # add data to buffer
        transition = (
            obs,
            obs_full,
            actions,
            rewards_2d,
            next_obs,
            next_obs_full,
            dones_2d,
        )

        buffer.push(transition)

        noise_scale *= noise_reduction
        obs = next_obs
        if np.any(dones):
            break
    return scores
