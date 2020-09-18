import os

from buffer import ReplayBuffer
import torch
import numpy as np
from tensorboardX import SummaryWriter
from tqdm import tqdm


# Default hyperparameters
NUM_EPISODES = 1000
LEARN_RATE = 1e-3
BATCHSIZE = 1000
EPISODE_LENGTH = 80
EPISODES_PER_UPDATE = 8
DISCOUNT_FACTOR = 0.95
TAU = 0.02

# Default neural network sizes
HIDDEN_IN_ACTOR = 0
HIDDEN_OUT_ACTOR = 0
OUT_ACTOR = 0
IN_CRITIC = 0
HIDDEN_IN_CRITIC = 0
HIDDEN_OUT_CRITIC = 0
LR_ACTOR = 1.0e-3
LR_CRITIC = 1.0e-3

GOAL_WINDOW_LEN = 100
REPLAY_BUFFER_LEN = 5000


def seeding(random_seed=1):
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)


def train_maddpg(
    env,
    agent,
    num_episodes=NUM_EPISODES,
    batchsize=BATCHSIZE,
    episode_length=EPISODE_LENGTH,
    episodes_per_update=EPISODES_PER_UPDATE,
    ou_noise=2.0,
    noise_reduction=0.9999,
    random_seed=237,
    save_interval=1000,
    progressbar=True,
):
    seeding(random_seed)

    brain_name = env.brain_names[0]
    env_info = env.reset(train_mode=True)[brain_name]
    num_agents = len(env_info.agents)

    log_path = os.getcwd() + "/log"
    model_dir = os.getcwd() + "/model_dir"
    os.makedirs(model_dir, exist_ok=True)

    buffer = ReplayBuffer(int(REPLAY_BUFFER_LEN * episode_length))

    # initialize policy and critic
    logger = SummaryWriter(log_dir=log_path)
    agent_rewards = []
    for _ in num_agents:
        agent_rewards.append([])

    range_iter = range(num_episodes)
    if progressbar:
        range_iter = tqdm(range_iter)
    for episode_idx in range_iter:
        env_info = env.reset(train_mode=True)[brain_name]
        scores = np.zeros(num_agents)
        states = env_info.vector_observations

        for episode_t in range(episode_length):
            # # action input needs to be transposed
            # actions = maddpg.act(transpose_to_tensor(obs), noise=noise)
            # noise *= noise_reduction
            # actions_array = torch.stack(actions).detach().numpy()
            # next_obs, next_obs_full, rewards, dones, info = env.step(actions_for_env)

            actions = agent.act(states, ou_noise)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            scores += env_info.rewards

            # add data to buffer
            transition = (
                states,
                actions,
                rewards,
                next_states,
                dones,
            )

            buffer.push(transition)

            states = next_states
            if np.any(dones):
                break

        # update once after every episode_per_update
        if len(buffer) > batchsize and episode_idx % episodes_per_update == 0:
            for a_i in range(num_agents):
                samples = buffer.sample(batchsize)
                agent.update(samples, a_i, logger)
            agent.update_targets()  # soft update the target network towards the actual networks

        # for i in range(num_agents):
        #     agent_rewards[i].append(scores[i])

        # if episode % 100 == 0 or episode == number_of_episodes - 1:
        #     avg_rewards = [
        #         np.mean(agent0_reward),
        #         np.mean(agent1_reward),
        #     ]
        #     agent0_reward = []
        #     agent1_reward = []
        #     for a_i, avg_rew in enumerate(avg_rewards):
        #         logger.add_scalar(
        #             "agent%i/mean_episode_rewards" % a_i, avg_rew, episode
        #         )

        # saving model
        save_dict_list = []
        if episode_idx % save_interval == 0:
            for agent in agent.maddpg_agent:
                save_dict = {
                    "actor_params": agent.actor.state_dict(),
                    "actor_optim_params": agent.actor_optimizer.state_dict(),
                    "critic_params": agent.critic.state_dict(),
                    "critic_optim_params": agent.critic_optimizer.state_dict(),
                }
                save_dict_list.append(save_dict)

                torch.save(
                    save_dict_list,
                    os.path.join(model_dir, "episode-{}.pt".format(episode_idx)),
                )

    env.close()
    logger.close()
