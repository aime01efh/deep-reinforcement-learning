# Lightly modified from the Deep Reinforcement Learning Nanodegree
# lunar lander exercise

from collections import deque
import torch
import numpy as np


def train_dqn(env, agent, n_episodes=2000, eps_start=1.0, eps_end=0.01,
              eps_decay=0.995, score_goal=15.0, savefile='checkpoint.pth'):
    """Train a Deep Q-Learning agent.

    Params
    ======
        env: Udacity environment
        agent: DQN Agent instance
        n_episodes (int): maximum number of training episodes
        eps_start (float): starting value of epsilon, for
                           epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for
                           decreasing epsilon
        score_goal: stop training when most recent 100 scores average
                           at least this value
    """
    brain_name = env.brain_names[0]
    # list containing scores from each episode
    scores = []
    # last 100 scores
    scores_window = deque(maxlen=100)
    eps = eps_start

    # Loop until max episodes or sufficient mean score
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        while True:
            action = agent.act(state, eps)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]

            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break

        scores_window.append(score)
        scores.append(score)
        eps = max(eps_end, eps_decay*eps)

        print('\rEpisode {}\tAverage Score: {:.2f}'
              .format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'
                  .format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= score_goal:
            print('\nEnvironment solved in {:d} episodes!\t'
                  'Average Score: {:.2f}'
                  .format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), savefile)
            break

    return scores
