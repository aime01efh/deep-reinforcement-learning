# main code that contains the neural network setup
# policy + critic updates
# see ddpg.py for other details in the network
import numpy as np
from tensorboardX.writer import SummaryWriter
from buffer import ReplayBuffer
from typing import List, Optional
from ddpg_agent import DDPGAgent, device
import torch
import torch.nn.functional as F
from utilities import soft_update, transpose_to_tensor  # , transpose_list


class MADDPG_Agent:
    """MADDPG Learning Algorithm

    Manages multiple DDPG agents, each with their own actor and critic
    networks.
    """

    def __init__(self, num_agents, ddpg_params, discount_factor, tau):
        # critic input = obs_full + all agent actions
        self.maddpg_agents: List[DDPGAgent] = []
        for _ in range(num_agents):
            self.maddpg_agents.append(DDPGAgent(ddpg_params))

        self.discount_factor = discount_factor
        self.tau = tau
        self.iter = 0

    def get_actors(self):
        """get actors of all the agents in the MADDPG object"""
        actors = [ddpg_agent.actor for ddpg_agent in self.maddpg_agents]
        return actors

    def get_target_actors(self):
        """get target_actors of all the agents in the MADDPG object"""
        target_actors = [ddpg_agent.target_actor for ddpg_agent in self.maddpg_agents]
        return target_actors

    def set_train_mode(self, train_mode=True):
        """Enabled or disable training mode for all agents
        """
        for agent in self.maddpg_agents:
            agent.actor.train(train_mode)
            agent.critic.train(train_mode)

    def act(self, obs_all_agents, noise=0.0):
        """Get actions from all agents in the MADDPG object.
        For evaluation only, not training
        """
        with torch.no_grad():
            self.set_train_mode(False)
            actions = [
                agent.act(obs, noise)
                for agent, obs in zip(self.maddpg_agents, obs_all_agents)
            ]
            self.set_train_mode(True)
        return actions

    def target_act(self, obs_all_agents, noise=0.0):
        """get target network actions from all the agents in the MADDPG object
        For evaluation only, not training
        """
        with torch.no_grad():
            self.set_train_mode(False)
            target_actions = [
                ddpg_agent.target_act(obs, noise)
                for ddpg_agent, obs in zip(self.maddpg_agents, obs_all_agents)
            ]
            self.set_train_mode(True)
        return target_actions

    def reset_episode(self):
        """Reset DDPG agents to begin an episode"""
        for agent in self.maddpg_agents:
            agent.reset()

    def update(
        self, buffer: ReplayBuffer, batchsize: int, logger: Optional[SummaryWriter]
    ):
        """update the critics and actors of all the agents"""
        for agent_number in range(len(self.maddpg_agents)):
            samples = buffer.sample(batchsize)
            self.update_one_agent(agent_number, samples, logger)

    def update_one_agent(
        self, agent_number: ReplayBuffer, samples: List, logger: Optional[SummaryWriter]
    ):
        """update the critic and actor of one of the agents"""
        agent = self.maddpg_agents[agent_number]

        # need to transpose each element of the samples
        # to flip obs[parallel_agent][agent_number] to
        # obs[agent_number][parallel_agent] (remember, only 1 parallel agent)
        obs, obs_full, actions, reward, next_obs, next_obs_full, done = map(
            transpose_to_tensor, samples
        )

        # -- REFORMAT THE BATCH OF SAMPLES --
        obs_full = torch.stack(obs_full).squeeze(0).to(device)
        next_obs_full = torch.stack(next_obs_full).squeeze(0).to(device)
        reward = torch.cat(
            [torch.unsqueeze(x, 0) for x in transpose_to_tensor(reward)], dim=0
        ).to(device)
        done = torch.cat(
            [torch.unsqueeze(x, 0) for x in transpose_to_tensor(done)], dim=0
        ).to(device)
        # Now we have the batch of samples split out as follows:
        # 2-element lists, one element for each agent:
        #     obs: per-agent observation tensors (batchsize x 24 observed variables)
        #     actions: per-agent action tensors (batchsize x 2 actions)
        #     done: per-agent "done" tensors (batchsize, values either 0.0 or 1.0)
        # obs_full: combined observations from all agents as a tensor (batchsize x 48),
        #           each row the concatenation of the corresponding values from "obs"
        # rewards: sample rewards as a tensor (batchsize x 2 agents)
        #
        # next_obs and next_obs_full with same structure as obs and obs_full above

        # -- UPDATE THE CRITIC NETWORK --
        # for each replay buffer sample (vectorized across the batch):
        #   for each agent:
        #     use target actor NN and agent's next_obs to get agent's target_actions
        #   concat next_obs_full and all agent target_actions as critic input
        #   use this critic input with agent's target critic to get Qnext
        #   y = sample_reward[agent_number] + (discount * Qnext)
        #
        #   concat obs_full and all agent sample actions as critic input
        #   use this critic input with agent's local critic to get Q
        #   optimize MSE loss between Q and y, updating agent's local critic

        # TODO detach other agents?

        # critic loss = batch mean of (y- Q(s,a) from target network)^2
        # y = reward of this timestep + discount * Q(st+1,at+1) from target network
        target_actions = self.target_act(next_obs)
        target_actions = torch.cat(target_actions, dim=1).to(device)
        target_critic_input = torch.cat((next_obs_full, target_actions), dim=1).to(
            device
        )
        q_next = agent.target_critic(target_critic_input).squeeze(-1)

        y = reward[:, agent_number] + self.discount_factor * q_next * (
            1 - done[:, agent_number]
        )

        local_actions = torch.cat(actions, dim=1).to(device)
        critic_input = torch.cat((obs_full, local_actions), dim=1)
        q = agent.critic(critic_input).squeeze(-1)

        critic_loss = F.mse_loss(q, y.detach())
        agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 1.0)
        agent.critic_optimizer.step()

        # -- UPDATE THE AGENT'S ACTOR NETWORK USING POLICY GRADIENT --
        # for each replay buffer sample (vectorized across the batch):
        #   use local actor NN and agent's obs to get this agent's new actions
        #   concat obs_full and all agent actions as critic input (for this
        #     agent, replace sampled actions with the new actions)
        #   use this critic input with agent's local critic to get Q
        #   negatize Q to use as loss function to optimizer of agent's local actor NN

        update_actions = [x.to(device) for x in actions]  # make a copy
        update_actions[agent_number] = agent.actor(obs[agent_number].to(device))
        update_actions = torch.cat(update_actions, dim=1)

        # combine all the actions and observations for input to critic
        critic_input = torch.cat((obs_full, update_actions), dim=1)

        # get the policy gradient for this agent
        actor_loss = -agent.critic(critic_input).mean(dim=0)
        agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        # torch.nn.utils.clip_grad_norm_(agent.actor.parameters(),0.5)
        agent.actor_optimizer.step()

        if logger:
            al = actor_loss.cpu().detach().item()
            cl = critic_loss.cpu().detach().item()
            logger.add_scalars(
                "agent%i/losses" % agent_number,
                {"critic loss": cl, "actor_loss": al},
                self.iter,
            )
            logger.add_scalars(
                "agent%i/mean_actions" % agent_number,
                {
                    "action0": np.mean(actions[agent_number][:, 0].numpy()),
                    "action1": np.mean(actions[agent_number][:, 1].numpy()),
                },
                self.iter,
            )

    def update_targets(self):
        """soft update targets"""
        self.iter += 1
        for ddpg_agent in self.maddpg_agents:
            soft_update(ddpg_agent.target_actor, ddpg_agent.actor, self.tau)
            soft_update(ddpg_agent.target_critic, ddpg_agent.critic, self.tau)
