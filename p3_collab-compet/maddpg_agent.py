# main code that contains the neural network setup
# policy + critic updates
# see ddpg.py for other details in the network
from typing import List
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
        self.maddpg_critic = DDPGAgent(ddpg_params)

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
        self.maddpg_critic.reset()  # not actually needed

    def update(self, samples, logger):
        """update the critics and actors of all the agents """
        for agent_number in range(self.maddpg_agents):
            self.update_one_agent(agent_number, samples, logger)

    def update_one_agent(self, agent_number, samples, logger):
        """update the critics and actors of all the agents """
        agent = self.maddpg_agents[agent_number]

        # need to transpose each element of the samples
        # to flip obs[parallel_agent][agent_number] to
        # obs[agent_number][parallel_agent] (remember, only 1 parallel agent)
        obs, obs_full, action, reward, next_obs, next_obs_full, done = map(
            transpose_to_tensor, samples
        )

        # -- update the central critic network --
        # for each replay buffer sample (vectorized):
        #   for each agent:
        #     use target actor NN and agent's next_obs to get agent's target_actions
        #   concat next_obs_full and all agent target_actions as critic input
        #   use this critic input with central target critic to get Qnext
        #   y = sample's reward[agent_number] + discounted Qnext[agent_number]
        #
        #   concat obs_full and all agent sample actions as critic input
        #   use this critic input with central local critic to get Q
        #   optimize MSE loss between Q[agent_number] and y, updating local critic

        # TODO detach other agents?

        obs_full = torch.stack(obs_full).squeeze(0).to(device)
        next_obs_full = torch.stack(next_obs_full).squeeze(0).to(device)
        reward = torch.cat(
            [torch.unsqueeze(x, 0) for x in transpose_to_tensor(reward)], dim=0
        ).to(device)
        done = torch.cat(
            [torch.unsqueeze(x, 0) for x in transpose_to_tensor(done)], dim=0
        ).to(device)

        # critic loss = batch mean of (y- Q(s,a) from target network)^2
        # y = reward of this timestep + discount * Q(st+1,at+1) from target network
        target_actions = self.target_act(next_obs)
        target_actions = torch.cat(target_actions, dim=1).to(device)
        target_critic_input = torch.cat((next_obs_full, target_actions), dim=1).to(
            device
        )
        q_next = self.maddpg_critic.target_critic(target_critic_input)[agent_number]

        y = reward[agent_number] + self.discount_factor * q_next * (
            1 - done[agent_number]
        )

        action = torch.cat(action, dim=1).to(device)
        critic_input = torch.cat((obs_full, action), dim=1)
        q = self.maddpg_critic.critic(critic_input)[agent_number]

        critic_loss = F.mse_loss(q, y.detach())
        self.maddpg_critic.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.maddpg_critic.critic.parameters(), 1.0)
        self.maddpg_critic.critic_optimizer.step()

        critic_loss = critic_loss.cpu().detach().item()
        if logger:
            logger.add_scalar("critic_loss", critic_loss, self.iter)

        # -- update each actor network using policy gradient --
        # for each agent:
        #   for each replay buffer sample (vectorized):
        #     use local actor NN and agent's obs to get all agents' new actions
        #       (detach networks of all agents except the one being trained)
        #     concat obs_full and all agent new actions as critic input
        #     use this critic input with central local critic to get Q for all agents
        #     select agent's Q value, negative, use as loss function to optimizer
        #       of agent's local actor NN

        # make input to agent
        # detach the other agents to save time computing derivative
        new_actions = []
        for i, ob in enumerate(obs):
            this_agent_action = self.maddpg_agents[i].actor(ob.to(device))
            if i == agent_number:
                this_agent_action.detach()
            new_actions.append(this_agent_action)
        new_actions = torch.cat(new_actions, dim=1)

        # combine all the actions and observations for input to critic
        # many of the obs are redundant, and obs[1] contains all useful information already
        critic_input = torch.cat((obs_full, new_actions), dim=1)

        # get the policy gradient for this agent
        actor_loss = -self.maddpg_critic.critic(critic_input).mean(dim=0)[agent_number]
        agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        # torch.nn.utils.clip_grad_norm_(agent.actor.parameters(),0.5)
        agent.actor_optimizer.step()

        if logger:
            logger.add_scalar(
                "agent%i/actor_loss" % agent_number,
                actor_loss.cpu().detach().item(),
                self.iter,
            )

    def update_targets(self):
        """soft update targets"""
        self.iter += 1
        for ddpg_agent in self.maddpg_agents:
            soft_update(ddpg_agent.target_actor, ddpg_agent.actor, self.tau)
            soft_update(ddpg_agent.target_critic, ddpg_agent.critic, self.tau)
