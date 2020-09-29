# main code that contains the neural network setup
# policy + critic updates
# see ddpg.py for other details in the network
from typing import List
from ddpg_agent import DDPGAgent, device
import torch
from utilities import soft_update, transpose_to_tensor  # , transpose_list


class MADDPG_Agent:
    """MADDPG Learning Algorithm

    Manages multiple DDPG agents, each with their own actor and critic
    networks.
    """
    def __init__(self, num_agents, ddpg_params, discount_factor=0.95, tau=0.02):
        # critic input = obs_full + actions = 14+2+2+2=20
        self.maddpg_agent: List[DDPGAgent] = []
        for _ in range(num_agents):
            self.maddpg_agent.append(DDPGAgent(ddpg_params))
        self.maddpg_critic = DDPGAgent(ddpg_params)

        self.discount_factor = discount_factor
        self.tau = tau
        self.iter = 0

    def get_actors(self):
        """get actors of all the agents in the MADDPG object"""
        actors = [ddpg_agent.actor for ddpg_agent in self.maddpg_agent]
        return actors

    def get_target_actors(self):
        """get target_actors of all the agents in the MADDPG object"""
        target_actors = [ddpg_agent.target_actor for ddpg_agent in self.maddpg_agent]
        return target_actors

    def set_train_mode(self, train_mode=True):
        """Enabled or disable training mode for all agents
        """
        for agent in self.maddpg_agent:
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
                for agent, obs in zip(self.maddpg_agent, obs_all_agents)
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
                for ddpg_agent, obs in zip(self.maddpg_agent, obs_all_agents)
            ]
            self.set_train_mode(True)
        return target_actions

    def reset_episode(self):
        """Reset DDPG agents to begin an episode"""
        for agent in self.maddpg_agent:
            agent.reset()
        self.maddpg_critic.reset()  # not actually needed

    def update(self, samples, logger):
        """update the critics and actors of all the agents """

        # need to transpose each element of the samples
        # to flip obs[parallel_agent][agent_number] to
        # obs[agent_number][parallel_agent]
        obs, obs_full, action, reward, next_obs, next_obs_full, done = map(
            transpose_to_tensor, samples
        )

        obs_full = torch.stack(obs_full).squeeze(0)
        next_obs_full = torch.stack(next_obs_full).squeeze(0)
        reward = torch.cat(
            [torch.unsqueeze(x, 0) for x in transpose_to_tensor(reward)], dim=0
        )
        done = torch.cat(
            [torch.unsqueeze(x, 0) for x in transpose_to_tensor(done)], dim=0
        )

        self.maddpg_critic.critic_optimizer.zero_grad()

        # critic loss = batch mean of (y- Q(s,a) from target network)^2
        # y = reward of this timestep + discount * Q(st+1,at+1) from target network
        target_actions = self.target_act(next_obs)
        target_actions = torch.cat(target_actions, dim=1)

        target_critic_input = torch.cat((next_obs_full, target_actions), dim=1).to(
            device
        )

        with torch.no_grad():
            q_next = self.maddpg_critic.target_critic(target_critic_input)

        y = reward + self.discount_factor * q_next * (1 - done)

        action = torch.cat(action, dim=1)
        critic_input = torch.cat((obs_full, action), dim=1).to(device)
        q = self.maddpg_critic.critic(critic_input)

        # huber_loss = torch.nn.SmoothL1Loss()
        # critic_loss = huber_loss(q, y.detach())
        mse_loss = torch.nn.MSELoss()
        critic_loss = mse_loss(q, y.detach())
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.maddpg_critic.critic.parameters(), 0.5)
        self.maddpg_critic.critic_optimizer.step()
        critic_loss = critic_loss.cpu().detach().item()
        logger.add_scalar("critic_loss", critic_loss, self.iter)

        # update actor network using policy gradient

        # make input to agent
        # detach the other agents to save computation
        # saves some time for computing derivative
        for agent_number, agent in enumerate(self.maddpg_agent):
            agent.actor_optimizer.zero_grad()
            all_actions = [
                self.maddpg_agent[i].actor(ob)
                if i == agent_number
                else self.maddpg_agent[i].actor(ob).detach()
                for i, ob in enumerate(obs)
            ]
            all_actions = torch.cat(all_actions, dim=1)

            # combine all the actions and observations for input to critic
            # many of the obs are redundant, and obs[1] contains all useful information already
            q_input = torch.cat((obs_full, all_actions), dim=1)

            # get the policy gradient
            actor_loss = -self.maddpg_critic.critic(q_input).mean(dim=0)[agent_number]
            actor_loss.backward()
            # torch.nn.utils.clip_grad_norm_(agent.actor.parameters(),0.5)
            agent.actor_optimizer.step()
            logger.add_scalar(
                "agent%i/actor_loss" % agent_number,
                actor_loss.cpu().detach().item(),
                self.iter,
            )

    def update_targets(self):
        """soft update targets"""
        self.iter += 1
        for ddpg_agent in self.maddpg_agent:
            soft_update(ddpg_agent.target_actor, ddpg_agent.actor, self.tau)
            soft_update(ddpg_agent.target_critic, ddpg_agent.critic, self.tau)
