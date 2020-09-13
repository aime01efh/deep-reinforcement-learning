from unityagents import UnityEnvironment
import ppo
import ppo_agent

env = UnityEnvironment(file_name="Reacher_Linux_20agents/Reacher.x86_64")
# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

agent = ppo_agent.Agent(state_size=len(env_info.vector_observations[0]),
                        action_size=brain.vector_action_space_size,
                        hidden_sizes=[256, 512],
                        seed=237)

mean_rewards = []
ppo.train_ppo(env, agent, report_every=20, mean_rewards_per_episode=mean_rewards)
