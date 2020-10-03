from unityagents import UnityEnvironment
import maddpg_train
import maddpg_agent
import ddpg_agent


env = UnityEnvironment(file_name="Tennis_Linux/Tennis.x86_64", no_graphics=True)
# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
action_size = brain.vector_action_space_size
# reset the environment
env_info = env.reset(train_mode=True)[brain_name]
states = env_info.vector_observations
state_size = states.shape[1]
num_agents = len(env_info.agents)

# If you switch this, also change obs in maddpg_train.get_train_obs
# in_actor_size = num_agents * state_size  # obs from both agents are concatenated
in_actor_size = state_size  # each actor gets only its own observations


# Test values
in_actor_size = 24
hidden_in_actor = 256
hidden_out_actor = 128
hidden_in_critic = 256
hidden_out_critic = 256
lr_actor = 5e-4
lr_critic = 1e-3
dropout = 0.2
ou_sigma = 0.2

ddpg_params = ddpg_agent.NN_Params(
    in_actor=in_actor_size,
    hidden_in_actor=hidden_in_actor,
    hidden_out_actor=hidden_out_actor,
    out_actor=action_size,
    in_critic=num_agents * state_size + num_agents * action_size,
    hidden_in_critic=hidden_in_critic,
    hidden_out_critic=hidden_out_critic,
    lr_actor=lr_actor,
    lr_critic=lr_critic,
    out_critic=1,
    dropout=dropout,
    ou_sigma=0.2,
)
print(ddpg_params)

batchsize = 512
episode_length = 512
update_step_interval = 1
update_iterations = 1
discount_factor = 0.99
tau = 1e-3
initial_noise_scale = 1.0
min_noise_scale = 0.0
episode_noise_end = 300
replay_buffer_len = 1_000_000

maddpg_params = maddpg_train.MADDPG_Params(
    batchsize=batchsize,
    episode_length=episode_length,
    update_step_interval=update_step_interval,
    update_iterations=update_iterations,
    discount_factor=discount_factor,
    tau=tau,
    initial_noise_scale=initial_noise_scale,
    min_noise_scale=min_noise_scale,
    episode_noise_end=episode_noise_end,
    replay_buffer_len=replay_buffer_len,
)
print(maddpg_params)

main_agent = maddpg_agent.MADDPG_Agent(
    num_agents=num_agents,
    ddpg_params=ddpg_params,
    discount_factor=maddpg_params.discount_factor,
    tau=maddpg_params.tau,
)

score_history = []
maddpg_train.train_maddpg(
    env,
    main_agent,
    maddpg_params=maddpg_params,
    num_episodes=10000,
    score_history=score_history,
    report_every=100,
)
