from unityagents import UnityEnvironment
import maddpg_train
import maddpg_agent
import ddpg_agent
import random


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

if True:
    print("Random values")
    hidden_in_actor = random.choice([64, 128, 256, 512])
    hidden_out_actor = random.choice([64, 128, 256, 512])
    hidden_in_critic = random.choice([128, 256, 512, 1024])
    hidden_out_critic = random.choice([128, 256, 512, 1024])
    lr_actor = random.choice([1e-5, 1e-4, 1e-3, 1e-2])
    lr_critic = random.choice([1e-5, 1e-4, 1e-3, 1e-2])
    dropout = random.choice([0.1, 0.2, 0.3, 0.4, 0.5])

    ddpg_params = ddpg_agent.NNParams(
        in_actor=in_actor_size,
        hidden_in_actor=hidden_in_actor,
        hidden_out_actor=hidden_out_actor,
        out_actor=action_size,
        in_critic=num_agents * state_size + num_agents * action_size,
        hidden_in_critic=hidden_in_critic,
        hidden_out_critic=hidden_out_critic,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        out_critic=num_agents,
        dropout=dropout,
    )
    print(ddpg_params)

    batchsize = random.choice([256, 512, 1024])
    maddpg_train.REPLAY_BUFFER_LEN = 1_000_000
    episode_length = random.choice([200, 500, 1000])
    episodes_per_update = 1
    update_iterations = random.choice([1, 5, 10])
    discount_factor = random.choice([1.0, 0.99, 0.98])
    tau = random.choice([1e-4, 1e-3, 1e-2])
    ou_noise = 1.0
    initial_noise_scale = random.choice([1.0, 3.0, 5.0])
    episode_noise_end = random.choice([100, 300, 1000, 5000])
    print(
        f"batchsize={batchsize}, "
        f"replay_buffer_len={maddpg_train.REPLAY_BUFFER_LEN}, "
        f"episode_length={episode_length}, "
        f"episodes_per_update={episodes_per_update}, "
        f"update_iteration={update_iterations}, "
        f"discount_factor={discount_factor}, "
        f"tau={tau}, initial_noise_scale={initial_noise_scale}, "
        f"episode_noise_end={episode_noise_end}"
    )

    agent = maddpg_agent.MADDPG_Agent(
        num_agents=num_agents,
        ddpg_params=ddpg_params,
        discount_factor=discount_factor,
        tau=tau,
    )

    score_history = []
    maddpg_train.train_maddpg(
        env,
        agent,
        report_every=200,
        num_episodes=10000,
        score_history=score_history,
        batchsize=batchsize,
        episode_length=episode_length,
        episodes_per_update=episodes_per_update,
        ou_noise=ou_noise,
        noise_scale=initial_noise_scale,
        episode_noise_end=episode_noise_end,
        progressbar=False,
    )

# if True:
#     # Test values
#     hidden_in_actor = 128
#     hidden_out_actor = 512
#     hidden_in_critic = 256
#     hidden_out_critic = 512
#     lr_actor = 0.01
#     lr_critic = 0.01
#     dropout = 0.3

#     ddpg_params = ddpg_agent.NNParams(
#         in_actor=in_actor_size,
#         hidden_in_actor=hidden_in_actor,
#         hidden_out_actor=hidden_out_actor,
#         out_actor=action_size,
#         in_critic=num_agents * state_size + num_agents * action_size,
#         hidden_in_critic=hidden_in_critic,
#         hidden_out_critic=hidden_out_critic,
#         lr_actor=lr_actor,
#         lr_critic=lr_critic,
#         dropout=dropout,
#     )
#     print(ddpg_params)

#     batchsize = 1024
#     maddpg_train.REPLAY_BUFFER_LEN = 1_000_000
#     episode_length = 1000
#     episodes_per_update = 2
#     discount_factor = 0.95  # gamma
#     tau = 1e-2
#     ou_noise = 2.0
#     noise_reduction = 0.999
#     print(f'batchsize={batchsize}, '
#           f'replay_buffer_len={maddpg_train.REPLAY_BUFFER_LEN}, '
#           f'episode_length={episode_length}, '
#           f'episodes_per_update={episodes_per_update}, '
#           f'discount_factor={discount_factor},'
#           f'tau={tau}, noise_reduction={noise_reduction}')

#     agent = maddpg_agent.MADDPG_Agent(
#         num_agents=num_agents,
#         ddpg_params=ddpg_params,
#         discount_factor=discount_factor,
#         tau=tau,
#     )

#     score_history = []
#     maddpg_train.train_maddpg(
#         env, agent, report_every=200, num_episodes=10000,
#         score_history=score_history,
#         batchsize=batchsize,
#         episode_length=episode_length,
#         episodes_per_update=episodes_per_update,
#         ou_noise=ou_noise,
#         noise_reduction=noise_reduction,
#     )
