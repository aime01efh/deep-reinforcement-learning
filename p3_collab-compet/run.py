from unityagents import UnityEnvironment
import maddpg_train
import maddpg_agent
import ddpg_agent
# import random

env = UnityEnvironment(file_name="Tennis_Linux/Tennis.x86_64")
# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
action_size = brain.vector_action_space_size
# reset the environment
env_info = env.reset(train_mode=True)[brain_name]
states = env_info.vector_observations
state_size = states.shape[1]
num_agents = len(env_info.agents)

# if True:
#     # Parameter defaults
#     ddpg_params = ddpg_agent.NNParams(
#         in_actor=num_agents * state_size,  # observations from both agents are concatenated
#         hidden_in_actor=maddpg_train.HIDDEN_IN_ACTOR,
#         hidden_out_actor=maddpg_train.HIDDEN_OUT_ACTOR,
#         out_actor=action_size,
#         in_critic=num_agents * state_size + num_agents * action_size,
#         hidden_in_critic=maddpg_train.HIDDEN_IN_CRITIC,
#         hidden_out_critic=maddpg_train.HIDDEN_OUT_CRITIC,
#         lr_actor=maddpg_train.LR_ACTOR,
#         lr_critic=maddpg_train.LR_CRITIC,
#     )
#     print(ddpg_params)

#     agent = maddpg_agent.MADDPG_Agent(
#         num_agents=num_agents,
#         ddpg_params=ddpg_params,
#         discount_factor=maddpg_train.DISCOUNT_FACTOR,
#         tau=maddpg_train.TAU,
#     )

#     score_history = []
#     maddpg_train.train_maddpg(
#         env, agent, report_every=20, score_history=score_history,
#     )

# if True:
#     # Random
#     hidden_in_actor = random.choice([32, 64, 128, 256, 512])
#     hidden_out_actor = random.choice([32, 64, 128, 256, 512])
#     hidden_in_critic = random.choice([32, 64, 128, 256, 512])
#     hidden_out_critic = random.choice([32, 64, 128, 256, 512])
#     lr_actor = random.choice([1e-5, 3e-5, 1e-4, 3e-4, 1e-3])
#     lr_critic = random.choice([1e-5, 3e-5, 1e-4, 3e-4, 1e-3])

#     ddpg_params = ddpg_agent.NNParams(
#         in_actor=num_agents * state_size,  # observations from both agents are concatenated
#         hidden_in_actor=hidden_in_actor,
#         hidden_out_actor=hidden_out_actor,
#         out_actor=action_size,
#         in_critic=num_agents * state_size + num_agents * action_size,
#         hidden_in_critic=hidden_in_critic,
#         hidden_out_critic=hidden_out_critic,
#         lr_actor=lr_actor,
#         lr_critic=lr_critic,
#     )
#     print(ddpg_params)

#     batchsize = random.choice([64, 128, 512, 1024])
#     maddpg_train.REPLAY_BUFFER_LEN = random.choice([1000, 10000, 100000])
#     episode_length = random.choice([80, 1000, 10000])
#     episodes_per_update = random.choice([1, 2, 4, 8])
#     discount_factor = random.choice([1.0, 0.99, 0.98, 0.95])
#     tau = random.choice([1e-4, 1e-3, 1e-2])
#     ou_noise = random.choice([0.5, 1.0, 2.0, 4.0])
#     noise_reduction = random.choice([0.99999, 0.9999, 0.999])
#     print(f'batchsize={batchsize}, '
#           f'replay_buffer_len={maddpg_train.REPLAY_BUFFER_LEN}, '
#           f'episode_length={episode_length}, '
#           f'episodes_per_update={episodes_per_update}, '
#           f'discount_factor={discount_factor},'
#           f'tau={tau}, ou_noise={ou_noise}, noise_reduction={noise_reduction}')

#     agent = maddpg_agent.MADDPG_Agent(
#         num_agents=num_agents,
#         ddpg_params=ddpg_params,
#         discount_factor=discount_factor,
#         tau=tau,
#     )

#     score_history = []
#     maddpg_train.train_maddpg(
#         env, agent, report_every=200, num_episodes=7000,
#         score_history=score_history,
#         batchsize=batchsize,
#         episode_length=episode_length,
#         episodes_per_update=episodes_per_update,
#         ou_noise=ou_noise,
#         noise_reduction=noise_reduction,
#         progressbar=False
#     )

if True:
    # Test values
    hidden_in_actor = 32
    hidden_out_actor = 256
    hidden_in_critic = 64
    hidden_out_critic = 512
    lr_actor = 3e-5
    lr_critic = 1e-4

    ddpg_params = ddpg_agent.NNParams(
        in_actor=num_agents * state_size,  # observations from both agents are concatenated
        hidden_in_actor=hidden_in_actor,
        hidden_out_actor=hidden_out_actor,
        out_actor=action_size,
        in_critic=num_agents * state_size + num_agents * action_size,
        hidden_in_critic=hidden_in_critic,
        hidden_out_critic=hidden_out_critic,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
    )
    print(ddpg_params)

    batchsize = 128
    maddpg_train.REPLAY_BUFFER_LEN = 1000
    episode_length = 80
    episodes_per_update = 1
    discount_factor = 0.99
    tau = 1e-4
    ou_noise = 4.0
    noise_reduction = 0.9999
    print(f'batchsize={batchsize}, '
          f'replay_buffer_len={maddpg_train.REPLAY_BUFFER_LEN}, '
          f'episode_length={episode_length}, '
          f'episodes_per_update={episodes_per_update}, '
          f'discount_factor={discount_factor},'
          f'tau={tau}, noise_reduction={noise_reduction}')

    agent = maddpg_agent.MADDPG_Agent(
        num_agents=num_agents,
        ddpg_params=ddpg_params,
        discount_factor=discount_factor,
        tau=tau,
    )

    score_history = []
    maddpg_train.train_maddpg(
        env, agent, report_every=200, num_episodes=100000,
        score_history=score_history,
        batchsize=batchsize,
        episode_length=episode_length,
        episodes_per_update=episodes_per_update,
        ou_noise=ou_noise,
        noise_reduction=noise_reduction,
    )
