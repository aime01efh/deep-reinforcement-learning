from unityagents import UnityEnvironment
import maddpg_train
import maddpg_agent
import ddpg_agent
import random
import datetime
import pathlib
import numpy as np


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

print("Random values")
hidden_in_actor = random.choice([128, 256])
hidden_out_actor = random.choice([128, 256])
hidden_in_critic = random.choice([256, 512])
hidden_out_critic = random.choice([256, 512])
lr_actor = random.choice([1e-5, 1e-4, 1e-3])
lr_critic = random.choice([1e-5, 1e-4, 1e-3])
dropout = random.choice([0.1, 0.2, 0.3, 0.4])
ou_sigma = random.choice([0.2, 0.3, 1.0])

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
    ou_sigma=ou_sigma,
)
print(ddpg_params)

batchsize = random.choice([256, 512, 1024])
episode_length = random.choice([200, 500, 1000])
update_step_interval = 1
update_iterations = random.choice([1, 3, 5])
discount_factor = random.choice([1.0, 0.999, 0.99, 0.98])
tau = random.choice([1e-4, 1e-3, 1e-2])
initial_noise_scale = random.choice([1.0, 3.0, 5.0])
min_noise_scale = random.choice([0.0, 0.01, 0.1])
episode_noise_end = random.choice([300, 1000, 2500])
replay_buffer_len = random.choice([500_000, 1_000_000])

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
    num_episodes=5000,
    score_history=score_history,
    report_every=200,
    progressbar=False,
)

best_mean = 0.0
WINLEN = 100
for idx in range(len(score_history) - WINLEN):
    this_mean = np.mean(score_history[idx : idx + WINLEN])
    best_mean = max(best_mean, this_mean)

datestr = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
outfile = pathlib.Path("savelog") / f"{datestr}-scores.txt"
outfile.parent.mkdir(exist_ok=True)
outfile.write_text("\n".join([str(x) for x in score_history]) + "\n")
print(f"Best mean {best_mean} : {outfile}")
