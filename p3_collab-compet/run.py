from unityagents import UnityEnvironment
import maddpg_train
import maddpg_agent
import ddpg_agent

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

ddpg_params = ddpg_agent.NNParams(
    in_actor=state_size,
    hidden_in_actor=maddpg_train.HIDDEN_IN_ACTOR,
    hidden_out_actor=maddpg_train.HIDDEN_OUT_ACTOR,
    out_actor=action_size,
    in_critic=state_size + num_agents * action_size,
    hidden_in_critic=maddpg_train.HIDDEN_IN_CRITIC,
    hidden_out_critic=maddpg_train.HIDDEN_OUT_CRITIC,
    lr_actor=maddpg_train.LR_ACTOR,
    lr_critic=maddpg_train.LR_CRITIC,
)

agent = maddpg_agent.MADDPG_Agent(
    num_agents=2,
    ddpg_params=ddpg_params,
    discount_factor=maddpg_train.DISCOUNT_FACTOR,
    tau=maddpg_train.TAU,
)

score_history = []
maddpg_train.train_maddpg(
    env, agent, report_every=20, score_history=score_history,
)
