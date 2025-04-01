# Challenge 07: Proximal Policy Optimization (PPO) and Reinforcement Learning (RL) Tutorial
# Student Name: Kailey Garrison
# NetId: kgarris7
# Program Description: This is a tutorial found online to start learning about PPO and (RL) 
#                      which my team will be utilizing in our final project. It utilizes 
#                      torchrl, tqdm, and gym[mujoco]. It uses Generalized Advantage Estimation
#                      to help stabilize the training. An actor and critic network to decide and
#                      evaluate actions
# Installation Guide:
#   - Python, whatever version should be fine I have the latest 3.13
#   - pip install torchrl
#   - pip install mujoco 
#       - The tutorial shows to install this as pip install gym[mujoco] I always got a pathing error
#       - even after install for mujoco. The only way that worked for me was to install mujoco then
#       - gym and avoid the combined.
#   - pip install gym
#   - pip install tqdm
# Final Project Adaptation:
#   -  Locomotion can be improved by PPO optimization
#   -  Generalized Advantage  Estimation can be used for tasks like walking and obstacle avoidance
#   -  The clipping found here will create smoother transitions in our own traing and help prevent
#      any drastic changes from occuring
#   -  Actor critics will be a big part of our training as well to help determine the correct action
# Sources:
#   - Everything is from: https://pytorch.org/tutorials/intermediate/reinforcement_ppo.html
#       - I just followed the tutorial to start learning the program and environments our gorup
#         will be utilizing
#   - I did have to do a lot of trouble shooting as my gym[mujoco] refused to install for about an
#     hour, this involved a lot of StackOverflow, but no code was used.

from torch import multiprocessing
from collections import defaultdict
import matplotlib.pyplot as plt
import torch
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (Compose, DoubleToFloat, ObservationNorm, StepCounter,TransformedEnv)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm

# This determines what device to use for computation, GPU if available, CPU otherwise
is_fork = multiprocessing.get_start_method() == "fork"
device = (
    torch.device(0)
    if torch.cuda.is_available() and not is_fork
    else torch.device("cpu")
)

# Parameters for training
num_cells = 256
lr = 3e-4
max_grad_norm = 1.0 # Gradient clipping threshold
frames_per_batch = 1000 # Num frames collected per batch
total_frames = 50_000 # Total amt of training frames
sub_batch_size = 64 # Mini batch size for updates
num_epochs = 10 # Training epochs per batch
clip_epsilon = 0.2 # Clipping parameter for PPO, calculated in the formula on the tutorial site

# Discount factor and GAE parameters
gamma = 0.99
lmbda = 0.95
entropy_eps = 1e-4 # Entropy regularization coefficient

# Define environment, this one it the Inverted Double Pendulum for OpenAI in Mujoco
base_env = GymEnv("InvertedDoublePendulum-v4", device=device)
env = TransformedEnv(
    base_env,
    Compose(
        ObservationNorm(in_keys=["observation"]), # Normalize observations
        DoubleToFloat(), # Convert observation types
        StepCounter(), # Count steps in an episode
    ),
)

# Initialize observation normalization stats
env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)
check_env_specs(env) # Double check environment specs are correct
rollout = env.rollout(3)

# Define the actor network, which outputs the parameters of a Gaussian policy
actor_net = nn.Sequential(
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(2 * env.action_spec.shape[-1], device=device),
    NormalParamExtractor(),
)

policy_module = TensorDictModule(
    actor_net, in_keys=["observation"], out_keys=["loc", "scale"]
)

policy_module = ProbabilisticActor(
    module=policy_module,
    spec=env.action_spec,
    in_keys=["loc", "scale"],
    distribution_class=TanhNormal,
    distribution_kwargs={
        "low": env.action_spec.space.low,
        "high": env.action_spec.space.high,
    },
    return_log_prob=True,
)

# Define the critic network
value_net = nn.Sequential(
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(1, device=device),
)

# Wrap the critic network into a ValueOperator
value_module = ValueOperator(
    module=value_net,
    in_keys=["observation"],
)

print("Running policy:", policy_module(env.reset()))
print("Running value:", value_module(env.reset()))

# Make data collector for training
collector = SyncDataCollector(
    env,
    policy_module,
    frames_per_batch=frames_per_batch,
    total_frames=total_frames,
    split_trajs=False,
    device=device,
)

# Store experiences
replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(max_size=frames_per_batch),
    sampler=SamplerWithoutReplacement(),
)

# Generalized Advantage Estimation for advantage computation
advantage_module = GAE(
    gamma=gamma, lmbda=lmbda, value_network=value_module, average_gae=True, device=device,
)

# PPO loss module
loss_module = ClipPPOLoss(
    actor_network=policy_module,
    critic_network=value_module,
    clip_epsilon=clip_epsilon,
    entropy_bonus=bool(entropy_eps),
    entropy_coef=entropy_eps,
    critic_coef=1.0,
    loss_critic_type="smooth_l1",
)

# Optimizer and learning rate scheduler
optim = torch.optim.Adam(loss_module.parameters(), lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optim, total_frames // frames_per_batch, 0.0
)

# Training loop
logs = defaultdict(list)
pbar = tqdm(total=total_frames)
eval_str = ""

for i, tensordict_data in enumerate(collector):
    for _ in range(num_epochs):
        advantage_module(tensordict_data)
        data_view = tensordict_data.reshape(-1)
        replay_buffer.extend(data_view.cpu())
        for _ in range(frames_per_batch // sub_batch_size):
            subdata = replay_buffer.sample(sub_batch_size)
            loss_vals = loss_module(subdata.to(device))
            loss_value = (
                loss_vals["loss_objective"]
                + loss_vals["loss_critic"]
                + loss_vals["loss_entropy"]
            )

            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
            optim.step()
            optim.zero_grad()

    logs["reward"].append(tensordict_data["next", "reward"].mean().item())
    pbar.update(tensordict_data.numel())
    cum_reward_str = (
        f"average reward={logs['reward'][-1]: 4.4f} (init={logs['reward'][0]: 4.4f})"
    )
    logs["step_count"].append(tensordict_data["step_count"].max().item())
    stepcount_str = f"step count (max): {logs['step_count'][-1]}"
    logs["lr"].append(optim.param_groups[0]["lr"])
    lr_str = f"lr policy: {logs['lr'][-1]: 4.4f}"
    if i % 10 == 0:
        with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
            eval_rollout = env.rollout(1000, policy_module)
            logs["eval reward"].append(eval_rollout["next", "reward"].mean().item())
            logs["eval reward (sum)"].append(
                eval_rollout["next", "reward"].sum().item()
            )
            logs["eval step_count"].append(eval_rollout["step_count"].max().item())
            eval_str = (
                f"eval cumulative reward: {logs['eval reward (sum)'][-1]: 4.4f} "
                f"(init: {logs['eval reward (sum)'][0]: 4.4f}), "
                f"eval step-count: {logs['eval step_count'][-1]}"
            )
            del eval_rollout
    pbar.set_description(", ".join([eval_str, cum_reward_str, stepcount_str, lr_str]))
    scheduler.step()

# Output results
plt.figure(figsize=(10, 10))
plt.subplot(2, 2, 1)
plt.plot(logs["reward"])
plt.title("training rewards (average)")
plt.subplot(2, 2, 2)
plt.plot(logs["step_count"])
plt.title("Max step count (training)")
plt.subplot(2, 2, 3)
plt.plot(logs["eval reward (sum)"])
plt.title("Return (test)")
plt.subplot(2, 2, 4)
plt.plot(logs["eval step_count"])
plt.title("Max step count (test)")
plt.savefig("ppo-pytorch-tutorial.png")
plt.show()