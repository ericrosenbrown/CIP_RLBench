import gym
import rlbench.gym
import sys
import os
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils_for_q_learning, buffer_class
from RBFDQN import Net
from rlbench_custom_env import RLBenchCustEnv
from utils import motion_plan_to_above_button
from rlbench.tasks import PushButton
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

hyper_parameter_name = sys.argv[1]
alg = 'rbf'
params = utils_for_q_learning.get_hyper_parameters(hyper_parameter_name, alg)
params['hyper_parameters_name'] = hyper_parameter_name

env = RLBenchCustEnv(PushButton,observation_mode='touch_forces',render_mode='human')

#wrap around to decrease observation space
params['env'] = env
params['seed_number'] = int(sys.argv[2])
s0 = env.reset()

# Load the trained agent
Q_object = Net(params,
                   env,
                   state_size=len(s0),
                   action_size=len(env.action_space.low),
                   device=device)
Q_object.load_state_dict(torch.load('./logs/obj_target_net_button_push2650_seed_0'))
Q_object.eval()


# Evaluate the agent
num_episodes = 1000
num_success = 0
for j in range(num_episodes):
    print("episode:", j)
    obs = env.reset()
    motion_plan_to_above_button(env)
    for i in range(200):
        action = Q_object.e_greedy_policy(obs, j + 1, 'test')

        obs, reward, done, info = env.step(action)

        #print("action taken:", action, "finished? ", done)
        
        #env.render()
        if done:
            num_success += 1
            break

print("succes_rate = ", num_success/num_episodes )
