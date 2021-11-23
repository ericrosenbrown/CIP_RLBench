import gym
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import pdb
import rlbench.gym
import time
import random
import utils_for_q_learning, buffer_class
import numpy
import numpy as np
import pickle
import torch
from pyrep.robots.end_effectors.panda_gripper import PandaGripper
from pyrep.objects.shape import Shape
from pyrep.robots.arms.panda import Panda
from pyrep.objects.joint import Joint
from pyrep.errors import ConfigurationPathError
from pyrep.objects.shape import Shape

from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig

from BaselineWrapper import BaselineTaskWrapper
from HapticWrapper import HapticWrapper
from RBFDQN import Net
from utils import motion_plan_to_above_button
from rlbench.tasks import PushButton
from rlbench_custom_env import RLBenchCustEnv


if __name__ == '__main__':
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

    env = RLBenchCustEnv(PushButton, observation_mode='touch_forces') #, render_mode='human'

    params['env'] = env
    params['seed_number'] = int(sys.argv[2])

    utils_for_q_learning.set_random_seed(params)

    s0 = env.reset()

    utils_for_q_learning.action_checker(env)

    Q_object = Net(params,
                   env,
                   state_size=len(s0),
                   action_size=len(env.action_space.low),
                   device=device)
    Q_object_target = Net(params,
                          env,
                          state_size=len(s0),
                          action_size=len(env.action_space.low),
                          device=device)
    Q_object_target.eval()

    utils_for_q_learning.sync_networks(target=Q_object_target,
                                       online=Q_object,
                                       alpha=params['target_network_learning_rate'],
                                       copy=True)

    G_li = []
    loss_li = []
    all_times_per_steps = []
    all_times_per_updates = []
    max_episode_steps = 200
    max_episode_steps_eval = 200
    # main training loop

    for episode in range(params['max_episode']):
        epsilon = 1.0 / np.power(episode+1, 1.0 / params['policy_parameter'])
        print("episode {}".format(episode), "epsilon {}".format(epsilon))

        #try to get as much info as possible for the button and arm
        gripper = PandaGripper()
        arm = Panda()
        tip = arm.get_tip()
        target_button_wrap = Shape('target_button_wrap')
        target_button = Shape('push_button_target')
        target_topPlate = Shape('target_button_topPlate')
        joint = Shape('target_button_wrap')
        s, done, t = env.reset(), False, 0

        motion_plan_to_above_button(env)

        success = False
        while not done:
            a = Q_object.execute_policy(s, episode + 1, 'train')
          
            sp, r, done, _ = env.step(numpy.array(a))

            #print(sp)
            if done:
                print('reach target!', 'reward: ', r, 'done in', t, 'steps')
                success = True

            t = t + 1
            done_p = False if t == max_episode_steps else done
            # end the current episode
            if t == max_episode_steps:
                done = True
            # we add original transition here for better performance
            Q_object.buffer_object.add(s, a, r, sp, done_p)

            s = sp

        # now update the Q network
        loss = []

        for count in range(params['updates_per_episode']):

            temp = Q_object.update(Q_object_target, count)
            loss.append(temp)

        loss_li.append(numpy.mean(loss))

        # tracking the agent's performance
        if (episode % 10 == 0 and episode>5) or (episode == params['max_episode'] - 1):
            temp = []
            for _ in range(10):
                s, G, done, t = env.reset(), 0, False, 0

                while done == False:
                    a = Q_object.e_greedy_policy(s, episode + 1, 'test')
                    sp, r, done, _ = env.step(numpy.array(a))
                    s, G, t = sp, G + r, t + 1
                    #if can't find within 200 steps, end it.
                    if t == max_episode_steps_eval:
                    	done = True
                temp.append(G)
            print(
                "after {} episodes, learned policy collects {} average returns".format(
                    episode, numpy.mean(temp)))

            G_li.append(numpy.mean(temp))
            utils_for_q_learning.save(G_li, loss_li, params, alg)
            if numpy.mean(temp) >=0.2 and episode % 50 == 0:
                torch.save(Q_object.state_dict(), './logs/obj_net_button_push' + str(episode) + "_seed_" + str(params['seed_number']))
                torch.save(Q_object_target.state_dict(), './logs/obj_target_net_button_push' +str(episode)+ "_seed_" + str(params['seed_number']))
