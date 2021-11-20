import gym
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import rlbench.gym
import random
import utils_for_q_learning, buffer_class
import numpy
import numpy as np
import pickle
import torch
#pyrep stuff
from pyrep.robots.end_effectors.panda_gripper import PandaGripper
from pyrep.objects.shape import Shape
from pyrep.robots.arms.panda import Panda
from pyrep.objects.joint import Joint
from pyrep.errors import ConfigurationPathError
from pyrep.objects.shape import Shape

from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig

def choose_random_subgoals(trajectory, transition_index):
    num_subgoals = 4

    subgoals_index = []

    for i in range(num_subgoals):
        index = random.randint(transition_index, len(trajectory)-1)
        #print('subgoal_index: ', index)

        subgoals_index.append(index)
    return subgoals_index

def reward_according_to_subgoal(subgoal, observation, err=1e-2):
    assert len(subgoal) == 3 and len(observation) == 3
    distance = np.linalg.norm(np.array(subgoal)-np.array(observation))
    if (distance < err):
        #print("added reward!")
        return 1 # we're at our subgoal, according to the margin of error
    else:
        return -0.01

'''
motion plans the arm to right above the button
'''
def motion_plan_to_above_button(env):
    task = env.task

    waypoints = task._task.get_waypoints()

    task._robot.arm.set_control_loop_enabled(True)

    done = False
    
    point_start = waypoints[0]
    point_end = waypoints[len(waypoints) - 1]
    point = waypoints[0]

    target_button = Shape('push_button_target')

    right_above_button = [target_button.get_position()[0], target_button.get_position()[1], 0.9]

    point.get_waypoint_object().set_position(right_above_button)

    point.start_of_path()
    i = 0; 
    path = None
    while path is None:     
        try:
            path = point.get_path()
        except ConfigurationPathError as e:
            right_above_button[2] -= 0.1
            point.get_waypoint_object().set_position(right_above_button)
            pass
        i += 1
        if (i > 10):
            # path not able to be found
            continue 
    i = 0
    while not done:
        print("moving to right above button: " + str(i))
        done = path.step()
        task._task.pyrep.step() 
        i += 1
    point.end_of_path()
    task._robot.arm.set_control_loop_enabled(False)

