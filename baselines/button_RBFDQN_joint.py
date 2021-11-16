import gym
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import pdb
import rlbench.gym
import sys
import time
import numpy
import random
import utils_for_q_learning, buffer_class

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy
import pickle

#pyrep stuff
from pyrep.robots.end_effectors.panda_gripper import PandaGripper
from pyrep.objects.shape import Shape
from pyrep.robots.arms.panda import Panda
from pyrep.objects.joint import Joint
from pyrep.errors import ConfigurationPathError
from pyrep.objects.shape import Shape

import numpy as np
from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig

from ImageTaskWrapper import ImageTaskWrapper
from BaselineWrapper import BaselineTaskWrapper
from HapticWrapper import HapticWrapper

# stable baseline stuff. 

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

def rbf_function_on_action(centroid_locations, action, beta):
    '''
	centroid_locations: Tensor [batch x num_centroids (N) x a_dim (action_size)]
	action_set: Tensor [batch x a_dim (action_size)]
	beta: float
		- Parameter for RBF function

	Description: Computes the RBF function given centroid_locations and one action
	'''
    assert len(centroid_locations.shape) == 3, "Must pass tensor with shape: [batch x N x a_dim]"
    assert len(action.shape) == 2, "Must pass tensor with shape: [batch x a_dim]"

    diff_norm = centroid_locations - action.unsqueeze(dim=1).expand_as(centroid_locations)
    diff_norm = diff_norm ** 2
    diff_norm = torch.sum(diff_norm, dim=2)
    diff_norm = torch.sqrt(diff_norm + 1e-7)
    diff_norm = diff_norm * beta * -1
    weights = F.softmax(diff_norm, dim=1)  # batch x N
    return weights


def rbf_function(centroid_locations, action_set, beta):
    '''
	centroid_locations: Tensor [batch x num_centroids (N) x a_dim (action_size)]
	action_set: Tensor [batch x num_act x a_dim (action_size)]
		- Note: pass in num_act = 1 if you want a single action evaluated
	beta: float
		- Parameter for RBF function

	Description: Computes the RBF function given centroid_locations and some actions
	'''
    assert len(centroid_locations.shape) == 3, "Must pass tensor with shape: [batch x N x a_dim]"
    assert len(action_set.shape) == 3, "Must pass tensor with shape: [batch x num_act x a_dim]"

    diff_norm = torch.cdist(centroid_locations, action_set, p=2)  # batch x N x num_act
    diff_norm = diff_norm * beta * -1
    weights = F.softmax(diff_norm, dim=2)  # batch x N x num_act
    return weights


class Net(nn.Module):
    def __init__(self, params, env, state_size, action_size, device):
        super(Net, self).__init__()

        self.env = env
        self.device = device
        self.params = params
        self.N = self.params['num_points']
        self.max_a = self.env.action_space.high[0]
        self.beta = self.params['temperature']

        self.buffer_object = buffer_class.ReplayBuffer(size=self.params['max_buffer_size'])

        self.state_size, self.action_size = state_size, action_size


        self.feature_extraction_module = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
        )

        self.value_module = nn.Sequential(
            nn.Linear(9216, self.params['layer_size']),
            nn.ReLU(),
            nn.Linear(self.params['layer_size'], self.params['layer_size']),
            nn.ReLU(),
            nn.Linear(self.params['layer_size'], self.params['layer_size']),
            nn.ReLU(),
            nn.Linear(self.params['layer_size'], self.N),
        )

        if self.params['num_layers_action_side'] == 1:
            self.location_module = nn.Sequential(
                nn.Linear(9216, self.params['layer_size_action_side']),
                nn.Dropout(p=self.params['dropout_rate']),
                nn.ReLU(),
                nn.Linear(self.params['layer_size_action_side'],
                          self.action_size * self.N),
                utils_for_q_learning.Reshape(-1, self.N, self.action_size),
                nn.Tanh(),
            )
        elif self.params['num_layers_action_side'] == 2:
            self.location_module = nn.Sequential(
                nn.Linear(9216, self.params['layer_size_action_side']),
                nn.Dropout(p=self.params['dropout_rate']),
                nn.ReLU(),
                nn.Linear(self.params['layer_size_action_side'],
                          self.params['layer_size_action_side']),
                nn.Dropout(p=self.params['dropout_rate']),
                nn.ReLU(),
                nn.Linear(self.params['layer_size_action_side'],
                          self.action_size * self.N),
                utils_for_q_learning.Reshape(-1, self.N, self.action_size),
                nn.Tanh(),
            )

        torch.nn.init.xavier_uniform_(self.location_module[0].weight)
        torch.nn.init.zeros_(self.location_module[0].bias)

        self.location_module[3].weight.data.uniform_(-.1, .1)
        self.location_module[3].bias.data.uniform_(-1., 1.)

        self.criterion = nn.MSELoss()

        self.params_dic = [{
            'params': self.value_module.parameters(), 'lr': self.params['learning_rate']
        },
        {
            'params': self.location_module.parameters(),
            'lr': self.params['learning_rate_location_side']
        }, 
        {
            'params': self.feature_extraction_module.parameters(),
            'lr': self.params['learning_rate_feature_extraction_module']
        }]
        try:
            if self.params['optimizer'] == 'RMSprop':
                self.optimizer = optim.RMSprop(self.params_dic)
            elif self.params['optimizer'] == 'Adam':
                self.optimizer = optim.Adam(self.params_dic)
            else:
                print('unknown optimizer ....')
        except:
            print("no optimizer specified ... ")

        self.to(self.device)

    def get_centroid_values(self, s):
        '''
		given a batch of s, get all centroid values, [batch x N]
		'''
        batch_size = s.shape[0]
        image_features = self.feature_extraction_module(s)
        image_features = image_features.reshape(batch_size, -1)
        centroid_values = self.value_module(image_features)
        return centroid_values

    def get_centroid_locations(self, s):
        '''
		given a batch of s, get all centroid_locations, [batch x N x a_dim]
		'''
        batch_size = s.shape[0]
        image_features = self.feature_extraction_module(s)
        image_features = image_features.reshape(batch_size, -1)
        centroid_locations = self.max_a * self.location_module(image_features)
        return centroid_locations

    def get_best_qvalue_and_action(self, s):
        '''
		given a batch of states s, return Q(s,a), max_{a} ([batch x 1], [batch x a_dim])
		'''
        all_centroids = self.get_centroid_locations(s)
        values = self.get_centroid_values(s)
        weights = rbf_function(all_centroids, all_centroids, self.beta)  # [batch x N x N]
        allq = torch.bmm(weights, values.unsqueeze(2)).squeeze(2)  # bs x num_centroids
        # a -> all_centroids[idx] such that idx is max(dim=1) in allq
        # a = torch.gather(all_centroids, dim=1, index=indices)
        # (dim: bs x 1, dim: bs x action_dim)
        best, indices = allq.max(dim=1)
        if s.shape[0] == 1:
            index_star = indices.item()
            a = all_centroids[0, index_star]
            return best, a
        else:
            return best, None

    def forward(self, s, a):
        '''
		given a batch of s,a , compute Q(s,a) [batch x 1]
		'''
        centroid_values = self.get_centroid_values(s)  # [batch_dim x N]
        centroid_locations = self.get_centroid_locations(s)
        # [batch x N]
        centroid_weights = rbf_function_on_action(centroid_locations, a, self.beta)
        output = torch.mul(centroid_weights, centroid_values)  # [batch x N]
        output = output.sum(1, keepdim=True)  # [batch x 1]
        return output

    def e_greedy_policy(self, s, episode, train_or_test):
        '''
		Given state s, at episode, take random action with p=eps if training
		Note - epsilon is determined by episode
		'''
        epsilon = 1.0 / numpy.power(episode, 1.0 / self.params['policy_parameter'])
        if train_or_test == 'train' and random.random() < epsilon:
            a = self.env.action_space.sample()
            return a.tolist()
        else:
            self.eval()
            
            with torch.no_grad():
                s = torch.from_numpy(s).float().to(self.device)
                s = torch.unsqueeze(s, axis=0)
                _, a = self.get_best_qvalue_and_action(s)
                a = a.cpu().numpy()
            self.train()
            return a

    def e_greedy_gaussian_policy(self, s, episode, train_or_test):
        '''
		Given state s, at episode, take random action with p=eps if training
		Note - epsilon is determined by episode
		'''
        epsilon = 1.0 / numpy.power(episode, 1.0 / self.params['policy_parameter'])
        if train_or_test == 'train' and random.random() < epsilon:
            a = self.env.action_space.sample()
            return a.tolist()
        else:
            self.eval()
            s_matrix = numpy.array(s).reshape(1, self.state_size)
            with torch.no_grad():
                s = torch.from_numpy(s_matrix).float().to(self.device)
                _, a = self.get_best_qvalue_and_action(s)
                a = a.cpu().numpy()
            self.train()
            noise = numpy.random.normal(loc=0.0,
                                        scale=self.params['noise'],
                                        size=len(a))
            a = a + noise
            return a

    def gaussian_policy(self, s, episode, train_or_test):
        '''
		Given state s, at episode, take random action with p=eps if training
		Note - epsilon is determined by episode
		'''
        self.eval()
        s_matrix = numpy.array(s).reshape(1, self.state_size)
        with torch.no_grad():
            s = torch.from_numpy(s_matrix).float().to(self.device)
            _, a = self.get_best_qvalue_and_action(s)
            a = a.cpu()
        self.train()
        noise = numpy.random.normal(loc=0.0, scale=self.params['noise'], size=len(a))
        a = a + noise
        return a

    def update(self, target_Q, count):
        if len(self.buffer_object) < self.params['batch_size']:
            return 0
        s_matrix, a_matrix, r_matrix, sp_matrix, done_matrix = self.buffer_object.sample(self.params['batch_size'])
        r_matrix = numpy.clip(r_matrix,
                              a_min=-self.params['reward_clip'],
                              a_max=self.params['reward_clip'])

        s_matrix = torch.from_numpy(s_matrix).float().to(self.device)
        a_matrix = torch.from_numpy(a_matrix).float().to(self.device)
        r_matrix = torch.from_numpy(r_matrix).float().to(self.device)
        done_matrix = torch.from_numpy(done_matrix).float().to(self.device)
        sp_matrix = torch.from_numpy(sp_matrix).float().to(self.device)

        Q_star, _ = target_Q.get_best_qvalue_and_action(sp_matrix)
        Q_star = Q_star.reshape((self.params['batch_size'], -1))
        with torch.no_grad():
            y = r_matrix + self.params['gamma'] * (1 - done_matrix) * Q_star
        y_hat = self.forward(s_matrix, a_matrix)
        loss = self.criterion(y_hat, y)
        self.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.zero_grad()
        utils_for_q_learning.sync_networks(
            target=target_Q,
            online=self,
            alpha=self.params['target_network_learning_rate'],
            copy=False)
        return loss.cpu().data.numpy()


    '''
    Chooses between 4 additional subgoals using the future sampling strategy
    As indicated by the paper this is the optimal configuration
    '''
    # trajectory.append((s, a, r, done_p, sp))
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


    # obs_config = ObservationConfig()
    # obs_config.set_all_high_dim(False)
    # obs_config.set_all_low_dim(True)
    # action_mode = ActionMode(ArmActionMode.ABS_JOINT_POSITION)
    # env = Environment(action_mode, obs_config=obs_config, headless=False)

    env = gym.make(params['env_name'])  # render_mode="human"
    #wrap the environment with the custom wrapper to reduce obs space

    # if using the low dimensional shape and you want to prune the state space down. 
    #env = ImageTaskWrapper(env)

    env = BaselineTaskWrapper(env)   
    #env = gym.make(params['env_name'], render_mode="human")  #check env sion
    #replacing Gym with RLBench's default Environment
    #live_demos = True
    #DATASET = ''

    #action_mode = ActionMode(ArmActionMode.ABS_JOINT_VELOCITY)
    #env = Environment(action_mode, DATASET, obs_config, False)
    #env.launch()
    #task = env.task
    #demos = task.get_demos(2, live_demos=live_demos)
    #pdb.set_trace()
    #print('demos: ', demos)


    params['env'] = env
    params['seed_number'] = int(sys.argv[2])
    if len(sys.argv) > 3:
        params['save_prepend'] = str(sys.argv[3])
        print("Save prepend is ", params['save_prepend'])
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

    ## DDPG CNN Stable Baselines code which needs to be fixed
    # The noise objects for DDPG
    # n_actions = env.action_space.shape[-1]
    # action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    # model = DDPG("CnnPolicy", env, action_noise=action_noise, verbose=1, buffer_size=50000)
    # model.learn(total_timesteps=10000, log_interval=10)
    # model.save("ddpg_pendulum")
    # env = model.get_env()

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
        epsilon = 1.0 / numpy.power(episode+1, 1.0 / params['policy_parameter'])
        print("episode {}".format(episode), "epsilon {}".format(epsilon))

        #try to get as much info as possible for the button and arm
        gripper = PandaGripper()
        arm = Panda()
        tip = arm.get_tip()
        target_button_wrap = Shape('target_button_wrap')
        target_button = Shape('push_button_target')
        target_topPlate = Shape('target_button_topPlate')
        joint = Shape('target_button_wrap')

        trajectory = [] # store the transitions from each trajectory here. Is cleared after each trajectory.
        #pdb.set_trace()
        s, done, t = env.reset(), False, 0

        motion_plan_to_above_button(env)

        #print("Initial State is: ", s)

        #print("target_button_wrap's position", target_button_wrap.get_position())
        #print("target_button's position", target_button.get_position())
        #print("target_button_topPlate", target_topPlate.get_position())
        #pdb.set_trace()

        #indicate whether the arm tip has reached the target
        success = False
        while not done:
            if params['policy_type'] == 'e_greedy':
                a = Q_object.e_greedy_policy(s, episode + 1, 'train')
            elif params['policy_type'] == 'e_greedy_gaussian':
                a = Q_object.e_greedy_gaussian_policy(s, episode + 1, 'train')
            elif params['policy_type'] == 'gaussian':
                a = Q_object.gaussian_policy(s, episode + 1, 'train')


            sp, r, done, _ = env.step(numpy.array(a))
            #print("Training loop diff is: ", sp-s)

            haptic_data = sp.gripper_touch_forces

            breakpoint()


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
        if (episode % 10 == 0) or (episode == params['max_episode'] - 1):
            temp = []
            for _ in range(10):
                s, G, done, t = env.reset(), 0, False, 0

                # new s is [arm.get_joint_positions(), tip.get_pose(), target.get_position()]

                while done == False:
                    a = Q_object.e_greedy_policy(s, episode + 1, 'test')
                    sp, r, done, _ = env.step(numpy.array(a))

                    # new s is [arm.get_joint_positions(), tip.get_pose(), target.get_position()]

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
            if numpy.mean(temp) >=0.8 or episode % 50 == 0:
                torch.save(Q_object.state_dict(), './logs/obj_net_button_push' + str(episode) + "_seed_" + str(params['seed_number']))
                torch.save(Q_object_target.state_dict(), './logs/obj_target_net_button_push' +str(episode)+ "_seed_" + str(params['seed_number']))
