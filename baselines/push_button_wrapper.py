from gym import wrappers
from gym import spaces
import gym
import numpy as np

#this wrapper is created to shrink the observation space of the original target reach from RLbench
class Push_button_wrapper(gym.Wrapper):


    def __init__(self, env):
        super(Push_button_wrapper, self).__init__(env)
        #over ride the original observation space

        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(17,), dtype='float32')


    def reset(self):
        obs = self.env.reset()
        # trim the obs space, only keep the stuff we need
        # new obs is [arm.get_joint_positions(), tip.get_pose(), target_button_topPlate]
        obs = [*obs[8:15], *obs[22:29], *obs[66:69]]

        return obs

    def step(self, action):
        #print('action:', action)
        obs, reward, done, info = self.env.step(action)
        obs = [*obs[8:15], *obs[22:29], *obs[66:69]]
        return obs, reward, done, info
