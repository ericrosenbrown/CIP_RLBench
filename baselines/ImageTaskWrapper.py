from gym import wrappers
from gym import spaces
import gym
import numpy as np

from collections import deque
import cv2

#this wrapper is created to shrink the observation space of the original target reach from RLbench
class ImageTaskWrapper(gym.Wrapper):

    def __init__(self, env):
        super(ImageTaskWrapper, self).__init__(env)
        #over ride the original observation space
        self.history_length = 3
        self.state_buffer = deque([], maxlen=self.history_length)

    def reset(self):
        obs = self.env.reset()
        first_image = cv2.cvtColor(obs['front_rgb'], cv2.COLOR_RGB2GRAY)
        for _ in range(self.history_length):
            self.state_buffer.append(np.zeros((128, 128)))
        self.state_buffer.append(first_image)
        return np.stack(list(self.state_buffer), axis=0)/255.0

    def step(self, action):
        #print('action:', action)
        obs, reward, done, info = self.env.step(action)
        obs_image = cv2.cvtColor(obs['front_rgb'], cv2.COLOR_RGB2GRAY)
        self.state_buffer.append(obs_image)
        return np.stack(list(self.state_buffer), axis=0)/255.0, reward, done, info
