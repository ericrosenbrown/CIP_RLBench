from gym import wrappers
from gym import spaces
import gym
import numpy as np

from collections import deque
import cv2

#this wrapper is created to shrink the observation space of the original target reach from RLbench
class ImagePendulumWrapper(gym.Wrapper):


    def __init__(self, env):
        super(ImagePendulumWrapper, self).__init__(env)
        #over ride the original observation space
        self.history_length = 3
        self.state_buffer = deque([], maxlen=self.history_length)



    def reset(self):
        s = self.env.reset()
        obs = self.env.render(mode='rgb_array')

        first_image = cv2.cvtColor(obs[100:400,100:400,:], cv2.COLOR_RGB2GRAY)
        resized_first_image = cv2.resize(first_image, (128,128))
        for _ in range(self.history_length):
            self.state_buffer.append(np.zeros((128, 128)))
        self.state_buffer.append(resized_first_image)
        return np.stack(list(self.state_buffer), axis=0)/255.0

    def step(self, action):
        #print('action:', action)
        obs, reward, done, info = self.env.step(action)
        obs = self.env.render(mode='rgb_array')

        obs_image = cv2.cvtColor(obs[100:400,100:400,:], cv2.COLOR_RGB2GRAY)
        resized_obs_image = cv2.resize(obs_image, (128,128))
        self.state_buffer.append(resized_obs_image)
        return np.stack(list(self.state_buffer), axis=0)/255.0, reward, done, info
