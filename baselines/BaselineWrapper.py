from gym import wrappers
from gym import spaces
import gym
import numpy as np

class BaselineTaskWrapper(gym.Wrapper):


    def __init__(self, env, max_episode_steps=200):
        super(BaselineTaskWrapper, self).__init__(env)
        self.reward_type = 'sparse'

        # try to implement Timelimit wrapper
        self.max_episode_steps = max_episode_steps
        self.env.spec.max_episode_steps = max_episode_steps
        self._max_episode_steps = self.max_episode_steps
        self._elapsed_steps = None
        
        obs = self.env.reset()['front_rgb']
        self.observation_space = spaces.Box(0, 255, [*obs.shape], dtype='uint8')

    def reset(self):
        obs = self.env.reset()['front_rgb']
        self._elapsed_steps = 0
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            done = True
        obs = self.env.reset()['front_rgb']
        return obs, reward, done, info