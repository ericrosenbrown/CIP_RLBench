from gym import wrappers
from gym import spaces
import gym
import numpy as np
from rlbench.tasks import PushButton

class HapticWrapper():


    def __init__(self, env, max_episode_steps=200):
        super(HapticWrapper, self).__init__(env)
        self.env = env 

        self.task = self.env.get_task(PushButton)

        self.reward_type = 'sparse'

        # try to implement Timelimit wrapper
        self.max_episode_steps = max_episode_steps
        self.env.spec.max_episode_steps = max_episode_steps
        self._max_episode_steps = self.max_episode_steps
        self._elapsed_steps = None
        
        obs = self.env.reset().gripper_touch_forces
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
        breakpoint()
        obs = self.env.reset()['front_rgb']
        return obs, reward, done, info