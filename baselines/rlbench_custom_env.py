from typing import Union, Dict, Tuple

import gym
from gym import spaces
from pyrep.const import RenderMode
from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor
from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
import numpy as np
from pyrep.objects.shape import Shape
from pyrep.errors import ConfigurationPathError

class RLBenchCustEnv(gym.Env):
    """An gym wrapper for RLBench."""

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, task_class, observation_mode='state',
                 render_mode: Union[None, str] = None):
        self._observation_mode = observation_mode
        self._render_mode = render_mode
        obs_config = ObservationConfig()
        if observation_mode == 'state' or observation_mode == 'touch_forces':
            obs_config.set_all_high_dim(False)
            obs_config.set_all_low_dim(True)
        elif observation_mode == 'vision':
            obs_config.set_all(True)
        else:
            raise ValueError(
                'Unrecognised observation_mode: %s.' % observation_mode)
        
        action_mode = ActionMode(ArmActionMode.DELTA_EE_POSE_WORLD_FRAME)
        
        #action_mode = ActionMode(ArmActionMode.ABS_JOINT_VELOCITY)
        #action_mode = ActionMode(ArmActionMode.DELTA_EE_POSE_PLAN_WORLD_FRAME)
        self.env = Environment(
            action_mode, obs_config=obs_config, headless=True)
        self.env.launch()
        self.task = self.env.get_task(task_class)

        _, obs = self.task.reset()
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.env.action_size,))
        if observation_mode == 'state':
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=obs.get_low_dim_data().shape)
        elif observation_mode == 'touch_forces':
            #self.observation_space = spaces.Box(-np.inf, np.inf, shape=(17,), dtype='float32')
            target_topPlate = Shape('target_button_topPlate')

            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape =np.array((*obs.gripper_touch_forces, *obs.gripper_pose[:3], *target_topPlate.get_position())).shape) #shape=)        obs.gripper_pose[:3].shape
    
        elif observation_mode == 'vision':
            self.observation_space = spaces.Dict({
                "state": spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=obs.get_low_dim_data().shape),
                "left_shoulder_rgb": spaces.Box(
                    low=0, high=1, shape=obs.left_shoulder_rgb.shape),
                "right_shoulder_rgb": spaces.Box(
                    low=0, high=1, shape=obs.right_shoulder_rgb.shape),
                "wrist_rgb": spaces.Box(
                    low=0, high=1, shape=obs.wrist_rgb.shape),
                "front_rgb": spaces.Box(
                    low=0, high=1, shape=obs.front_rgb.shape),
                })

        if render_mode is not None:
            # Add the camera to the scene
            cam_placeholder = Dummy('cam_cinematic_placeholder')
            self._gym_cam = VisionSensor.create([640, 360])
            self._gym_cam.set_pose(cam_placeholder.get_pose())
            if render_mode == 'human':
                self._gym_cam.set_render_mode(RenderMode.OPENGL3_WINDOWED)
            else:
                self._gym_cam.set_render_mode(RenderMode.OPENGL3)
        self.action_high = np.array((self.task._scene._workspace_maxx-0.01, self.task._scene._workspace_maxy-0.01, self.task._scene._workspace_maxz-0.01 ))
        self.action_low = np.array((self.task._scene._workspace_minx+0.01, self.task._scene._workspace_miny+0.01, self.task._scene._workspace_minz+0.01 ))

    def _extract_obs(self, obs) -> Dict[str, np.ndarray]:
        if self._observation_mode == 'state':
            return obs.get_low_dim_data()
        elif self._observation_mode == 'touch_forces':
            #obs = obs.get_low_dim_data()
            #return [*obs[8:15], *obs[22:29], *obs[66:69]]  #
            target_topPlate = Shape('target_button_topPlate')
            return np.array((*obs.gripper_touch_forces, *obs.gripper_pose[:3], *target_topPlate.get_position()))
        elif self._observation_mode == 'vision':
            return {
                "state": obs.get_low_dim_data(),
                "left_shoulder_rgb": obs.left_shoulder_rgb,
                "right_shoulder_rgb": obs.right_shoulder_rgb,
                "wrist_rgb": obs.wrist_rgb,
                "front_rgb": obs.front_rgb,
            }

    def render(self, mode='human') -> Union[None, np.ndarray]:
        if mode != self._render_mode:
            raise ValueError(
                'The render mode must match the render mode selected in the '
                'constructor. \nI.e. if you want "human" render mode, then '
                'create the env by calling: '
                'gym.make("reach_target-state-v0", render_mode="human").\n'
                'You passed in mode %s, but expected %s.' % (
                    mode, self._render_mode))
        if mode == 'rgb_array':
            return self._gym_cam.capture_rgb()

    def reset(self) -> Dict[str, np.ndarray]:
        descriptions, obs = self.task.reset()
        del descriptions  # Not used.
        return self._extract_obs(obs)

    def step(self, action) -> Tuple[Dict[str, np.ndarray], float, bool, dict]:
        clipped_action = np.array((*np.clip(action[0:3], self.action_low, self.action_high), 1,0,0,0,0))

        try:
            obs, reward, terminate = self.task.step(clipped_action)
        except ConfigurationPathError as e:
            terminate = True
            reward = -1
            obs = self.task._scene.get_observation()
            pass

        
        return self._extract_obs(obs), reward, terminate, {}

    def close(self) -> None:
        self.env.shutdown()