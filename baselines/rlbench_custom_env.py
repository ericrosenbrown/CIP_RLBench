from os import terminal_size
from typing import Union, Dict, Tuple

import gym
from gym import spaces
from pyrep.const import RenderMode
from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.objects import Camera
from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.task_environment import InvalidActionError
from rlbench.observation_config import ObservationConfig
import numpy as np
from pyrep.objects.shape import Shape
from pyrep.errors import ConfigurationPathError
from pyrep.robots.end_effectors.panda_gripper import PandaGripper
from pyrep.objects.shape import Shape
from pyrep.robots.arms.panda import Panda
import rlbench

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
        
        #action_mode = ActionMode(ArmActionMode.DELTA_EE_POSE_PLAN_WORLD_FRAME)
        
        action_mode = ActionMode(ArmActionMode.ABS_JOINT_VELOCITY)
        #action_mode = ActionMode(ArmActionMode.DELTA_EE_POSE_PLAN_WORLD_FRAME)
        self.env = Environment(action_mode, obs_config=obs_config, headless=True)
        self.env.launch()
        self.task = self.env.get_task(task_class)

        camera = Dummy('cam_cinematic_placeholder')
        camera.set_pose([-0.175, 0, 2.43, 0, 1, 0, 0])
        
        _, obs = self.task.reset()
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.env.action_size,)) # don't include the quaternion or gripper open-close state in the action space for now. 
        if observation_mode == 'state':
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=obs.get_low_dim_data().shape)
        elif observation_mode == 'touch_forces':

            # some thoughts            
            # In the future we should find a more elegant way of doing this agnostic to task
            # for now, our goal needs to be manually appended to the observation space. 
            
            # note the best way to figure out what the goal is to open up the task ttm in CoppeliaSim
            # Once you load the coppelia sim GUI on my (executable in ./CoppeliaSim_Edu_V4_1_0_Ubuntu20_04/coppeliaSim)

            # click file --> load model --> and select the ttm (located in rlbench task_ttms)

            # then poke around until you find the waypoint you want to use as a goal for the agent to reach
            # usually this is the key piece of the task which needs to be manipulated for successful completion (such as a door handle etc)
            
            goal = None 
            if self.task.get_name() == "open_drawer":
                goal = Dummy("waypoint1")
            else:
                goal = Shape('target_button_topPlate')
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(12,)) #shape=)        obs.gripper_pose[:3].shape
    
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
            if self.task.get_name() == "open_drawer":
                goal = Dummy("waypoint1")
                return np.array((*obs.gripper_touch_forces, *obs.gripper_pose[:3], *goal.get_position()))  
            else:
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

    '''
    Called within reset to orient the arm in position favorable for task. 
    Use waypoints to move the arm to a good starting point for learning. 
    '''
    def initialize_agent_for_task(self):
        
        if self.task.get_name() == "open_drawer":
            return self.plan_to_handle()
        else:
            return self.goto_button()

    def reset(self) -> Dict[str, np.ndarray]:
        descriptions, obs = self.task.reset()
        del descriptions  # Not used.
        return self.initialize_agent_for_task() #self._extract_obs(obs)

    def step(self, action) -> Tuple[Dict[str, np.ndarray], float, bool, dict]:

        #normalize quaternion
        orientation = np.array(action[3:7])/np.linalg.norm(action[3:7])

        clipped_action = np.array(( action[0]*0.01,action[1]*0.01, action[2]*0.01, *orientation, action[7]))
        try:
            obs, reward, terminate = self.task.step(clipped_action)
        except ConfigurationPathError as e:
            terminate = True
            reward = -1
            obs = self.task._scene.get_observation()
            pass
        except InvalidActionError as e:
            #print("Target is outside of workspace!!!!.")
            terminate = True
            reward = -1
            obs = self.task._scene.get_observation()
            pass    
        
        return self._extract_obs(obs), reward, terminate, {}

    def plan_to_handle(self):
        # motion plan to the drawer handle with the gripper open
        goal = Dummy("waypoint1")
        arm = Panda()
        tip = arm.get_tip()
        
        finished = False
        should_terminate_env = False

        while not finished:
            waypoints = self.task._task.get_waypoints()
            grasped = False

            self.task._robot.arm.set_control_loop_enabled(True)

            for i, point in enumerate(waypoints):
                done = False
                #i = 0
                #point = waypoints[0]

                point.start_of_path()

                try:
                    path = point.get_path()
                except ConfigurationPathError as e:
                    print("=============== COULDN'T FIND PATH, RESET TASK!================")
                    finished = True
                    should_terminate_env = True

                ext = point.get_ext()
                #print("Ext:",ext)

                while not done:
                    done = path.step()
                    self.task._task.pyrep.step() 

                point.end_of_path()

                if "close_gripper()" in ext:
                    done = False
                    while not done:
                        done = self.task._robot.gripper.actuate(0.0,0.04)
                        self.task._task.pyrep.step()
                    finished = True
                    break
        self.task._robot.arm.set_control_loop_enabled(False)

        obs = self.task._scene.get_observation()
        return self._extract_obs(obs), 0, should_terminate_env, {}

    def goto_button(self):
        target_topPlate = Shape('target_button_topPlate')
        arm = Panda()
        tip = arm.get_tip()
        distance = target_topPlate.get_position()-tip.get_position()
        action = np.array((distance[0], distance[1], distance[2]+0.05, 0,0,0,1,0))
        try:
            obs, reward, terminate = self.task.step(action)
        except ConfigurationPathError as e:
            terminate = True
            reward = -1
            obs = self.task._scene.get_observation()
            pass
        
        return self._extract_obs(obs), reward, terminate, {}

    def close(self) -> None:
        self.env.shutdown()