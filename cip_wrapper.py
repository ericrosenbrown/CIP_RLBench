import gym
from typing import Union, Dict, Tuple
import numpy as np

def chunks(lst, n):
	"""Yield successive n-sized chunks from lst."""
	for i in range(0, len(lst), n):
		yield lst[i:i + n]

class CIPWrapper(gym.Wrapper):
	def __init__(self, env):
		super(CIPWrapper, self).__init__(env)
		self.env = env

	def reset(self):
		descriptions, _ = self.task.reset()

		######## CIP ##########
		finished = False
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
					break

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

					self.task._robot.arm.set_control_loop_enabled(False)
					finished = True
			self.task._robot.arm.set_control_loop_enabled(False)

			obs = self.task.get_observation()
			return self.env._extract_obs(obs)

if __name__ == "__main__":
	import gym
	import rlbench.gym

	#real_env = gym.make('reach_target-state-v0', render_mode="human")
	real_env = gym.make('toilet_seat_down-state-v0', render_mode="human")
	cip_env = CIPWrapper(real_env)
	# Alternatively, for vision:
	# env = gym.make('reach_target-vision-v0')

	training_steps = 120
	episode_length = 40
	for i in range(training_steps):
		if i % episode_length == 0:
			print('Reset Episode')
			obs = cip_env.reset()
		obs, reward, terminate, _ = cip_env.step(cip_env.action_space.sample())
		cip_env.render()  # Note: rendering increases step time.

	print('Done')
	cip_env.close()
