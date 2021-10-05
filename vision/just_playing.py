from rlbench.environment import Environment
from rlbench.action_modes import ActionMode, ArmActionMode
from rlbench.tasks import ReachTarget
import numpy as np

action_mode = ActionMode(ArmActionMode.ABS_JOINT_VELOCITY)
env = Environment(action_mode)
env.launch()

task = env.get_task(ReachTarget)
descriptions, obs = task.reset()
while True:
	obs, reward, terminate = task.step(np.random.normal(size=env.action_size))