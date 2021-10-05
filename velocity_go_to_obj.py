from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import ReachTarget
from rlbench.tasks import OpenDrawer
from rlbench.tasks import OpenDoor
from rlbench.tasks import OpenWindow
from rlbench.tasks import ToiletSeatUp
from pyrep.errors import ConfigurationPathError
import numpy as np

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


obs_config = ObservationConfig()
obs_config.set_all_high_dim(False)
obs_config.set_all_low_dim(True)

action_mode = ActionMode(ArmActionMode.ABS_JOINT_VELOCITY)
env = Environment(
    action_mode, obs_config=obs_config, headless=False)
env.launch()

#task = env.get_task(ReachTarget)
task = env.get_task(ToiletSeatUp)
#task = env.get_task(OpenWindow)

episodes = 5
for _ in range(episodes): #Try the task this many times
    task.reset()

    print("Task waypoints:")

    waypoints = task._task.get_waypoints()
    grasped = False

    task._robot.arm.set_control_loop_enabled(True)

    for i, point in enumerate(waypoints):
        done = False
        #i = 0
        #point = waypoints[0]

        point.start_of_path()

        try:
            path = point.get_path()
        except ConfigurationPathError as e:
            raise DemoError(
                'Could not get a path for waypoint %d.' % i,
                self._active_task) from e

        while not done:
            done = path.step()
            task._task.pyrep.step() 

        point.end_of_path()

        ext = point.get_ext()
        print("Ext:",ext)

        if "close_gripper()" in ext:
            done = False
            while not done:
                done = task._robot.gripper.actuate(0.0,0.04)
                task._task.pyrep.step()
            input("gripper closed")
            break


    task._robot.arm.set_control_loop_enabled(False)

env.shutdown()