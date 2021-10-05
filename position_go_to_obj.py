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

action_mode = ActionMode(ArmActionMode.ABS_JOINT_POSITION)
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
    for i, point in enumerate(waypoints):
        #i = 0
        #point = waypoints[0]

        point.start_of_path()

        try:
            path = point.get_path()
        except ConfigurationPathError as e:
            raise DemoError(
                'Could not get a path for waypoint %d.' % i,
                self._active_task) from e

        ext = point.get_ext()
        print("Ext:",ext)

        for joint_pose in chunks(path._path_points,7):
            full_joint_pose = np.concatenate([joint_pose, [1.0]], axis=-1) #gripper open
            obs, reward, terminate = task.step(full_joint_pose)
            #print("Observations:",obs.task_low_dim_state.shape)

        point.end_of_path()

        if "close_gripper()" in ext:
            full_joint_pose = np.concatenate([joint_pose, [0]], axis=-1) #gripper open
            for _ in range(10):
                obs, reward, terminate = task.step(full_joint_pose)

            break

env.shutdown()