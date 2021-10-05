from rlbench.environment import Environment
from rlbench.action_modes import ActionMode, ArmActionMode
from rlbench.tasks import OpenDrawer
from rlbench.tasks import PushButton
import numpy as np
from rlbench.observation_config import ObservationConfig, CameraConfig


import matplotlib.pyplot as plt
import numpy as np
import math

from open3d import *

import grasp_pose_generator as gpg

from pyrep.objects.vision_sensor import VisionSensor

import gym
import rlbench.gym

from scipy.spatial.transform import Rotation

def posRotMat2Mat(pos, rot_mat):
    t_mat = np.eye(4)
    t_mat[:3, :3] = rot_mat
    t_mat[:3, 3] = np.array(pos)
    return t_mat

def quat2Mat(quat):
    if len(quat) != 4:
        print("Quaternion", quat, "invalid when generating transformation matrix.")
        raise ValueError
    # MuJoCo and Open3D use wxyz, scipy uses xyzw
    quat_xyzw = quat[1:] + [quat[0]]
    quat_scipy = Rotation(quat_xyzw)
    return quat_scipy.as_matrix()

def depthimg2Meters(depth, near, far, extent):
		near = near * extent
		far = far * extent
		image = near / (1 - depth * (1 - near / far))
		return image


def chunks(lst, n):
	"""Yield successive n-sized chunks from lst."""
	for i in range(0, len(lst), n):
		yield lst[i:i + n]


action_mode = ActionMode(ArmActionMode.ABS_JOINT_VELOCITY)
#front_cam_config = CameraConfig(depth_in_meters=True)
#left_shoulder_cam_config = CameraConfig(depth_in_meters=True)
#obs_config = ObservationConfig(left_shoulder_camera=left_shoulder_cam_config,front_camera=front_cam_config)
#obs_config.set_all(True)
env = Environment(action_mode)
env.launch()

#task = env.get_task(ReachTarget)
task = env.get_task(PushButton)
#task = env.get_task(OpenWindow)
#task = env.get_task(OpenDrawer)

descriptions, obs = task.reset()

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

	last_obs = task.get_observation()

task._robot.arm.set_control_loop_enabled(False)

my_front_cam = VisionSensor('cam_front')
my_wrist_cam = VisionSensor('cam_over_shoulder_left')

#clipping planes:
near = my_front_cam.get_near_clipping_plane()
far = my_front_cam.get_far_clipping_plane()

print("front cam position:")
#print(my_front_cam.__dir__())
#print(my_front_cam.get_perspective_angle())
front_cam_pos = my_front_cam.get_position()
front_cam_quat = my_front_cam.get_quaternion()

#x = front_cam_quat[0]
#w = front_cam_quat[3]

#front_cam_quat[0] = w
#front_cam_quat[3] = x

front_cam_rotation_matrix = open3d.geometry.get_rotation_matrix_from_quaternion(front_cam_quat)
print(front_cam_pos)
print("wrist cam position:")
#print(my_front_cam.__dir__())
#print(my_front_cam.get_perspective_angle())
wrist_cam_pos = my_wrist_cam.get_position()
wrist_cam_quat = my_wrist_cam.get_quaternion()

#x = wrist_cam_quat[0]
#w = wrist_cam_quat[3]

#wrist_cam_quat[0] = w
#wrist_cam_quat[3] = x

wrist_cam_rotation_matrix = open3d.geometry.get_rotation_matrix_from_quaternion(wrist_cam_quat)

print(wrist_cam_pos)
env.shutdown()

print("Grasp pose:",last_obs.gripper_pose[:3])
gripper_pos = last_obs.gripper_pose[:3]




#########################################################################################

front_rgb = obs.front_rgb
front_depth = obs.front_depth

wrist_rgb = obs.left_shoulder_rgb
wrist_depth = obs.left_shoulder_depth

print(front_depth)

#extent = 1
#front_depth = depthimg2Meters(front_depth,near,far,extent)



H = front_rgb.shape[0]
W = front_rgb.shape[1]

front_rgb_image = open3d.geometry.Image(front_rgb)
front_depth_image = open3d.geometry.Image(front_depth)

wrist_rgb_image = open3d.geometry.Image(wrist_rgb)
wrist_depth_image = open3d.geometry.Image(wrist_depth)

#########################################################################################

#intrinsics = open3d.camera.PinholeCameraIntrinsic(
#		open3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
#print(intrinsics.intrinsic_matrix)

#intrinsics = open3d.camera.PinholeCameraIntrinsic(width=128,height=128,fx=525,fy=525,cx=319.5,cy=239.5)
#intrinsics = open3d.camera.PinholeCameraIntrinsic(width=128,height=128,fx=525,fy=128,cx=319.5,cy=239.5)

intrinsics = open3d.camera.PinholeCameraIntrinsic(width=128,height=128,fx=175.8385604,fy=175.8385604,cx=64,cy=64) #39.99999883637168
#intrinsics = open3d.camera.PinholeCameraIntrinsic(width=128,height=128,fx=202.98206735 ,fy=202.98206735,cx=64,cy=64) #35
#intrinsics = open3d.camera.PinholeCameraIntrinsic(width=128,height=128,fx=175.83855485,fy=175.83855485,cx=64,cy=64) #40
#intrinsics = open3d.camera.PinholeCameraIntrinsic(width=128,height=128,fx=137.24844291,fy=137.24844291,cx=64,cy=64) #60

#40 is best I found
# List of camera intrinsic matrices
fovy = math.radians(39.99999883637168)
f = H / (2 * math.tan(fovy / 2))
cam_mat = np.array(((f, 0, W / 2), (0, f, H / 2), (0, 0, 1)))
print(cam_mat)

#depthimg2Meters


front_rgbd_image = open3d.geometry.RGBDImage.create_from_color_and_depth(front_rgb_image,front_depth_image)
#pcd = open3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
front_pcd = open3d.geometry.PointCloud.create_from_rgbd_image(
	front_rgbd_image,
	intrinsics)

wrist_rgbd_image = open3d.geometry.RGBDImage.create_from_color_and_depth(wrist_rgb_image,wrist_depth_image)
#pcd = open3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
wrist_pcd = open3d.geometry.PointCloud.create_from_rgbd_image(
	wrist_rgbd_image,
	intrinsics)

#front_pcd.rotate(front_cam_rotation_matrix)
#wrist_pcd.rotate(wrist_cam_rotation_matrix)

#front_pcd.translate(front_cam_pos)
#wrist_pcd.translate(wrist_cam_pos)


#front_b2w_r = quat2Mat([0, 1, 0, 0])
#front_c2w_r = np.matmul(front_cam_rotation_matrix, front_b2w_r)
front_c2w = posRotMat2Mat(front_cam_pos, front_cam_rotation_matrix)
front_pcd = front_pcd.transform(front_c2w)

#wrist_b2w_r = quat2Mat([0, 1, 0, 0])
#wrist_c2w_r = np.matmul(wrist_cam_rotation_matrix, wrist_b2w_r)
wrist_c2w = posRotMat2Mat(wrist_cam_pos, wrist_cam_rotation_matrix)
wrist_pcd = wrist_pcd.transform(wrist_c2w)

pcd = front_pcd #+ wrist_pcd



# flip the orientation, so it looks upright, not upside-down
pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])

#open3d.visualization.draw_geometries([pcd])    # visualize the point cloud

#########################################################################################
pcd.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=250))

#open3d.visualization.draw_geometries([pcd])    # visualize the point cloud

#########################################################################################
pose_gen = gpg.GraspPoseGenerator(pcd, rotation_values_about_approach=[0, math.pi/2])

index = 1500*10


#visualization point cloud location we want to grasp
pc_loc = pose_gen.cloud_points[index]
print("POINT IS:",pc_loc)
mesh_sphere = open3d.geometry.TriangleMesh.create_sphere(radius=0.0000025)
mesh_sphere.compute_vertex_normals()
mesh_sphere.translate(pc_loc)
mesh_sphere.paint_uniform_color([0.1, 0.9, 0.1])

#calculate grasp poses for that point cloud location
grasp_pose = pose_gen.proposeGraspPosesAtCloudIndex(index)

#visualization of a grasp pose
#mesh_cylinder = open3d.geometry.TriangleMesh.create_cylinder(radius = 0.000005, height = 0.00003)
#mesh_cylinder.compute_vertex_normals()
#mesh_cylinder.transform(grasp_pose[0])
axes = open3d.geometry.TriangleMesh.create_coordinate_frame()
scaling_maxtrix = np.ones((4,4))
scaling_maxtrix[:3, :3] = scaling_maxtrix[:3, :3]/100000
scaled_pose = grasp_pose[0]*scaling_maxtrix
axes.transform(scaled_pose)

######## VISUALIZE WHERE I THINK SUGGESTED GRASP POSITION WAS
mesh_sphere2 = open3d.geometry.TriangleMesh.create_sphere(radius=0.0000025)
mesh_sphere2.compute_vertex_normals()
#mesh_sphere2.translate([0,0,0])
mesh_sphere2.translate(front_cam_pos)
mesh_sphere2.paint_uniform_color([0.9, 0.1, 0.1])

######## 0 0 0 sphere
mesh_sphere3 = open3d.geometry.TriangleMesh.create_sphere(radius=0.0000025)
mesh_sphere3.compute_vertex_normals()
mesh_sphere3.translate([0,0,0])
#mesh_sphere3.translate(gripper_pos)
mesh_sphere3.paint_uniform_color([0.1, 0.1, 0.9])

open3d.visualization.draw_geometries([pcd])#, mesh_sphere, mesh_sphere3, axes])    # visualize the point cloud