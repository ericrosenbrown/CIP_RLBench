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
import os, os.path

import sys


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


for l in range(1000):
	button_folder = "ButtonMasks/"


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


	descriptions, last_obs = task.reset()

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

	front_rgb = last_obs.front_rgb
	front_depth = last_obs.front_depth

	wrist_rgb = last_obs.left_shoulder_rgb
	wrist_depth = last_obs.left_shoulder_depth

	plt.imshow(front_rgb)
	plt.savefig(button_folder+"RGB_"+str(l)+".png")

	filtered_rgb_values = [[255,0,0]]
	#epsilon = 50
	epsilon = 150

	def in_filter(p):
		for pixel_value in filtered_rgb_values:
			if np.linalg.norm(p-pixel_value) < epsilon:
				return True
		return False

	def get_xy_across_image(img):
		valid_xy_pixel_idxs = []
		for x in range(img.shape[0]):
			for y in range(img.shape[1]):
				if in_filter(img[x,y]):
					valid_xy_pixel_idxs.append([x,y])
		return(valid_xy_pixel_idxs)

	xys = get_xy_across_image(front_rgb)
	print("Here!")
	print(xys)
	filtered_front_rgb = np.zeros([front_rgb.shape[0],front_rgb.shape[1],1])
	for xy in xys:
		filtered_front_rgb[xy[0]][xy[1]] = 1

	plt.imshow(filtered_front_rgb)
	plt.savefig(button_folder+"Mask_"+str(l)+".png")


