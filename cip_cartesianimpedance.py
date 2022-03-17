import rospy 
import tf2_ros
from iiwa_msgs.msg import CartesianImpedanceControlMode, CartesianPose, JointVelocity, JointPosition
from iiwa_msgs.srv import ConfigureControlMode
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState

import copy
import numpy as np

CONTROL_FREQ = 10
SAFETY_FACTOR = 0.8

class ArmCommand(object):
    """docstring for ArmCommand"""
    def __init__(self):
        
        # reference frame of pose commands 
        self.frame = None

        # cartesian pose
        self.x = None
        self.y = None 
        self.z = None
        self.qx = None 
        self.qy = None
        self.qz = None
        self.qw = None 

        # target cartesian pose 
        self.target_x = None
        self.target_y = None
        self.target_z = None
        self.target_qx = None
        self.target_qy = None
        self.target_qz = None
        self.target_qw = None

        # joint data 
        self.qpos = np.zeros(7)
        self.qvel = np.zeros(7)
        self.effort = np.zeros(7)

        # cartesian pose pub 
        self.pub_cart = rospy.Publisher("/iiwa_left/command/CartesianPose", PoseStamped, queue_size=1)
        self.pub_q = rospy.Publisher("/iiwa_left/command/JointPosition", JointPosition, queue_size=1)

        # define limits 
        self.safe = True
        self.qpos_lower = SAFETY_FACTOR * np.array([-170, -120, -170, -120, -170, -120, -175])*np.pi / 180.0
        self.qpos_upper = -self.qpos_lower
        
        self.effort_lower = SAFETY_FACTOR * np.array([-176, -176, -110, -110, -110, -40, -40])
        self.effort_upper = -self.effort_lower

        self.qvel_lower = SAFETY_FACTOR * np.array([-98, -98, -100, -130, -140, -180, -180])*np.pi / 180.0
        self.qvel_upper = -self.qvel_lower

        # transform listener
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        self.link_names = ["iiwa_left_link_" + str(i) for i in range(1,8)]
        self.link_pos = {}

        
    def cart_callback(self, msg):
        self.frame = msg.poseStamped.header.frame_id
        self.x = msg.poseStamped.pose.position.x
        self.y = msg.poseStamped.pose.position.y
        self.z = msg.poseStamped.pose.position.z

        self.qx = msg.poseStamped.pose.orientation.x
        self.qy = msg.poseStamped.pose.orientation.y
        self.qz = msg.poseStamped.pose.orientation.z
        self.qw = msg.poseStamped.pose.orientation.w

    def set_target(self, xyz, quat):      
        self.target_x = xyz[0]
        self.target_y = xyz[1]
        self.target_z = xyz[2]
        self.target_qx = quat[0]
        self.target_qy = quat[1]
        self.target_qz = quat[2]
        self.target_qw = quat[3]

    def joint_callback(self, msg):
        self.qpos = np.array(msg.position)
        self.effort = np.array(msg.effort)

    def joint_vel_callback(self, msg):
        self.qvel[0] = msg.velocity.a1
        self.qvel[1] = msg.velocity.a2
        self.qvel[2] = msg.velocity.a3
        self.qvel[3] = msg.velocity.a4
        self.qvel[4] = msg.velocity.a5
        self.qvel[5] = msg.velocity.a6
        self.qvel[6] = msg.velocity.a7

    def check_safety(self):
        """ returns True when arm is safe"""

        # check jointspace limits 
        if np.any( self.qpos < self.qpos_lower ) or np.any( self.qpos > self.qpos_upper ):
            self.safe = False
            return False 

        if np.any( self.effort < self.effort_lower ) or np.any( self.effort > self.effort_upper ):
            self.safe = False
            return False 

        if np.any( self.qvel < self.qvel_lower ) or np.any( self.qvel > self.qvel_upper ):
            self.safe = False
            return False 

        # check workspace limits 
        # TODO 

        self.last_safe_state = copy.deepcopy(self.qpos)
        return True 

        
    def publish_target_command(self):
 
        if self.target_x is None:
            return 

        if self.unsafe:
            return 

        message = PoseStamped()
        message.header.frame_id = self.frame

        message.pose.position.x = self.target_x
        message.pose.position.y = self.target_y
        message.pose.position.z = self.target_z

        message.pose.orientation.x = self.target_qx
        message.pose.orientation.y = self.target_qy
        message.pose.orientation.z = self.target_qz
        message.pose.orientation.w = self.target_qw
        self.pub_cart.publish(message)
        return 

    def publish_current_command(self): 
        message = PoseStamped()
        message.header.frame_id = self.frame

        message.pose.position.x = self.x
        message.pose.position.y = self.y
        message.pose.position.z = self.z

        message.pose.orientation.x = self.qx
        message.pose.orientation.y = self.qy
        message.pose.orientation.z = self.qz
        message.pose.orientation.w = self.qw
        self.pub_cart.publish(message)
        return 

    def publish_q(self, q):
        message = JointPosition()
        message.position.a1 = q[0]
        message.position.a2 = q[1]
        message.position.a3 = q[2]
        message.position.a4 = q[3]
        message.position.a5 = q[4]
        message.position.a6 = q[5]
        message.position.a7 = q[6]
        self.pub_q.publish(message)

    def update_link_poses(self):

        for link_name in self.link_names:

            got_transform = False
            while not got_transform:
                try:
                    trans = self.tfBuffer.lookup_transform("iiwa_left_link_1", "iiwa_left_link_0", rospy.Time())
                    self.link_pos[link_name] = trans[0]
                    got_transform = True 
                except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                    print(e)

        print(self.link_pos)
                


if __name__ == '__main__':
    
    rospy.init_node('arm_commander', anonymous=True)    

    armcommand = ArmCommand()

    rospy.Subscriber("/iiwa_left/state/CartesianPose", CartesianPose, armcommand.cart_callback)
    rospy.Subscriber("/iiwa_left/joint_states", JointState, armcommand.joint_callback)
    rospy.Subscriber("/iiwa_left/state/JointVelocity", JointVelocity, armcommand.joint_vel_callback)

    last_safe_state = None
    target_q = np.array([0, 0, 0, 10, 160, 0, 45]) * np.pi / 180. 

    rate = rospy.Rate(CONTROL_FREQ)
    while not rospy.is_shutdown():

        armcommand.update_link_poses()

        # if armcommand.check_safety():
        #     armcommand.publish_q(target_q)
        # else:
        #     if last_safe_state is None:
        #         last_safe_state = copy.deepcopy(armcommand.qpos)

        #     armcommand.publish_q(last_safe_state)
        #     rospy.loginfo(armcommand.qpos)

        rate.sleep()