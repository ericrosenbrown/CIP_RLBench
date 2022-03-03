import rospy 
from iiwa_msgs.msg import CartesianImpedanceControlMode, CartesianPose, JointVelocity
from iiwa_msgs.srv import ConfigureControlMode
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState

import numpy as np

CONTROL_FREQ = 10

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
        self.cur_q = np.zeros(7)
        self.cur_qdot = np.zeros(7)
        self.cur_effort = np.zeros(7)

        # cartesian pose pub 
        self.pub = rospy.Publisher("/iiwa_left/command/CartesianPose", PoseStamped)

        # define limits 
        self.qpos_lower = np.array([-170, -120, -170, -120, -170, -120, -175])*np.pi / 180.0
        self.qpos_upper = -self.qpos_lower
        
        self.torque_lower = np.array([-176, -176, -110, -110, -110, -40, -40])
        self.torque_upper = -self.torque_lower

        self.qvel_lower = np.array([-98, -98, -100, -130, -140, -180, -180])*np.pi / 180.0
        self.qvel_upper = -self.qvel_lower
        
    def cart_callback(self, msg):
        self.x = msg.poseStamped.pose.position.x
        self.y = msg.poseStamped.pose.position.y
        self.z = msg.poseStamped.pose.position.z

        self.qx = msg.poseStamped.pose.orientation.x
        self.qy = msg.poseStamped.pose.orientation.y
        self.qz = msg.poseStamped.pose.orientation.z
        self.qw = msg.poseStamped.pose.orientation.w

        if self.target_x is None:
            self.frame = msg.poseStamped.header.frame_id
            self.target_x = self.x 
            self.target_y = self.y 
            self.target_z = self.z
            self.target_qx = self.qx
            self.target_qy = self.qy
            self.target_qz = self.qz
            self.target_qw = self.qw

    def joint_callback(self, msg):
        self.cur_q = np.array(msg.position)
        self.cur_effort = np.array(msg.effort)

    def joint_vel_callback(self, msg):
        self.cur_qdot[0] = msg.velocity.a1
        self.cur_qdot[1] = msg.velocity.a2
        self.cur_qdot[2] = msg.velocity.a3
        self.cur_qdot[3] = msg.velocity.a4
        self.cur_qdot[4] = msg.velocity.a5
        self.cur_qdot[5] = msg.velocity.a6
        self.cur_qdot[6] = msg.velocity.a7
        
    def test_command(self):
 
        if self.target_x is None:
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
        self.pub.publish(message)

        rospy.loginfo('-'*20)
        rospy.loginfo('Got pose %f, %f, %f'  % (self.x, self.y, self.z))
        rospy.loginfo('Target pose %f, %f, %f'  % (self.target_x, self.target_y, self.target_z))
        rospy.loginfo('-'*20)
        return 

if __name__ == '__main__':
    armcommand = ArmCommand()
    rospy.init_node('arm_commander', anonymous=True)
    
    rospy.Subscriber("/iiwa_left/state/CartesianPose", CartesianPose, armcommand.cart_callback)
    rospy.Subscriber("/iiwa_left/joint_states", JointState, armcommand.joint_callback)
    rospy.Subscriber("/iiwa_left/state/JointVelocity", JointVelocity, armcommand.joint_vel_callback)

    rate = rospy.Rate(CONTROL_FREQ)
    while not rospy.is_shutdown():
        armcommand.test_command()
        rate.sleep()