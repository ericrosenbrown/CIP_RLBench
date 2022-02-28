import rospy 
from iiwa_msgs.msg import CartesianImpedanceControlMode, CartesianPose
from iiwa_msgs.srv import ConfigureControlMode
from geometry_msgs.msg import PoseStamped

class ArmCommand(object):
    """docstring for ArmCommand"""
    def __init__(self):
        
        self.frame = None

        self.x = None
        self.y = None 
        self.z = None
        self.qx = None 
        self.qy = None
        self.qz = None
        self.qw = None 

        self.target_x = None
        self.target_y = None
        self.target_z = None
        self.target_qx = None
        self.target_qy = None
        self.target_qz = None
        self.target_qw = None

        self.pub = rospy.Publisher("/iiwa_left/command/CartesianPose", PoseStamped)
        
    def callback(self, msg):
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
            self.target_z = self.z + 0.05
            self.target_qx = self.qx
            self.target_qy = self.qy
            self.target_qz = self.qz
            self.target_qw = self.qw

        
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
    rospy.Subscriber("/iiwa_left/state/CartesianPose", CartesianPose, armcommand.callback)
    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        armcommand.test_command()
        rate.sleep()