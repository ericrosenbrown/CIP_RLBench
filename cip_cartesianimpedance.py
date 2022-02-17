import rospy 
from iiwa_msgs.msg import CartesianImpedanceControlMode, CartesianPose
from iiwa_msgs.srv import ConfigureControlMode
from geometry_msgs.msg import PoseStamped

class ArmCommand(object):
    """docstring for ArmCommand"""
    def __init__(self):
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
            self.target_x = self.x 
            self.target_y = self.y 
            self.target_z = self.z + 0.05
            self.target_qx = self.qx
            self.target_qy = self.qy
            self.target_qz = self.qz
            self.target_qw = self.qw

        rospy.loginfo('--')
        rospy.loginfo('Got pose %f, %f, %f'  % (self.x, self.y, self.z))
        rospy.loginfo('Target pose %f, %f, %f'  % (self.target_x, self.target_y, self.target_z))

    def test_command(self):

        if self.target_x is None:
            return 

        message = PoseStamped()

        message.pose.position.x = self.target_x
        message.pose.position.y = self.target_y
        message.pose.position.z = self.target_z

        message.pose.orientation.x = self.target_qx
        message.pose.orientation.y = self.target_qy
        message.pose.orientation.z = self.target_qz
        message.pose.orientation.w = self.target_qw
        self.pub.publish(message)

        
        return 

if __name__ == '__main__':

    armcommand = ArmCommand()
    rospy.init_node('cart_subscriber', anonymous=True)
    rospy.Subscriber("/iiwa_left/state/CartesianPose", CartesianPose, armcommand.callback)

    rospy.wait_for_service("/iiwa_left/configuration/ConfigureControlMode")
    try:
        imp = CartesianImpedanceControlMode()
        imp.cartesian_stiffness.x = 150.0
        imp.cartesian_stiffness.y = 150.0
        imp.cartesian_stiffness.z = 150.0
        imp.cartesian_stiffness.a = 150.0
        imp.cartesian_stiffness.b = 150.0
        imp.cartesian_stiffness.c = 150.0

        imp.cartesian_damping.x = 0.5
        imp.cartesian_damping.y = 0.5
        imp.cartesian_damping.z = 0.5
        imp.cartesian_damping.a = 0.5
        imp.cartesian_damping.b = 0.5
        imp.cartesian_damping.c = 0.5

        imp.nullspace_stiffness = 100.0
        imp.nullspace_damping = 0.7

        service = rospy.ServiceProxy("/iiwa_left/configuration/ConfigureControlMode", ConfigureControlMode)
        result = service(control_mode=2, 
                         cartesian_impedance=imp)

    except rospy.ServiceException as e:
        print("Service call failed: %e"%e)

    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        rospy.loginfo("Running test command")
        armcommand.test_command()
        rate.sleep()