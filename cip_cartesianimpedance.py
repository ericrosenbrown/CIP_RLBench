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
        
    def callback(self, msg):
        self.x = msg.poseStamped.pose.position.x
        self.y = msg.poseStamped.pose.position.y
        self.z = msg.poseStamped.pose.position.z

        self.qx = msg.poseStamped.pose.orientation.x
        self.qy = msg.poseStamped.pose.orientation.y
        self.qz = msg.poseStamped.pose.orientation.z
        self.qw = msg.poseStamped.pose.orientation.w

    def test_command(self):
        self.x += 0.1
        self.y += 0.1
        self.z += 0.1

        self.qx += 0.1
        self.qy += 0.1
        self.qz += 0.1
        self.qw += 0.1

if __name__ == '__main__':

    armcommand = ArmCommand()
    rospy.init_node('cart_subscriber', anonymous=True)
    rospy.Subscriber("/iiwa_right/state/CartesianPose", CartesianPose, armcommand.callback)

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

    pub = rospy.Publisher("test_command", CartesianPose)
    while not rospy.is_shutdown():
        message = armcommand.test_command
        rospy.loginfo(message)
        pub.publish(message)