import rospy 
from iiwa_msgs.msg import CartesianImpedanceControlMode
from iiwa_msgs.srv import ConfigureControlMode

if __name__ == '__main__':
    
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