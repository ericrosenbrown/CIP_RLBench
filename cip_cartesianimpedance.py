import rospy 
from iwaa_stack.msg import CartesianImpedanceControlMode.msg, ConfigureSmartServo

def cartesianimpedancecontrol(req):
    rospy.wait_for_service("/iiwa/configuration/configureSmartServo")
    try:
        service = rospy.ServiceProxy("/iiwa/configuration/configureSmartServo", ConfigureSmartServo)
        # figure out what to add here
        config = ConfigureSmartServo()
        config.control_mode = 2
        config.cartesian_stiffness = [1, 1, 1, 1, 1, 1]
        config.nullspace_stiffness = 1
        nullspace_damping = .7
        service(config)
        return service
    except rospy.ServiceException as e:
        print("Service call failed: %e"%e)

if __name__ == "__main__":
    if len(sys.argv) == 4:
        control_mode = 2 #CartesianImpedance
        cartesian_stiffness = [1, 1, 1, 1, 1, 1]
        nullspace_stiffness = 1
        nullspace_damping = .7
    else:
        print("Config failed")