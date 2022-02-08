#include "ros/ros.h"
#include "iwaa_stack"
#include <cstdlib>

import rospy 
from iwaa_stack.msg import CartesianImpedanceControlMode.msg

def cartesianimpedancecontrol(req):
    rospy.wait_for_service("/iiwa/configuration/configureSmartServo")
    try:
        config = rospy.ServiceProxy("/iiwa/configuration/configureSmartServo", ConfigureSmartServo)
        # figure out what to add here
        return 
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
    print("Requesting %s+%s"%(x, y))
    print("%s + %s = %s"%(x, y, add_two_ints_client(x, y)))