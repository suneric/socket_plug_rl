#!/usr/bin/env python
from __future__ import print_function

import numpy as np
import rospy
from envs.manipulator import ArmController, KukaArm
from envs.camera import CameraSensor

if __name__ == "__main__":
    rospy.init_node("auto_plug", anonymous=True, log_level=rospy.INFO)
    # rospy.sleep(1) # wait for other node ready, such a gazebo
    camera = CameraSensor()
    arm = KukaArm()
    controller = ArmController(arm)
    rospy.sleep(1) #
    rate = rospy.Rate(30)
    try:
        while not rospy.is_shutdown():
            camera.show()
            controller.move_to_goal([-0.04, 1, 0, -1.2, 0, -0.6, 0], 0)
            rate.sleep()
    except rospy.ROSInterruptException:
        pass
