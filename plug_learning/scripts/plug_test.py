#!/usr/bin/env python
from __future__ import print_function

import numpy as np
from numpy import pi
import rospy
from envs.manipulator import ArmController, KukaArm
from envs.camera import RPIv2, RSD435
from geometry_msgs.msg import Pose

def initialize_endeffector():
    init = Pose()
    init.position.x = 0.6
    init.position.y = 0.0
    init.position.z = 0.4
    init.orientation.x = 0
    init.orientation.y = 0.70710678119
    init.orientation.z = 0
    init.orientation.w = 0.70710678119
    return init

def change_position(cp, dx, dy, dz):
    newcp = Pose()
    newcp.position.x = cp.position.x+dx
    newcp.position.y = cp.position.y+dy
    newcp.position.z = cp.position.z+dz
    newcp.orientation.x = 0
    newcp.orientation.y = 0.70710678119
    newcp.orientation.z = 0
    newcp.orientation.w = 0.70710678119
    return newcp

def keyboard_control(arm, controller):
    dx, dy, dz = 0.0, 0.0, 0.0
    scale = 0.005
    key = raw_input("move (w: +z, x:-z, a:-y, d:+y, f:+x, b:-x, s: stop)\n")
    if key == 'w':
        dz = scale
    elif key == 'x':
        dz = -scale
    elif key == 'a':
        dy = -scale
    elif key == 'd':
        dy = scale
    elif key == 'f':
        dx = scale
    elif key == 'b':
        dx = -scale
    elif key == 's':
        controller.stop()
    else:
        print("w: +z, x:-z, a:-y, d:+y, f:+z, b:-x, s: stop\n")

    joints = arm.joint_position()
    cp = arm.forward_kinematics(joints)
    print("current", joints, cp)
    newcp = change_position(cp,dx,dy,dz)
    newjoints = arm.inverse_kinematics(newcp)
    print("next", newjoints, newcp)
    controller.set_goal(newjoints)
    controller.move_to_goal()

if __name__ == "__main__":
    rospy.init_node("auto_plug", anonymous=True, log_level=rospy.INFO)
    #rospy.sleep(1) # wait for other node ready, such a gazebo
    rpiv2 = RPIv2()
    rs435 = RSD435()
    arm = KukaArm()
    controller = ArmController(arm)
    rospy.sleep(1) #
    rate = rospy.Rate(30)
    try:
        while not rospy.is_shutdown():
            rpiv2.show()
            rs435.show()
            if controller.status == "none":
                cp = initialize_endeffector()
                init = arm.inverse_kinematics(cp)
                print("initlize", init, cp)
                controller.set_goal(init)
                controller.move_to_goal()
            else:
                keyboard_control(arm, controller)
            # if controller.status == "set":
            #     controller.move_to_goal()
            #     print("forces", arm.tool_force())
            # if controller.status == "reached":
            #     print("goal reached")
            #     joints = arm.joint_position()
            #     print("current joint",joints)
            #     cp = arm.forward_kinematics(joints)

            rate.sleep()
    except rospy.ROSInterruptException:
        pass
