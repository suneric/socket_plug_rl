#!/usr/bin/env python3
from __future__ import print_function

import numpy as np
from numpy import pi
import rospy
from envs.manipulator import ArmController, KukaArm
from envs.camera import RPIv2, RSD435
from geometry_msgs.msg import Pose

def change_position(arm, controller, dx, dy, dz):
    cp = arm.forward_kinematics(arm.joint_position())
    newcp = Pose()
    newcp.position.x = cp.position.x+dx
    newcp.position.y = cp.position.y+dy
    newcp.position.z = cp.position.z+dz
    newcp.orientation.x = 0
    newcp.orientation.y = 0.70710678119
    newcp.orientation.z = 0
    newcp.orientation.w = 0.70710678119
    joints = arm.inverse_kinematics(newcp)
    controller.move(joints,0.001)
    current = arm.forward_kinematics(arm.joint_position())
    print("initialize", current)
    return current.position.x - cp.position.x

def keyboard_control(arm, controller):
    dx, dy, dz = 0.0, 0.0, 0.0
    scale = 0.01
    key = input("move (w: +z, x:-z, a:-y, d:+y, f:+x, b:-x, s: stop)\n")
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

    delta = change_position(arm,controller,dx,dy,dz)

if __name__ == "__main__":
    rospy.init_node("auto_plug", anonymous=True, log_level=rospy.INFO)
    #rospy.sleep(1) # wait for other node ready, such a gazebo
    arm = KukaArm()
    controller = ArmController(arm)
    rospy.sleep(1) #
    rate = rospy.Rate(10)
    try:
        while not rospy.is_shutdown():
            if controller.status == "none":
                cp = Pose()
                cp.position.x = 0.8369868
                cp.position.y = -0.029758
                cp.position.z = 0.3
                cp.orientation.x = 0
                cp.orientation.y = 0.70710678119
                cp.orientation.z = 0
                cp.orientation.w = 0.70710678119
                joints = arm.inverse_kinematics(cp)
                print("initialize", cp)
                print("initialize", joints)
                controller.init(joints,0.01)
            else:
                keyboard_control(arm, controller)

            rate.sleep()
    except rospy.ROSInterruptException:
        pass
