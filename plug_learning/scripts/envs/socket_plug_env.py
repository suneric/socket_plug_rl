#!/usr/bin/env python

from __future__ import absolute_import, division, print_function

import numpy as np
from numpy import pi, sqrt, cos, sin, arctan2, array, matrix
import rospy
import os
from .gym_gazebo_env import GymGazeboEnv
from gym.envs.registration import register
from geometry_msgs.msg import Pose
from .camera import RPIv2
from .manipulator import BumpSensor, FTSensor, KukaArm, ArmController
import cv2

class SocketPlugEnv(GymGazeboEnv):
    def __init__(self,resolution=(64,64),cam_noise=0.0):
        super(SocketPlugEnv, self).__init__(
            start_init_physics_parameters=False,
            reset_world_or_sim="WORLD"
        )

        self.action_space = self._action_space()
        self.resolution = resolution
        self.arm = KukaArm()
        self.arm_controller = ArmController(self.arm)
        self.camera = RPIv2(resolution,'/rpi/image',cam_noise)
        self.ft_sensor = FTSensor('/iiwa/state/CartesianWrench')
        self.contact_sensor = BumpSensor('/bumper_plug')

        rospy.logdebug("Start ScoeketPlugEnv INIT...")
        self.gazebo.unpauseSim()
        self._check_all_systems_ready()
        self.gazebo.pauseSim()
        rospy.logdebug("Finished ScoeketPlugEnv INIT...")

        self.success = False
        self.force = 0.0
        self.delta = 0.0

    def _check_all_systems_ready(self):
        """
        Checks that all the sensors, publishers and other simulation systems are
        operational.
        """
        self._check_all_sensors_ready()
        self._check_publisher_connection()

    def _check_all_sensors_ready(self):
        self.ft_sensor.check_sensor_ready()
        self.arm.check_sensor_ready()
        rospy.logdebug("All Sensors READY")

    def _check_publisher_connection(self):
        rospy.logdebug("All Publishers READY")

    def _get_observation(self):
        # visual observation (64x64x1)
        image = self.camera.grey_arr()
        # force sensor information (x,y,z)
        forces = self.ft_sensor.data()
        return (image, forces)

    def _display_images(self):
        img = self.camera.rgb_image
        # img = cv2.resize(img, None, fx=0.5, fy=0.5)
        cv2.imshow('rpiv2',img)
        cv2.waitKey(1)

    # return the robot footprint and door position
    def _post_information(self):
        # self._display_images()
        info = []
        return info

    def _set_init(self):
        self.arm_controller.stop()
        self.init_endeffoctor()
        self.success = False
        self.force = 0.0
        self.delta = 0.0
        self.ft_sensor.reset_filtered()

    def _take_action(self, action_idx):
        action = self.action_space[action_idx]
        # delta change in x
        self.delta = self.change_position(action[0],action[1],action[2])
        self.success = self.contact_sensor.connected()
        self.force = self.max_force()
        if self.success or self.force > 30:
            self.arm_controller.stop()

    def _is_done(self):
        if self.success or self.force > 30:
            return True
        else:
            return False

    def _compute_reward(self):
        reward = 0
        if self.success:
            reward = 100
        elif self.force > 30:
            reward = -10
        else:
            penalty = 0.01 + 0.01*self.force # step and force
            reward = 10000*self.delta-penalty
        print("reward", self.force, self.delta, reward)
        return reward

    def _action_space(self):
        s = 0.01 # 1 mm
        actions = [[s,0,0],[-s,0,0],[0,s,0],[0,-s,0],[0,0,s],[0,0,-s]]
        return actions

    def action_dimension(self):
        dim = len(self.action_space)
        print("action dimension", dim)
        return dim

    def visual_dimension(self):
        res = self.resolution
        dim = (res[0], res[1], 1)
        print("visual dimension", dim)
        return dim

    def max_force(self):
        f = np.max(np.absolute(self.ft_sensor.data()))
        print("max force", f)
        return f

    def filtered_force_record(self):
        return self.ft_sensor.filtered()

    def init_endeffoctor(self):
        cp = Pose()
        cp.position.x = 0.9526252
        cp.position.y = -0.0342420
        cp.position.z = 0.3210818
        cp.orientation.x = 0
        cp.orientation.y = 0.70710678119
        cp.orientation.z = 0
        cp.orientation.w = 0.70710678119
        joints = self.arm.inverse_kinematics(cp)
        print("initialize", cp)
        print("initialize", joints)
        self.arm_controller.init(joints,0.01)

    # change endeffector position in x,y,z with remaining the direction
    def change_position(self, dx, dy, dz, duration=0.2):
        cp = self.arm.forward_kinematics(self.arm.joint_position())
        newcp = Pose()
        newcp.position.x = cp.position.x+dx
        newcp.position.y = cp.position.y+dy
        newcp.position.z = cp.position.z+dz
        newcp.orientation.x = 0
        newcp.orientation.y = 0.70710678119
        newcp.orientation.z = 0
        newcp.orientation.w = 0.70710678119
        joints = self.arm.inverse_kinematics(newcp)
        self.arm_controller.move(joints,0.001)
        rospy.sleep(duration)
        current = self.arm.forward_kinematics(self.arm.joint_position())
        return current.position.x - cp.position.x
