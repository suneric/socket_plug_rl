#!/usr/bin/env python

from __future__ import absolute_import, division, print_function

import numpy as np
from numpy import pi, sqrt, cos, sin, arctan2, array, matrix
import rospy
import os
from .gym_gazebo_env import GymGazeboEnv
from gym.envs.registration import register
from geometry_msgs.msg import Pose, Quaternion
from .camera import RPIv2
from .manipulator import BumpSensor, FTSensor, KukaArm, ArmController
from plug_control.msg import WalloutletInfo
import cv2

class SocketPlugEnv(GymGazeboEnv):
    def __init__(self,resolution=(300,300),cam_noise=0.0):
        # do not reset simulation as it will reset the time which will
        # make the trajectory control of kuka break as there will be message using old time (#just my guess).
        super(SocketPlugEnv, self).__init__(
            start_init_physics_parameters=False,
            reset_world_or_sim="NO_RESET_SIM"
        )
        self.camera = RPIv2(resolution,'/rpi/image',cam_noise)
        self.ft_sensor = FTSensor('/iiwa/state/CartesianWrench')
        self.contact_sensor = BumpSensor('/bumper_plug')
        self.detector = rospy.Subscriber('/detection/walloutlet', WalloutletInfo, self.detectcb)
        self.arm = KukaArm()
        self.arm_controller = ArmController(self.arm)

        rospy.logdebug("Start ScoeketPlugEnv INIT...")
        self.gazebo.unpauseSim()
        self._check_all_systems_ready()
        self.gazebo.pauseSim()
        rospy.logdebug("Finished ScoeketPlugEnv INIT...")

        # the position of endeffector where is ready to plug
        self.ready_position = None
        self.success = False
        self.fail = False
        self.ready2plug = False
        self.detectable = False
        self.boxcenter = (0,0)
        self.dist2goal = 0

    def detectcb(self, data):
        self.detectable = data.detectable
        self.boxcenter = (int(data.u+0.5*data.w), int(data.v+0.5*data.h))

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
        obs = dict(image = self.camera.grey_arr(),
                force = np.array(self.ft_sensor.data()),
                joint = np.array(self.arm.joint_position()))
        # print(obs['force'], obs['joint'])
        return obs

    def _post_information(self):
        info = []
        return info

    def action_dimension(self):
        return 2

    def state_dimension(self):
        return (self.camera.resolution[0], self.camera.resolution[1], 1)

    ###########################################################
    def _set_init(self):
        self.ready_position = self.init_endeffoctor()
        self.ft_sensor.reset_filtered()
        self.success = False
        self.fail = False
        self.ready2plug = False
        self.detectable = False
        self.boxcenter = (0,0)

    def _take_action(self, action): # action = [dy,dz]
        self.ready_position = self.adjust_adaptor(action)
        self.dist2goal = self.distance2target()
        if self.dist2goal < 0.005:
            self.ready2plug = True
            self.success = self.plug_adaptor()
            self.pull_adaptor()
        else:
            self.ready2plug = False
            self.success = False

        self.fail = not self.walloutlet_detachable()

    def _is_done(self):
        return self.success or self.fail

    def _compute_reward(self):
        reward = 0
        if self.success:
            reward = 100
        elif self.fail:
            reward = -50
        else:
            if self.ready2plug:
                reward = 50
            else:
                reward = -1
        print("step reward: {:.3f}".format(reward))
        return reward

    ###########################################################
    ## force sensor
    def max_force(self):
        f = np.max(np.absolute(self.ft_sensor.data()))
        return f

    def filtered_force_record(self):
        return self.ft_sensor.filtered()

    def distance2target(self):
        cp = self.arm.forward_kinematics(self.arm.joint_position())
        y = cp.position.y
        z = cp.position.z
        dist = sqrt((y+0.0339)**2 + (z-0.3609)**2)
        return dist

    ####
    # fix the orientation of end-effector, prependicular to the wall
    def endeffector_orientation(self):
        orientation = Quaternion()
        orientation.x = 0
        orientation.y = 0.70710678119
        orientation.z = 0
        orientation.w = 0.70710678119
        return orientation

    # initial end-effector to a position where is 20 cm to the wall outlet
    def random_init_position(self):
        ref = np.random.uniform(size=3)
        cp = Pose()
        cp.position.x = 0.85 + 0.1*(ref[0]-0.5)
        cp.position.y = -0.0339 + 0.05*(ref[1]-0.5)
        cp.position.z = 0.3609 + 0.05*(ref[2]-0.5)
        cp.orientation = self.endeffector_orientation()
        return cp

    def init_endeffoctor(self):
        # self.arm_controller.init([0,0,0,0,0,0,0])
        cp = self.random_init_position()
        joints = self.arm.inverse_kinematics(cp)
        self.arm_controller.init(joints)
        print("initialize (", cp.position.x, cp.position.y, cp.position.z, ")")
        return self.arm.forward_kinematics(self.arm.joint_position())

    # step actions for trying to plug adaptor
    # adjust, plug, and pull back
    def adjust_adaptor(self, action):
        cp = Pose()
        cp.position.x = self.ready_position.position.x + 0.005
        cp.position.y = self.ready_position.position.y + action[0]
        cp.position.z = self.ready_position.position.z + action[1]
        cp.orientation = self.endeffector_orientation()
        joints = self.arm.inverse_kinematics(cp)
        self.arm_controller.init(joints)
        print("adjust (y:", action[0], " z:", action[1], ")")
        return self.arm.forward_kinematics(self.arm.joint_position())

    def plug_adaptor(self, max_force = 100):
        print("pluging...")
        cp = Pose()
        cp.position.x = self.ready_position.position.x
        cp.position.y = self.ready_position.position.y
        cp.position.z = self.ready_position.position.z
        cp.orientation = self.endeffector_orientation()
        while not self.contact_sensor.connected() and self.max_force() < max_force:
            cp.position.x += 0.01
            cp.position.y = self.ready_position.position.y
            cp.position.z = self.ready_position.position.z
            cp.orientation = self.endeffector_orientation()
            joints = self.arm.inverse_kinematics(cp)
            self.arm_controller.move(joints)
            cp = self.arm.forward_kinematics(self.arm.joint_position())
        print("max force detected", self.max_force())
        self.arm_controller.stop()
        return self.contact_sensor.connected()

    def pull_adaptor(self):
        cp = self.ready_position
        print("pull back (", cp.position.x, cp.position.y, cp.position.z, ")")
        joints = self.arm.inverse_kinematics(cp)
        self.arm_controller.init(joints)

    def walloutlet_detachable(self):
        cp = self.arm.forward_kinematics(self.arm.joint_position())
        if cp.position.y > -0.0339 + 0.03 or cp.position.y < -0.0339 - 0.03:
            return False
        if cp.position.z > 0.3609 + 0.03 or cp.position.z < 0.3609 - 0.03:
            return False
        return True

        # (cu, cv) = self.boxcenter
        # print("walloutlet is detectable:",self.detectable, cu, cv)
        # if self.detectable and cu > 100 and cu < 500 and cv > 100 and cv < 600:
        #     return True
        # else:
        #     return False
