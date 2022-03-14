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

class SocketPlugEnv2(GymGazeboEnv):
    def __init__(self,resolution=(300,300),cam_noise=0.0):
        # do not reset simulation as it will reset the time which will
        # make the trajectory control of kuka break as there will be message using old time (#just my guess).
        super(SocketPlugEnv2, self).__init__(
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
        self.touch_forces = np.zeros(3)
        self.touch_position = np.zeros(3)

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
        obs = dict(force = self.touch_forces, position = self.touch_position)
        # print(obs["force"], obs["position"])
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
        self.ready_position = self.plug_adaptor()
        self.success = False
        self.fail = False
        self.ready2plug = False
        self.detectable = False
        self.boxcenter = (0,0)

    def _take_action(self, action): # action = [dy,dz]
        print(">>> actions dy: {:.4f}, dz: {:.4f}".format(action[0], action[1]))
        self.adjust_adaptor(0.0,action[0],action[1])
        self.plug_adaptor(max_force=300,step=0.005)
        self.success = self.contact_sensor.connected()
        self.fail = not self.walloutlet_detachable()
        if not self.success:
            self.pull_adaptor()
            self.plug_adaptor()

    def _is_done(self):
        return self.success or self.fail

    def _compute_reward(self):
        reward = 0
        if self.success:
            reward = 100
        elif self.fail:
            reward = -50
        else:
            reward = -1
        print(">>> step reward: {:.3f}".format(reward))
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
        print("=== arm initialize (", cp.position.x, cp.position.y, cp.position.z, ")")
        return self.arm.forward_kinematics(self.arm.joint_position())

    # step actions for trying to plug adaptor
    # adjust, plug, and pull back
    def adjust_adaptor(self,dx,dy,dz):
        self.ready_position.position.x += dx
        self.ready_position.position.y += dy
        self.ready_position.position.z += dz
        joints = self.arm.inverse_kinematics(self.ready_position)
        self.arm_controller.init(joints)
        cp = self.ready_position
        print("=== arm move to (", cp.position.x, cp.position.y, cp.position.z, ")")

    def plug_adaptor(self, max_force = 5, step = 0.01):
        print("=== arm pluging...")
        cp = Pose()
        cp.position.x = self.ready_position.position.x
        cp.position.y = self.ready_position.position.y
        cp.position.z = self.ready_position.position.z
        cp.orientation = self.endeffector_orientation()
        while self.ft_sensor.data()[2] > -max_force:
            cp.position.x += step
            cp.position.y = self.ready_position.position.y
            cp.position.z = self.ready_position.position.z
            cp.orientation = self.endeffector_orientation()
            joints = self.arm.inverse_kinematics(cp)
            self.arm_controller.move(joints)
            cp = self.arm.forward_kinematics(self.arm.joint_position())
            # print("forces detected", self.ft_sensor.data())
        self.arm_controller.stop()
        self.touch_forces = np.array(self.ft_sensor.data())
        self.touch_position = np.array([cp.position.x, cp.position.y, cp.position.z])
        print(">>> info force", self.touch_forces, "position", self.touch_position)
        self.ready_position.position.x = self.touch_position[0]-0.02
        return self.ready_position

    def pull_adaptor(self):
        cp = self.ready_position
        print("=== arm pull back (", cp.position.x, cp.position.y, cp.position.z, ")")
        joints = self.arm.inverse_kinematics(cp)
        self.arm_controller.init(joints)

    def walloutlet_detachable(self):
        detachable = True
        cp = self.arm.forward_kinematics(self.arm.joint_position())
        if cp.position.y > -0.0339 + 0.04 or cp.position.y < -0.0339 - 0.04:
            detachable = False
        if cp.position.z > 0.3609 + 0.05 or cp.position.z < 0.3609 - 0.05:
            detachable = False
        # print("detachable", detachable)
        return detachable
