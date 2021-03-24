#!/usr/bin/env python

from __future__ import absolute_import, division, print_function

import numpy as np
from numpy import pi, sqrt, cos, sin, arctan2, array, matrix
import rospy
from std_msgs.msg import Float64MultiArray, Time, Header, Duration, Float64
from geometry_msgs.msg import WrenchStamped
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

"""
FTSensor for the kuka iiwa joint 7 (end effector)
"""
class FTSensor:
    def __init__(self, topic='/iiwa/state/CartesianWrench'):
        self.topic=topic
        self.force_sub = rospy.Subscriber(self.topic, WrenchStamped, self._force_cb)
        self.record = []
        self.number_of_points = 8
        self.filtered_record = []
        self.step_record = []

    def _force_cb(self,data):
        force = data.wrench.force
        if len(self.record) <= self.number_of_points:
            self.record.append([force.x,force.y,force.z])
        else:
            self.record.pop(0)
            self.record.append([force.x,force.y,force.z])
            self.filtered_record.append(self.data())
            self.step_record.append(self.data())

    def _moving_average(self):
        force_array = np.array(self.record)
        return np.mean(force_array,axis=0)

    # get sensored force data in x,y,z direction
    def data(self):
        return self._moving_average()

    # get force record of entire trajectory
    def reset_filtered(self):
        self.filtered_record = []

    # get force record of a step range
    def reset_step(self):
        self.step_record = []

    def step(self):
        return self.step_record

    def filtered(self):
        return self.filtered_record

    def check_sensor_ready(self):
        self.force_data = None
        while self.force_data is None and not rospy.is_shutdown():
            try:
                data = rospy.wait_for_message(self.topic, WrenchStamped, timeout=5.0)
                self.force_data = data.wrench.force
                rospy.logdebug("Current force sensor READY=>")
            except:
                rospy.logerr("Current force sensor not ready yet, retrying for getting force info")


"""
KUKA IIWA robotic arm
"""
class KukaArm:
    def __init__(self):
        self.arm_l02 = 0.36 # arm length from joint 0-2
        self.arm_l24 = 0.42 # arm length from joint 2-4
        self.arm_l46 = 0.4 # arm length from joint 4-6
        self.arm_l6E = 0.126 # arm length from joint 6-endeffector
        self.tool_length = 0.07 # the length of the plug
        self.wrench = FTSensor(topic='/iiwa/state/CartesianWrench')
        self.joint_pos = None
        self.joint_sub = rospy.Subscriber('/iiwa/joint_states', JointState, self._joint_cb)

    def _joint_cb(self,data):
        self.joint_pos = data.position

    def joint_position(self):
        return self.joint_pos

    def tool_position(self):
        def trig(angle):
            return cos(angle), sin(angle)
        def Hrrt(ty, tz, l):
            cy, sy = trig(ty)
            cz, sz = trig(tz)
            return matrix([[cy * cz, -sz, sy * cz, 0.0],
                           [cy * sz, cz, sy * sz, 0.0],
                           [-sy, 0.0, cy, l],
                           [0.0, 0.0, 0.0, 1.0]])
        def armbase2tool(joints):
            H02 = Hrrt(joints[1],joints[0],self.arm_l02)
            H24 = Hrrt(-joints[3],joints[2],self.arm_l24)
            H46 = Hrrt(joints[5],joints[4],self.arm_l46)
            H6E = Hrrt(0.0,joints[6],self.arm_l6E+self.tool_length)
            H0E = H02 * H24 * H46 * H6E
            return H0E
        def matrix_to_cartesian(mat):
            cp = Pose()
            cp.position.x = mat[0,3]
            cp.position.y = mat[1,3]
            cp.position.z = mat[2,3]
            q = quaternion_from_matrix(mat)
            cp.orientation.x = q[0]
            cp.orientation.y = q[1]
            cp.orientation.z = q[2]
            cp.orientation.w = q[3]
        return cp
        # forward kinematics
        mat = armbase2tool(self.joint_pos)
        return matrix_to_cartesian(mat)

    def tool_force(self):
        return self.wrench.data()

    def check_sensor_ready(self):
        self.joint_pos = None
        rospy.logdebug("Waiting for /iiwa/joint_states to be READY...")
        while self.joint_pos is None and not rospy.is_shutdown():
            try:
                data = rospy.wait_for_message("/iiwa/joint_states", JointState, timeout=5.0)
                self.joint_pos = data.position
                rospy.logdebug("Current /iiwa/joint_states READY=>")
            except:
                rospy.logerr("Current /iiwa/joint_states not ready yet, retrying for getting  /iiwa/joint_states")

"""
Arm Controller
"""
class ArmController:
    def __init__(self, arm):
        self.arm = arm
        self.trajectory_pub = rospy.Publisher('iiwa/PositionJointInterface_trajectory_controller/command', JointTrajectory, queue_size=1)

    def reached(self,goal,tolerance=0.001):
        err = np.array(goal)-np.array(self.arm.joint_position())
        if abs(err[0]) > tolerance:
            print("error", err)
            return False
        if abs(err[1]) > tolerance:
            print("error", err)
            return False
        if abs(err[2]) > tolerance:
            print("error", err)
            return False
        if abs(err[3]) > tolerance:
            print("error", err)
            return False
        if abs(err[4]) > tolerance:
            print("error", err)
            return False
        if abs(err[5]) > tolerance:
            print("error", err)
            return False
        if abs(err[6]) > tolerance:
            print("error", err)
            return False
        print("goal reached")
        return True

    def move_to_goal(self, goal, speed=0.1, tolerance = 0.001):
        if self.reached(goal,tolerance):
            return
            
        msg = JointTrajectory()
        msg.header = Header()
        msg.header.stamp = rospy.Time.now()
        msg.joint_names=['iiwa_joint_1','iiwa_joint_2','iiwa_joint_3','iiwa_joint_4','iiwa_joint_5','iiwa_joint_6','iiwa_joint_7']
        point = JointTrajectoryPoint()
        point.positions = goal
        point.velocities = [speed]*len(goal)
        point.time_from_start = rospy.Duration(2)
        msg.points.append(point)
        self.trajectory_pub.publish(msg)
