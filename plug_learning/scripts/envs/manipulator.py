#!/usr/bin/env python

from __future__ import absolute_import, division, print_function

import numpy as np
from numpy import pi, sqrt, cos, sin, arctan2, array, matrix
import rospy
from std_msgs.msg import Float64MultiArray, Time, Header, Duration, Float64
from geometry_msgs.msg import WrenchStamped, Pose, Twist
from gazebo_msgs.msg import ContactsState
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from tf.transformations import quaternion_from_matrix, quaternion_matrix, euler_from_quaternion, quaternion_from_euler
from numpy.linalg import norm

class BumpSensor:
    def __init__(self, topic='/bumper_plug'):
        self.topic = topic
        self.contact_sub = rospy.Subscriber(self.topic, ContactsState, self._contact_cb)
        self.touched = False

    def connected(self):
        return self.touched

    def _contact_cb(self, data):
        states = data.states
        if len(states) > 0:
            self.touched = True
        else:
            self.touched = False
        # print(self.touched)

    def check_sensor_ready(self):
        self.touched = False
        rospy.logdebug("Waiting for /bumper_plug to be READY...")
        while not rospy.is_shutdown():
            try:
                data = rospy.wait_for_message("/bumper_plug", ContactsState, timeout=5.0)
                self.touched = len(data.states) > 0
                rospy.logdebug("Current /bumper_plugs READY=>")
            except:
                rospy.logerr("Current /bumper_plug not ready yet, retrying for getting  /bumper_plug")

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
        self.arm_l6E = 0.126 # + 0.0964 # arm length from joint 6-endeffector, plus the tool length (0.05+0.7+0.214)
        self.tr = 0.0 # redundency check if possible
        self.joint_pos = None
        self.joint_sub = rospy.Subscriber('/iiwa/joint_states', JointState, self._joint_cb)

    def _joint_cb(self,data):
        self.joint_pos = data.position

    def joint_position(self):
        return self.joint_pos

    def forward_matrix(self,joints,tool_length=0.0964):
        def trig(angle):
            return cos(angle), sin(angle)
        def Hrrt(ty, tz, l):
            cy, sy = trig(ty)
            cz, sz = trig(tz)
            return matrix([[cy * cz, -sz, sy * cz, 0.0],
                           [cy * sz, cz, sy * sz, 0.0],
                           [-sy, 0.0, cy, l],
                           [0.0, 0.0, 0.0, 1.0]])
        def forward(t):
            H02 = Hrrt(joints[1],joints[0],self.arm_l02)
            H24 = Hrrt(-joints[3],joints[2],self.arm_l24)
            H46 = Hrrt(joints[5],joints[4],self.arm_l46)
            H6E = Hrrt(0.0,joints[6],self.arm_l6E+tool_length)
            H0E = H02 * H24 * H46 * H6E
            return H0E
        return forward(joints)

    def forward_kinematics(self,joints,tool_length=0.0964):
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
        ###
        mat = self.forward_matrix(joints)
        cp = matrix_to_cartesian(mat)
        return cp

    def inverse_kinematics(self,cp,tool_length=0.0964):
        def trig(angle):
            return cos(angle), sin(angle)
        def R(q):
            return matrix(quaternion_matrix(q)[:3,:3])
        def rr(p):
            ty = arctan2(sqrt(p[0,0]**2 + p[1,0]**2), p[2,0])
            tz = arctan2(p[1,0], p[0,0])
            if tz < -pi/2.0:
                ty = -ty
                tz += pi
            elif tz > pi/2.0:
                ty = -ty
                tz -= pi
            return (ty, tz)
        def Rz(tz):
            (cz, sz) = trig(tz)
            return matrix([[ cz, -sz, 0.0],
                           [ sz,  cz, 0.0],
                           [0.0, 0.0, 1.0]])
        def Ryz(ty, tz):
            (cy, sy) = trig(ty)
            (cz, sz) = trig(tz)
            return matrix([[cy * cz, -sz, sy * cz],
                           [cy * sz, cz, sy * sz],
                           [-sy, 0.0, cy]])
        ###
        t = 7*[0.0]
        pE0 = matrix([[cp.position.x],
                      [cp.position.y],
                      [cp.position.z]])
        qE0 = array([cp.orientation.x,
                     cp.orientation.y,
                     cp.orientation.z,
                     cp.orientation.w])
        pE6 = matrix([[0.0], [0.0], [self.arm_l6E+tool_length]])
        p20 = matrix([[0.0], [0.0], [self.arm_l02]])
        RE0 = R(qE0)
        p6E0 = RE0 * pE6
        p60 = pE0 - p6E0
        p260 = p60 - p20
        s = norm(p260)
        if s > self.arm_l24 + self.arm_l46:
            print('invalid pose command')
            return None

        (tys, tzs) = rr(p260)
        tp24z0 = 1/(2.0 * s) * (self.arm_l24**2 - self.arm_l46**2 + s**2)
        tp240 = matrix([[-sqrt(self.arm_l24**2 - tp24z0**2)], [0.0], [tp24z0]])
        p240 = Ryz(tys, tzs) * Rz(self.tr) * tp240
        (t[1], t[0]) = rr(p240)

        R20 = Ryz(t[1], t[0])
        p40 = p20 + p240
        p460 = p60 - p40
        p462 = R20.T * p460
        (t[3], t[2]) = rr(p462)
        t[3] = -t[3]

        R42 = Ryz(-t[3], t[2])
        R40 = R20 * R42
        p6E4 = R40.T * p6E0
        (t[5], t[4]) = rr(p6E4)

        R64 = Ryz(t[5], t[4])
        R60 = R40 * R64
        RE6 = R60.T * RE0
        t[6] = arctan2(RE6[1,0], RE6[0,0])
        return t

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
        self.status = "none"
        self.goal = None
        self.velocity = 0.0

    def reached(self,goal,tolerance=0.02):
        err = np.array(goal)-np.array(self.arm.joint_position())
        # print(err)
        if abs(err[0]) > tolerance:
            return False
        if abs(err[1]) > tolerance:
            return False
        if abs(err[2]) > tolerance:
            return False
        if abs(err[3]) > tolerance:
            return False
        if abs(err[4]) > tolerance:
            return False
        if abs(err[5]) > tolerance:
            return False
        if abs(err[6]) > tolerance:
            return False
        return True

    def rotate(self, angle):
        joints = list(self.arm.joint_position())
        print("current joints",joints)
        # joint 6 rotation
        j6 = joints[6]
        if j6 + angle < pi and j6 + angle > -pi:
            joints[6] = j6+angle
        print("new joints",joints)
        self.move(joints)

    def move(self, goal, duration=0.2):
        if goal == None:
            # print("invalid goal")
            return self.arm.joint_position()

        if self.reached(goal):
            self.status = "reached"
            return goal

        msg = JointTrajectory()
        msg.header = Header()
        msg.header.stamp = rospy.Time.now()
        msg.joint_names=['iiwa_joint_1','iiwa_joint_2','iiwa_joint_3','iiwa_joint_4','iiwa_joint_5','iiwa_joint_6','iiwa_joint_7']
        point = JointTrajectoryPoint()
        point.positions = goal
        point.velocities = 7*[0.0]
        point.time_from_start = rospy.Duration(1.5)
        msg.points.append(point)
        self.trajectory_pub.publish(msg)
        # self.status = "moving"
        rospy.sleep(duration)
        return self.arm.joint_position()

    def init(self, goal, duration=0.2):
        while not self.reached(goal):
            msg = JointTrajectory()
            msg.header = Header()
            msg.header.stamp = rospy.Time.now()
            msg.joint_names=['iiwa_joint_1','iiwa_joint_2','iiwa_joint_3','iiwa_joint_4','iiwa_joint_5','iiwa_joint_6','iiwa_joint_7']
            point = JointTrajectoryPoint()
            point.positions = goal
            point.velocities = 7*[0.0]
            point.time_from_start = rospy.Duration(1.5)
            msg.points.append(point)
            self.trajectory_pub.publish(msg)
            self.status = "initlizing"
            # print("initializing")
            rospy.sleep(duration)
        else:
            self.status = "initialized"
            # print("initialized")

    def stop(self, duration=0.2):
        goal = self.arm.joint_position()
        while not self.reached(goal):
            msg = JointTrajectory()
            msg.header = Header()
            msg.header.stamp = rospy.Time.now()
            msg.joint_names=['iiwa_joint_1','iiwa_joint_2','iiwa_joint_3','iiwa_joint_4','iiwa_joint_5','iiwa_joint_6','iiwa_joint_7']
            point = JointTrajectoryPoint()
            point.positions = goal
            point.velocities = 7*[0.0]
            point.time_from_start = rospy.Duration(1.5)
            msg.points.append(point)
            self.trajectory_pub.publish(msg)
            self.status = "stopping"
            # print("stopping")
            rospy.sleep(duration)
        else:
            self.status = "stop"
            # print("stop")
