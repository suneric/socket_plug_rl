#!/usr/bin/env python3
from __future__ import print_function

import numpy as np
from numpy import pi, sqrt, cos, sin, arctan2, array, matrix
from geometry_msgs.msg import Pose
from tf.transformations import quaternion_from_matrix, quaternion_matrix, euler_from_quaternion, quaternion_from_euler
from envs.manipulator import ArmController, KukaArm
from envs.camera import RPIv2, RSD435
from plug_control.msg import WalloutletInfo
import rospy

class Detector:
    def __init__(self):
        self.detection_sub = rospy.Subscriber('/detection/walloutlet', WalloutletInfo, self._cb)
        self.socket_boundary = None
        self.info = False

    def _cb(self,data):
        self.info = data

    def socket_center(self):
        if not self.info.detectable:
            print("no walloutlet detected")
            return None
        else:
            u = self.info.u
            v = self.info.v
            cu = u + int(self.info.w/2)
            cv = v + int(self.info.h/2)
        return (cu,cv)

class PositionEstimator:
    def __init__(self, arm, cam2d, cam3d):
        self.arm = arm
        self.cam2d = cam2d
        self.cam3d = cam3d
        self.configure1 = [0.295,0.0,-0.0425,0,0,-pi/2] # x,y,z,r,p,y from base to cam3d
        self.configure2 = [-0.015,0.0,0.062,0,0,-pi/2] #x,y,z,r,p,y from ee to cam2d
        self.configure3 = [0.0,0.0,0.096,0,0,0] # x,y,z from ee to plug end tip

    def transform(self,translation,rotation):
        def trig(angle):
            return cos(angle),sin(angle)
        Cx,Sx = trig(rotation[0])
        Cy,Sy = trig(rotation[1])
        Cz,Sz = trig(rotation[2])
        dX = translation[0]
        dY = translation[1]
        dZ = translation[2]
        mat_trans = matrix([[1,0,0,dX],
                          [0,1,0,dY],
                          [0,0,1,dZ],
                          [0,0,0,1]])
        mat_rotX = matrix([[1,0,0,0],
                         [0,Cx,-Sx,0],
                         [0,Sx,Cx,0],
                         [0,0,0,1]])
        mat_rotY = matrix([[Cy,0,Sy,0],
                         [0,1,0,0],
                         [-Sy,0,Cy,0],
                         [0,0,0,1]])
        mat_rotZ = matrix([[Cz,-Sz,0,0],
                         [Sz,Cz,0,0],
                         [0,0,1,0],
                         [0,0,0,1]])
        return mat_rotZ*mat_rotY*mat_rotX*mat_trans

    def base2cam2d(self):
        mat0 = self.arm.forward_matrix(joints=self.arm.joint_position(),tool_length=0.0)
        mat1 = self.transform(translation=self.configure2[:3],rotation=self.configure2[3:])
        return mat0*mat1

    def base2cam3d(self):
        mat0 = self.transform(translation=self.configure1[:3],rotation=self.configure1[3:])
        return mat0

    def estimate_uv(self,u,v):
        p0 = np.array([[u],[v],[0],[1]]) # column vector
        print(p0)
        mat0 = self.base2cam2d()
        mat1 = self.base2cam3d()
        p1 = np.linalg.inv(mat1)*mat0*p0
        return (int(p1[0]), int(p1[1]))

    def estimate_pt(self,u,v):
        mat1 = self.base2cam3d()
        p = self.cam3d.point3d(u,v)
        pt = mat1*np.array([[p[0]],[p[1]],[p[2]],[1]])
        return pt[:3]

if __name__ == "__main__":
    rospy.init_node("auto_plug", anonymous=True, log_level=rospy.INFO)
    #rospy.sleep(1) # wait for other node ready, such a gazebo
    detector = Detector()
    arm = KukaArm()
    cam2d = RPIv2()
    cam3d = RSD435()
    controller = ArmController(arm)
    estimator = PositionEstimator(arm,cam2d,cam3d)
    rospy.sleep(1) #
    rate = rospy.Rate(10)
    try:
        while not rospy.is_shutdown():
            info = detector.socket_center()
            if info != None:
                print(info)
                p = estimator.estimate_uv(info[0],info[1])
                print(p)
                controller.move([0,0,0,0,0,0,0],5)
                while controller.status != "reached":
                    controller.move([0,0,0,0,0,0,0])
                pt = estimator.estimate_pt(p[0],p[1])
                print(pt)
            rate.sleep()
    except rospy.ROSInterruptException:
        pass
