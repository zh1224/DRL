#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy, copy, math
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion

IDEAL_DT   = 0.1            # 期望的 Odom 步长 (s)
LPF_TAU    = 0.3            # 低通滤波时间常数 (s)
ANTENNA_DX = 0.11           # 天线在车体前向 +X 方向偏移 (m)

class LowPassFilter:
    """一阶指数低通；α 可以动态传进来"""
    def __init__(self):
        self.x = None
        self.y = None

    def update(self, x, y, alpha):
        if self.x is None:
            self.x, self.y = x, y
        else:
            self.x = alpha * x + (1 - alpha) * self.x
            self.y = alpha * y + (1 - alpha) * self.y
        return self.x, self.y

class OdomReTimestamp:
    def __init__(self):
        rospy.init_node("odom_restamp_filter")

        self.filter   = LowPassFilter()
        self.sub      = rospy.Subscriber("/odom", Odometry, self.cb, queue_size=50)
        self.pub      = rospy.Publisher("/odom_filtered", Odometry, queue_size=50)

        self.t0       = None          # 第一帧时间
        self.frame_id = 0             # 计数器

    # ----------------------- callback -----------------------
    def cb(self, msg):
        # 1. 生成“理想”时间戳
        if self.t0 is None:
            self.t0 = msg.header.stamp.to_sec()
        ideal_stamp = self.t0 + self.frame_id * IDEAL_DT
        self.frame_id += 1

        # 2. 提取 & 位置修正（把天线坐标转到车辆几何中心）
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        yaw = euler_from_quaternion([msg.pose.pose.orientation.x,
                                     msg.pose.pose.orientation.y,
                                     msg.pose.pose.orientation.z,
                                     msg.pose.pose.orientation.w])[2]

        center_x = x - ANTENNA_DX * math.sin(yaw)
        center_y = y + ANTENNA_DX * math.cos(yaw)

        # 3. 动态计算 α 做低通
        alpha = IDEAL_DT / (LPF_TAU + IDEAL_DT)
        fx, fy = self.filter.update(center_x, center_y, alpha)

        # 4. 组装并发布
        out = copy.deepcopy(msg)
        out.header.stamp = rospy.Time.from_sec(ideal_stamp)
        out.pose.pose.position.x = fx     # 平滑后的
        out.pose.pose.position.y = fy
        self.pub.publish(out)

# -----------------------------------------------------------
if __name__ == "__main__":
    OdomReTimestamp()
    rospy.spin()
