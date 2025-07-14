#!/bin/bash
# 杀掉 Gazebo & ROS 相关进程（留下 python、本身以及 timeout）
pkill -9 -f gzserver
pkill -9 -f gzclient
pkill -9 -f roslaunch
pkill -9 -f roscore
pkill -9 -f rosmaster
pkill -9 -f robot_state_publisher
pkill -9 -f rviz
# 精准地杀掉测试脚本残留（如果有）
pkill -9 -f test_velodyne_td3.py
echo "[shutdown] Gazebo & ROS 清理完成。"
exit 0
