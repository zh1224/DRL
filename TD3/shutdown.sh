#!/bin/bash

# 仅终止与 ROS 和 Gazebo 相关的关键进程
killall -9 rosout roslaunch rosmaster gzserver nodelet robot_state_publisher gzclient python python3  rviz   pt_main_thread     static_transform_publisher  timeout             


# 不要使用 pkill -9 -f "python3"，避免杀掉所有 Python3 进程

# 清理 ROS 日志文件
rm -rf ~/.ros/log/*

# 清理系统缓存（需 sudo）
sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null

echo "深度清理完成。"
