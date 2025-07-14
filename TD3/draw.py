#!/usr/bin/env python3
import rospy
import numpy as np
import atexit
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator
from matplotlib.animation import FuncAnimation
from nav_msgs.msg import Odometry
from threading import Lock
from collections import deque

class TrajectoryPlotter:
    def __init__(self):
        rospy.init_node("trajectory_plotter", anonymous=True)
        
        # 原始参数保持不变
        self.figure_width = 11.18
        self.figure_height = 6.56
        self.output_filename = "trajectory_final.png"
        
        # 坐标范围
        self.x_start, self.x_end = 0, 10
        self.y_start, self.y_end = -3, 3
        
        # 高性能数据结构
        self.x_data = deque(maxlen=2000)
        self.y_data = deque(maxlen=2000)
        self.data_lock = Lock()
        self.last_update_time = rospy.get_time()
        self.last_x = None
        self.last_y = None
        self.last_time = None
        
   
        
        self.fig, self.ax = plt.subplots(figsize=(self.figure_width, self.figure_height))
        self.line, = self.ax.plot([], [], 'b-', lw=8)
        
        # 图形设置
        self.setup_plot()
        
        # ROS订阅
        self.odom_sub = rospy.Subscriber(
            "/odom", 
            Odometry, 
            self.odom_callback,
            queue_size=1
        )
        
        # 动画设置
        self.ani = FuncAnimation(
            self.fig,
            self.update_plot,
            interval=50,
            blit=True,
            cache_frame_data=False
        )
        
        atexit.register(self.save_figure)
        plt.show()

    def setup_plot(self):
        """保持原始图形设置"""
        self.ax.set_xlim(self.x_start, self.x_end)
        self.ax.set_ylim(self.y_start, self.y_end)
        self.ax.yaxis.set_major_locator(MultipleLocator(0.5))
        self.ax.xaxis.set_major_locator(MultipleLocator(1))
        
        # 网格绘制
        for x in np.arange(self.x_start, self.x_end, 1.0):
            for y in np.arange(self.y_start, self.y_end, 0.5):
                self.ax.add_patch(Rectangle(
                    (x, y), 1.0, 0.5,
                    facecolor='white', edgecolor='lightgray',
                    linewidth=2.8, zorder=0
                ))
        
        # 标签和标题
        self.ax.set_xlabel("X position (m)", fontsize=25)
        self.ax.set_ylabel("Y position (m)", fontsize=25)
        self.ax.set_title("Robot Trajectory", fontsize=40)
        self.ax.tick_params(axis='both', labelsize=20)
        self.ax.yaxis.set_tick_params(labelsize=20)
        
        # 边距
        self.fig.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.9)

    def odom_callback(self, msg):
        """带节流的数据收集和平滑插值"""
        current_time = rospy.get_time()
        if current_time - self.last_update_time < 0.02:  # 50Hz节流
            return
            
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        
        if self.x_start <= x <= self.x_end:
            # with self.data_lock:
            #     if self.last_x is not None and self.last_y is not None and self.last_time is not None:
            #         # 计算与上一个点的距离和时间差
            #         distance = np.sqrt((x - self.last_x)**2 + (y - self.last_y)**2)
            #         dt = current_time - self.last_time
                    
            #         if dt > 0:
            #             # 计算速度
            #             speed = distance / dt
                        
            #             # 根据速度决定插入的点数
            #             num_intermediate = max(1, int(speed * dt * 10))  # 调整系数以控制插入点密度
                        
            #             if num_intermediate > 0:
            #                 for i in range(1, num_intermediate + 1):
            #                     intermediate_x = self.last_x + (x - self.last_x) * i / (num_intermediate + 1)
            #                     intermediate_y = self.last_y + (y - self.last_y) * i / (num_intermediate + 1)
            #                     self.x_data.append(intermediate_x)
            #                     self.y_data.append(intermediate_y)
                
                # 添加当前点
                self.x_data.append(x)
                self.y_data.append(y)
                
                # 更新上一个位置和时间
                self.last_x = x
                self.last_y = y
                self.last_time = current_time
           # self.last_update_time = current_time

    def update_plot(self, frame):
        """安全的显示更新"""
        with self.data_lock:
            if len(self.x_data) > 0:
                self.line.set_data(self.x_data, self.y_data)
        return [self.line]

    def save_figure(self):
        """保存图像"""
        try:
            with self.data_lock:
                self.fig.savefig(
                    self.output_filename,
                    dpi=300,
                    bbox_inches='tight'
                )
            rospy.loginfo(f"轨迹已保存至: {self.output_filename}")
        except Exception as e:
            rospy.logerr(f"保存失败: {str(e)}")

if __name__ == "__main__":
    try:
        TrajectoryPlotter()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass