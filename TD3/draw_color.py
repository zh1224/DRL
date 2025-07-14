#!/usr/bin/env python3
import rospy
import atexit
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
from matplotlib.ticker import MultipleLocator, MaxNLocator
from matplotlib.patches import Rectangle
from matplotlib import colors
import numpy as np
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion
import math

class TrajectoryPlotter:
    def __init__(self):
        rospy.init_node("trajectory_plotter", anonymous=True)
        self.odom_sub = rospy.Subscriber("/odom_filtered", Odometry, self.odom_callback)

        # Configuration parameters
        self.figure_width = 11.18  # inches
        self.figure_height = 6.36  # inches
        self.save_dpi = 300
        self.output_filename = "no_APF_Liquidity_Backstop.png"
        
        # Initialize data storage
        self.x_data = []
        self.y_data = []
        self.t_data = []
        self.time_origin = None  # Time reference point
        self.last_max_time = 0  # Track maximum time value
        self.poses = []  # Store robot pose information

        # Initialize the figure with configurable aspect ratio
        self.fig, self.ax = plt.subplots(figsize=(self.figure_width, self.figure_height))
        
        # Set axis limits
        self.x_start, self.x_end = -3, 3
        self.y_start, self.y_end = -4, 7
        self.ax.set_xlim(self.x_start, self.x_end)
        self.ax.set_ylim(self.y_start, self.y_end)

        # Draw background grid
        self.draw_grid_background(grid_size_x=1.0, grid_size_y=1)

        # Create trajectory line
        self.lc = LineCollection([], cmap='jet', linewidth=8, 
                               norm=colors.Normalize(0, 1))  # Initial placeholder range
        self.ax.add_collection(self.lc)

        # Configure color bar
        self.cbar = self.fig.colorbar(self.lc, ax=self.ax)
        self.cbar.set_label('Time Progression (s)', fontsize=25)
        
        # Initialize color bar tick parameters
        self.cbar.ax.yaxis.set_major_locator(
            MaxNLocator(nbins=5, steps=[1, 2, 5, 10], prune='lower')
        )
        self.cbar.ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x:.2f}")  # Enforce two decimal places
        )
        self.cbar.ax.tick_params(labelsize=25)

        # Set axis parameters
        self.ax.yaxis.set_major_locator(MultipleLocator(1))
        self.ax.xaxis.set_major_locator(MultipleLocator(1))
        self.ax.set_xlabel("X position (m)", fontsize=25)
        self.ax.set_ylabel("Y position (m)", fontsize=25)
        self.ax.set_title("Robot Trajectory", fontsize=40)
        # Set x-axis tick label font size
        self.ax.xaxis.set_tick_params(labelsize=20)
        # Set y-axis tick label font size
        self.ax.yaxis.set_tick_params(labelsize=20)

        # Initialize animation
        self.ani = animation.FuncAnimation(self.fig, self.update_plot, interval=100)
        atexit.register(self.save_figure_on_exit)
        plt.show()

    def draw_grid_background(self, grid_size_x=1.0, grid_size_y=0.5):
        """Draw grid background"""
        for x in np.arange(self.x_start, self.x_end, grid_size_x):
            for y in np.arange(self.y_start, self.y_end, grid_size_y):
                rect = Rectangle((x, y), grid_size_x, grid_size_y,
                               facecolor='white', edgecolor='lightgray',
                               linewidth=2.8, zorder=0)
                self.ax.add_patch(rect)

    def odom_callback(self, msg):
        """Process odometry data"""
        if self.time_origin is None:
            self.time_origin = msg.header.stamp.to_sec()
            
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        
        if self.x_start <= x <= self.x_end:
            self.x_data.append(x)
            self.y_data.append(y)
            # Store elapsed time in seconds relative to initial time
            self.t_data.append(msg.header.stamp.to_sec() - self.time_origin)
            
            # Store pose information
            pose = msg.pose.pose
            self.poses.append(pose)

    def update_plot(self, frame):
        """Dynamically update the plot"""
        if len(self.x_data) >= 2:
            # Generate line segments
            segments = [
                [(self.x_data[i], self.y_data[i]), 
                 (self.x_data[i+1], self.y_data[i+1])]
                for i in range(len(self.x_data)-1)
            ]
            
            # Calculate intermediate time points
            t_array = np.array(self.t_data)
            segment_times = (t_array[:-1] + t_array[1:]) / 2

            # Update color range dynamically
            current_max_time = max(t_array) if len(t_array) > 0 else 1.0
            if current_max_time > self.last_max_time:
                self.last_max_time = current_max_time
                self.lc.set_norm(colors.Normalize(vmin=0, vmax=self.last_max_time))

            self.lc.set_segments(segments)
            self.lc.set_array(segment_times)

            # Update color bar parameters
            self.cbar.update_normal(self.lc)
            self.cbar.ax.yaxis.set_major_locator(
                MaxNLocator(nbins=5, steps=[1, 2, 5, 10], 
                prune='lower'))  
            self.cbar.ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, _: f"{x:.1f}"))  
            self.cbar.draw_all()  

        return self.lc,

    def save_figure_on_exit(self):
        """Save the figure when exiting"""
        self.fig.savefig(self.output_filename, dpi=self.save_dpi, bbox_inches='tight')
        rospy.loginfo(f"Trajectory saved to: {self.output_filename}")

if __name__ == "__main__":
    try:
        TrajectoryPlotter()
    except rospy.ROSInterruptException:
        pass