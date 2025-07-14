#!/usr/bin/env python3
import rospy
import rosbag
import time
import signal
import sys
import os
from datetime import datetime
import roslib.message  # 用于根据字符串获取消息类型类

class RosbagRecorder:
    def __init__(self, topics=None, output_dir="./", prefix="rosbag"):
        """
        ROS bag录制器
        :param topics: 要录制的话题列表 (默认录制所有话题)
        :param output_dir: 输出目录
        :param prefix: 文件名前缀
        """
        rospy.init_node('rosbag_recorder', anonymous=True)

        self.topics = topics if topics else []
        self.output_dir = output_dir
        self.prefix = prefix

        timestamp = datetime.now().strftime("APF_Liquidity_Backstop")
        self.bag_filename = f"{timestamp}.bag"

        self.bag = None
        self.is_recording = False
        self.subscribers = []

        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def start_recording(self):
        """开始录制"""
        try:
            self.bag = rosbag.Bag(self.bag_filename, 'w')
            self.is_recording = True
            rospy.loginfo(f"Started recording to {self.bag_filename}")

            if not self.topics:
                self.topics = self._get_active_topics()

            rospy.loginfo(f"Recording topics: {self.topics}")

            published_topics = dict(rospy.get_published_topics())
            for topic in self.topics:
                if topic not in published_topics:
                    rospy.logwarn(f"Topic {topic} is not currently published.")
                    continue

                msg_type_str = published_topics[topic]
                msg_class = roslib.message.get_message_class(msg_type_str)

                if msg_class is None:
                    rospy.logwarn(f"Unable to load message class for topic {topic} with type {msg_type_str}")
                    continue

                sub = rospy.Subscriber(topic, msg_class, self.callback, callback_args=topic)
                self.subscribers.append(sub)

            rospy.spin()

        except Exception as e:
            rospy.logerr(f"Recording failed: {str(e)}")
        finally:
            self.stop_recording()

    def callback(self, msg, topic):
        """话题回调函数"""
        if self.is_recording:
            try:
                self.bag.write(topic, msg, rospy.Time.now())
            except Exception as e:
                rospy.logerr(f"Failed to write message: {str(e)}")

    def stop_recording(self):
        """停止录制"""
        if self.bag is not None:
            try:
                self.is_recording = False
                self.bag.close()
                rospy.loginfo(f"Bag file saved to: {self.bag_filename}")
                rospy.loginfo(f"File size: {self.get_file_size_mb():.2f} MB")
            except Exception as e:
                rospy.logerr(f"Error closing bag: {str(e)}")

    def _get_active_topics(self):
        """获取所有活跃话题名"""
        return [topic for topic, _ in rospy.get_published_topics()]

    def get_file_size_mb(self):
        """获取文件大小（MB）"""
        return os.path.getsize(self.bag_filename) / (1024 * 1024)

    def signal_handler(self, sig, frame):
        """处理中断信号"""
        rospy.loginfo("Received shutdown signal, stopping recording...")
        self.stop_recording()
        sys.exit(0)

if __name__ == "__main__":
    recorder = RosbagRecorder(
        topics=[
            '/odom',
        ],
        output_dir="./bags",
        prefix="robot_data"
    )
    recorder.start_recording()