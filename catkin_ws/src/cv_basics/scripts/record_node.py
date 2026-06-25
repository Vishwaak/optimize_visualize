#! /usr/bin/python3


import rospy
import rosbag
import datetime
import os
import sys
import select
import tty
from sensor_msgs.msg import Image
from kairos_laser.msg import obstacle
import matplotlib.pyplot as plt
import numpy as np

import pathlib
global_path = pathlib.Path(__file__).parent


class ExperimentRecorder:
    def __init__(self):
        self.experiment_data = []

    def record_data(self, timestamp, obstacle_direction, distance):
        self.experiment_data.append({
            'timestamp': timestamp,
            'obstacle_direction': obstacle_direction,
            "distance": distance
        })

    def save_to_file(self, filename, start_time):
        total_time = 0
        if self.experiment_data:
            total_time = datetime.datetime.now() - start_time
        with open(filename, 'a') as file:
            file.write(f"Total Time: {total_time}\n")
            for data in self.experiment_data:
                file.write(f"time: {data['timestamp']}, direction: {data['obstacle_direction']}, distance: {data['distance']}\n")
    
   

    def plot_collisions(self):
        timestamps = [data['timestamp'] for data in self.experiment_data]
        collision_counts = list(range(1, len(timestamps) + 1))

        plt.figure(figsize=(10, 5))
        plt.plot(timestamps, collision_counts, marker='o', linestyle='-', color='b')
        plt.xlabel('Timestamp')
        plt.ylabel('Number of Collisions')
        plt.title('Number of Collisions Over Time')
        plt.grid(True)
        plt.show()


class RosbagRecorder:
    def __init__(self, start_time):
        self.start_time = start_time
        rospy.loginfo("Starting rosbag recorder node.")
        rospy.init_node('rosbag_recorder', anonymous=True)
        self.min_distance = np.float32(0.90)
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        record_dir = global_path / 'experimental_records'
        if not os.path.exists(record_dir):
            rospy.loginfo(f"Creating directory {record_dir}")
            os.makedirs(record_dir)
        self.bag = rosbag.Bag(record_dir / f'recorded_data_{self.timestamp}.bag', 'w')
        self.sub1 = rospy.Subscriber('/robot/front_rgbd_camera/rgb/image_raw', Image, self.callback)
        self.exp_recorder = ExperimentRecorder()
        self.obstacle_info = rospy.Subscriber('/obstacle_info', obstacle, self.callback_obstacle) 
        rospy.on_shutdown(self.shutdown_hook)

    def callback(self, data):
        self.bag.write('/robot/front_rgbd_camera/rgb/image_raw', data)
    
    def callback_obstacle(self, data):
        self.bag.write('/obstacle_info', data)
        if float(data.distance.data) <= self.min_distance:
            rospy.loginfo("Detected collision")
            current_time = rospy.get_time()
            if not hasattr(self, 'last_recorded_time') or (current_time - self.last_recorded_time) >= 30:
                self.exp_recorder.record_data(current_time, data.direction, data.distance.data)
                self.last_recorded_time = current_time

    def shutdown_hook(self):
        rospy.loginfo("Shutting down and closing bag file.")
        record_dir = global_path / 'experimental_records'
        self.exp_recorder.save_to_file(record_dir / f'experiment_data_{self.timestamp}.csv', self.start_time)
        self.exp_recorder.plot_collisions()
        self.bag.close()
        rospy.loginfo("Bag file closed.")

def get_key():
    tty.setraw(sys.stdin.fileno())
    select.select([sys.stdin], [], [], 0)
    key = sys.stdin.read(1)
  
    return key

if __name__ == '__main__':
    start_time = datetime.datetime.now()
    try:
        recorder = RosbagRecorder(start_time)
        while not rospy.is_shutdown():
            if select.select([sys.stdin], [], [], 0)[0]:
                key = get_key()
                if key == 'c':
                    rospy.signal_shutdown("User requested shutdown.")
                    recorder.shutdown_hook()
            rospy.sleep(0.1)
    except rospy.ROSInterruptException:
        pass