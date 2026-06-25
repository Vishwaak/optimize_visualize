#! /usr/bin/env python3

import rospy
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import LaserScan, PointCloud2
from laser_geometry import laser_geometry
import pcl


class MergeLaser:
    def __init__(self):
        rospy.init_node('merge_laser_node', anonymous=True)
        self.frame_id = 'laser'
        self.sub_front = rospy.Subscriber('/robot/front_laser/scan', LaserScan, self.front_callback)
        self.sub_back = rospy.Subscriber('/robot/back_laser/scan', LaserScan, self.back_callback)
        
        tpcl.

    def point_cloud_laserscan(self, laser_data):
        laser_projector = laser_geometry.LaserProjection()
        cloud = laser_projector.projectLaser(laser_data)
        return cloud
    def concatenate_point_clouds(self, cloud1, cloud2):
        return cloud1 + cloud2
        pcl.PCL.