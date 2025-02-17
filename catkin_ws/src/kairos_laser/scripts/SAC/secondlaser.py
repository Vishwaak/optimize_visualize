#! /usr/bin/env python3

#-------------------------------------------------------------------------------------------------
#   @file       : sharedAutonomyController_v16.py
#   @author     : PARAM D SALUNKHE | T VISHWAAK CHANDRAN | UTARI - AIS
#   @comments   : 
#-------------------------------------------------------------------------------------------------
 
#-------------------------------------------------------------------------------------------------
#   Package Imports
#-------------------------------------------------------------------------------------------------

import rospy
import math
import tf
import time
import numpy as np
import tf.transformations
from tf import TransformListener
from scipy.spatial import ConvexHull
from geometry_msgs.msg import Point
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import LaserScan, PointCloud2
from sensor_msgs.msg import Joy
from sensor_msgs import point_cloud2

#-------------------------------------------------------------------------------------------------
#   Class Definitions
#-------------------------------------------------------------------------------------------------

class LaserTransformer:
    #------------------------------------ CLASS CONSTRUCTOR --------------------------------------
    def __init__(self):
        rospy.init_node('second_laser_analysis_node', anonymous=True)                      
        self.front_pcld2_pub = rospy.Publisher('/front_transformed_pcld2', PointCloud2, queue_size=10) 
        self.rear_pcld2_pub = rospy.Publisher('/rear_transformed_pcld2', PointCloud2, queue_size=10)
        self.merged_pcld2_pub = rospy.Publisher('/merged_transformed_pcld2', PointCloud2, queue_size=10)
        self.tf_listener = TransformListener()
        self.front_minAngle = 0                             # unit: radian
        self.front_maxAngle = 0                             # unit: radian
        self.front_angIncrement = 0                         # unit: radian
        self.front_minRange = 0                             # unit: meters
        self.front_maxRange = 0                             # unit: meters 
        self.front_ranges = []                              # unit: meters
        self.front_FoV = 0                                  # unit: count
        self.front_noOfScans = 0                            # unit: meters
        self.rear_minAngle = 0                              # unit: radian
        self.rear_maxAngle = 0                              # unit: radian
        self.rear_angIncrement = 0                          # unit: radian
        self.rear_minRange = 0                              # unit: meters
        self.rear_maxRange = 0                              # unit: meters 
        self.rear_ranges = []                               # unit: meters
        self.rear_FoV = 0                                   # unit: count
        self.rear_noOfScans = 0                             # unit: meters
        self.loop_frequency = 250                           # unit: hertz
        self.base_r_values = []
        self.base_theta_values = []
        self.merged_pointcould = []

    #--------------------------------- CLASS FUNCTION DEFINITIONS --------------------------------
    
    # ROS Callback function for the Front LIDAR LaserScan subscriber
    def frontScan_callback(self, scan_msg):                                         
        self.front_minAngle = scan_msg.angle_min
        self.front_maxAngle = scan_msg.angle_max
        self.front_angIncrement = scan_msg.angle_increment
        self.front_FoV = self.front_maxAngle - self.front_minAngle
        self.front_noOfScans = math.degrees(self.front_FoV)/ math.degrees(self.front_angIncrement)
        self.front_minRange = scan_msg.range_min
        self.front_maxRange = scan_msg.range_max
        self.front_ranges = scan_msg.ranges
        
    # ROS Callback function for the Rear LIDAR LaserScan subscriber
    def rearScan_callback(self, scan_msg):                                         
        self.rear_minAngle = scan_msg.angle_min #+ math.radians(180)
        self.rear_maxAngle = scan_msg.angle_max #+ math.radians(180)
        self.rear_angIncrement = scan_msg.angle_increment
        self.rear_FoV = self.rear_maxAngle - self.rear_minAngle
        self.rear_noOfScans = math.degrees(self.rear_FoV)/ math.degrees(self.rear_angIncrement)
        self.rear_minRange = scan_msg.range_min
        self.rear_maxRange = scan_msg.range_max
        self.rear_ranges = scan_msg.ranges
    
        # For calculation of angular position based on LaserScan ranges array index
    def getFrontAngle(self, range_index) -> float:
        angle = self.front_minAngle + (range_index * self.front_angIncrement)
        return angle
    
    # For calculation of angular position based on LaserScan ranges array index
    def getRearAngle(self, range_index) -> float:
        angle = self.rear_minAngle + (range_index * self.rear_angIncrement)
        return angle
        
    # For obtaining the translation and rotation information for transforming from source_frame to target_frame
    def getTransform(self, target_frame, source_frame):
        try:
            self.tf_listener.waitForTransform(target_frame, source_frame, rospy.Time(0), rospy.Duration(4.0))
            (trans, rot) = self.tf_listener.lookupTransform(target_frame, source_frame, rospy.Time(0))
            return trans, rot
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logerr("Transform lookup from LiDAR frame to robot base frame not available")
            return None, None

    # ROS Publisher function to publish the Front LIDAR transformed points w.r.t. "robot_base_link"
    def get_front_transformed_pointCloud(self):
        # Obtaining transfomation from "LiDAR frame" to the "Base frame"
        trans, rot = self.getTransform("robot_base_link", "robot_front_laser_link")
        if trans is None or rot is None:
            print("Frame transformation skipped. \n")
            return
        transformed_points = []                                             # List of points transformed from the "LiDAR frame" to the "Base frame"
        for i, r in enumerate(self.front_ranges):
            if self.front_minRange < r < self.front_maxRange:
                theta = self.getFrontAngle(i)                               # Converting Polar coordinates of the point to --
                x_lidar = r * math.cos(theta)                               # -- Cartesian coordinates in the "LiDAR frame"
                y_lidar = r * math.sin(theta)
                z_lidar = 0.0  
                lidar_point = np.array([x_lidar, y_lidar, z_lidar, 1.0])    # Defining lidar_point as a 4x1 vector (homogeneous coordinates)
                Rq = tf.transformations.quaternion_matrix(rot)              # Obtaining 4x4 Quaternion matrix Rq which contains the 3x3 rotation matrix R
                Tr = np.array([                                             # Obtaining 4x4 Homogeneous translation matrix Tr which contains the 3x1 translation vector t                                              
                    [1, 0, 0, trans[0]],
                    [0, 1, 0, trans[1]],
                    [0, 0, 1, trans[2]],
                    [0, 0, 0, 1]
                ])
                Tx = np.dot(Tr, Rq)                                         # Obtaining 4x4 Transformation matrix Tx = Tr * Rq = [[R], [t]; [0], [1]]
                base_point = np.dot(Tx, lidar_point)                        # Obtaining 4x1 vector base_point = H * lidar_point, which is the transformed point
                x_base = base_point[0]                                      # Extracting Cartesian coordinates of the point in the "Base frame"
                y_base = base_point[1]
                z_base = base_point[2]
                transformed_points.append((x_base, y_base, z_base))
                r_base = math.sqrt(x_base**2 + y_base**2)                   # Converting Cartesian coordinates of base_point to Polar form
                theta_base = math.atan2(y_base, x_base)
                self.base_r_values.append(r_base)
                self.base_theta_values.append(theta_base)
        self.merged_pointcould.extend(transformed_points)
        #self.front_pcld2_pub.publish(pcld2_msg)
        
    # ROS Publisher function to publish the Rear LIDAR transformed points w.r.t. "robot_base_link"
    def get_rear_transformed_pointCloud(self):
        # Obtaining transfomation from "LiDAR frame" to the "Base frame"
        trans, rot = self.getTransform("robot_base_link", "robot_rear_laser_link")
        if trans is None or rot is None:
            print("Frame transformation skipped. \n")
            return
        transformed_points = []                                             # List of points transformed from the "LiDAR frame" to the "Base frame"
        for i, r in enumerate(self.rear_ranges):
            if self.rear_minRange < r < self.rear_maxRange:
                theta = self.getRearAngle(i)                                # Converting Polar coordinates of the point to --
                x_lidar = r * math.cos(theta)                               # -- Cartesian coordinates in the "LiDAR frame"
                y_lidar = r * math.sin(theta)
                z_lidar = 0.0  
                lidar_point = np.array([x_lidar, y_lidar, z_lidar, 1.0])    # Defining lidar_point as a 4x1 vector (homogeneous coordinates)
                Rq = tf.transformations.quaternion_matrix(rot)              # Obtaining 4x4 Quaternion matrix Rq which contains the 3x3 rotation matrix R
                Tr = np.array([                                             # Obtaining 4x4 Homogeneous translation matrix Tr which contains the 3x1 translation vector t                                              
                    [1, 0, 0, trans[0]],
                    [0, 1, 0, trans[1]],
                    [0, 0, 1, trans[2]],
                    [0, 0, 0, 1]
                ])
                Tx = np.dot(Tr, Rq)                                         # Obtaining 4x4 Transformation matrix Tx = Tr * Rq = [[R], [t]; [0], [1]]
                base_point = np.dot(Tx, lidar_point)                        # Obtaining 4x1 vector base_point = H * lidar_point, which is the transformed point
                x_base = base_point[0]                                      # Extracting Cartesian coordinates of the point in the "Base frame"
                y_base = base_point[1]
                z_base = base_point[2]
                transformed_points.append((x_base, y_base, z_base))
                r_base = math.sqrt(x_base**2 + y_base**2)                   # Converting Cartesian coordinates of base_point to Polar form
                theta_base = math.atan2(y_base, x_base)
                self.base_r_values.append(r_base)
                self.base_theta_values.append(theta_base)
        self.merged_pointcould.extend(transformed_points)
        #self.rear_pcld2_pub.publish(pcld2_msg)
    
    # Merged pointcloud publisher
    def publish_merged_transformed_pointCloud(self):
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "robot_base_link"
        pcld2_msg = point_cloud2.create_cloud_xyz32(header, self.merged_pointcould)
        self.merged_pcld2_pub.publish(pcld2_msg)

#-------------------------------------------------------------------------------------------------
#   Main Function
#-------------------------------------------------------------------------------------------------
 
if __name__ == '__main__':
    try:
        lt = LaserTransformer()
        rospy.Subscriber('/robot/front_laser/scan_filtered', LaserScan, lt.frontScan_callback)
        rospy.Subscriber('/robot/rear_laser/scan_filtered', LaserScan, lt.rearScan_callback)
        loopRate = rospy.Rate(lt.loop_frequency)             # Loop rate frequency (250 Hz)
        while not rospy.is_shutdown():
            lt.get_front_transformed_pointCloud()       
            lt.get_rear_transformed_pointCloud()
            lt.publish_merged_transformed_pointCloud()
            lt.base_r_values = []
            lt.base_theta_values = []
            lt.merged_pointcould = []
            loopRate.sleep()
    except rospy.ROSInterruptException:
        rospy.logerr("*** ERROR *** \n")
        pass