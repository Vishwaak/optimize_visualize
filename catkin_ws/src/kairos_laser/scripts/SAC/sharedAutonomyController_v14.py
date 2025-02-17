#! /usr/bin/env python

#-------------------------------------------------------------------------------------------------
#   @file       : sharedAutonomyController_v14.py
#   @author     : PARAM D SALUNKHE | UTARI - AIS
#   @comments   : Finetuning k_rep, rho_0, rho_cap
#                 green_arrow normalization - if |red_arrow| > 0 then normFactor = 2 else  = 1
#-------------------------------------------------------------------------------------------------
 
#-------------------------------------------------------------------------------------------------
#   Package Imports
#-------------------------------------------------------------------------------------------------

import rospy
import math
import tf
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
#  Global Variable and Constants Declarations
#-------------------------------------------------------------------------------------------------

# Config constants for obtaining repulsion from all obstacles / closest obstacle (@self.publish_repulsiveResultant)
ALL_OBSTACLES = 0
CLOSEST_OBSTACLE = 1

'''# Normalization Factors (for vector normalization experiments)
NORM_FACTOR_0 = 1       # @self.publish_PotentialFields()
NORM_FACTOR_1 = 1       # @self.publish_repulsiveResultant()
NORM_FACTOR_2 = 1       # @self.publish_finalVelocity_marker()'''

# Normalization Factors (for no normalization)
NORM_FACTOR_0 = 1       # @self.publish_PotentialFields()
NORM_FACTOR_1 = 1       # @self.publish_repulsiveResultant()
NORM_FACTOR_2 = 1       # @self.publish_finalVelocity_marker()

# APF Repulsion scaling parameter (@self.publish_PotentialFields)
# K_REP = 0.111111075     # For normalization experiments
# K_REP = 0.416705        # For no normalization
K_REP = 0.6

#-------------------------------------------------------------------------------------------------
#   Class Definitions
#-------------------------------------------------------------------------------------------------

class SharedAutonomyController:
    #------------------------------------ CLASS CONSTRUCTOR --------------------------------------
    def __init__(self):
        rospy.init_node('shared_autonomy_controller_node', anonymous=True)                       # Initializing a ROS node
        self.pcld2_pub = rospy.Publisher('/transformed_pcld2', PointCloud2, queue_size=10)       # Creating a ROS Publisher for pointCloud of the subscribed and transformed laserScan points
        self.marker_pub = rospy.Publisher('/obstacle_marker', Marker, queue_size=10)             # Creating a ROS Publisher for obstacle boundaries
        self.centroid_pub = rospy.Publisher('/centroid_marker', Marker, queue_size=10)           # Creating a ROS Publisher for obstacle centroids
        self.scan1_pub = rospy.Publisher('/ROI1_laserScan', LaserScan, queue_size=10)            # Creating a ROS Publisher for processed ROI#1 LaserScan messages
        self.scan2_pub = rospy.Publisher('/ROI2_laserScan', LaserScan, queue_size=10)            # Creating a ROS Publisher for processed ROI#2 LaserScan messages
        self.repForce_pub = rospy.Publisher('/rep_marker', MarkerArray, queue_size=10)           # Creating a ROS Publisher for obstacle Repulsive force arrow markers
        self.resForce_pub = rospy.Publisher('/res_marker', Marker, queue_size=10)                # Creating a ROS Publisher for resultant repulsive force arrow marker
        self.refSignal_pub = rospy.Publisher('/refSignal_marker', Marker, queue_size=10)         # Creating a ROS Publisher for input joystick reference signal arrow marker
        self.vfinal_marker_pub = rospy.Publisher('/vfinal_marker', Marker, queue_size=10)        # Creating a ROS Publisher for the final velocity signal arrow marker
        self.vfinal_joy_pub = rospy.Publisher('/robot/joy', Joy, queue_size=10)                  # Creating a ROS Publisher for the final velocity signal joy message
        self.front_minAngle = 0                             # unit: radian
        self.front_maxAngle = 0                             # unit: radian
        self.front_angIncrement = 0                         # unit: radian
        self.front_minRange = 0                             # unit: meters
        self.front_maxRange = 0                             # unit: meters 
        self.front_ranges = []                              # unit: meters
        self.front_FoV = 0                                  # unit: count
        self.front_noOfScans = 0                            # unit: meters
        self.scan_threshold_range = 5                       # unit: meters
        self.marker_threshold_range = 2.0                   # unit: meters
        self.rho_0 = 1.0                                   # unit: meters
        self.rho_cap = 0.75                                  # unit: meters
        self.loop_frequency = 250                           # unit: hertz
        self.marker_lifetime = 1/self.loop_frequency        # uint: seconds
        self.k_rep = K_REP                                  # unit: repulsive field constant     ...use for no normalization
        self.base_r_values = []                             # Polar coordinate - r values of LaserScan points that are transformed from "robot_front_laser_link" to "robot_base_link" 
        self.base_theta_values = []                         # Polar coordinate - theta values of the transformed LaserScan points from "robot_front_laser_link" to "robot_base_link"
        self.roi1_ranges = []                               # LaserScan ranges[] for the ROI#1 of all sub-threshold obstacles captured in one LIDAR scan. Re-initialized in publish_obstacles()
        self.roi2_ranges = []                               # LaserScan ranges[] for the ROI#2 of all sub-threshold obstacles captured in one LIDAR scan. Re-initialized in publish_obstacles()
        self.centroids = []                                 # List of centroids of obstacles detected in a single LIDAR scan
        self.closestPoints = []                             # List of points from each detected obstacles that are the closest to the robot
        self.rep_points = []                                # List of endpoints of repulsive force vectors gennerated by each of the sub-threshold obstacles
        self.rep_resultant = Point(0,0,0)                   # The endpoint of the resultant of the repulsive force vectors
        self.ref_signal = Point(0,0,0)                      # The endpoint of the vector that represents input joystick reference signal
        self.vfinal_signal = Point(0,0,0)                   # The endpoint of the vector that represents the final velocity reference signal to be sent to the robot
        self.deadman_switch = 0                             # To indicate the state of the joystick's deadman switch - 1 => Pressed, 0 => Released
        self.autonomy_switch = 0                            # To indicate the state of the joystick's autonomy switch 
        self.tf_listener = TransformListener()              # To transform the measured obstacle distances from "robot_front_laser_link" to "robot_base_link"
        self.vfinal_joy = Joy()                             # To store the final Joy message to be published by the shared_autonomy_controller_node
        self.rep_from = CLOSEST_OBSTACLE                   # repulsion experienced from ALL_OBSTACLES / CLOSEST_OBSTACLE

    #--------------------------------- CLASS FUNCTION DEFINITIONS --------------------------------
    
    # ROS Callback function for the LaserScan subscriber
    def frontScan_callback(self, scan_msg):                                         
        self.front_minAngle = scan_msg.angle_min
        self.front_maxAngle = scan_msg.angle_max
        self. front_angIncrement = scan_msg.angle_increment
        self.front_FoV = self.front_maxAngle - self.front_minAngle
        self.front_noOfScans = math.degrees(self.front_FoV)/ math.degrees(self.front_angIncrement)
        self.front_minRange = scan_msg.range_min
        self.front_maxRange = scan_msg.range_max
        self.front_ranges = scan_msg.ranges
    
    # ROS Callback function for the Joy subscriber
    def joy_bs_callback(self, joy_bs_msg):
        self.vfinal_joy = joy_bs_msg
        self.ref_signal.x = joy_bs_msg.axes[1]
        self.ref_signal.y = joy_bs_msg.axes[0]
        self.ref_signal.z = 0
        self.deadman_switch = joy_bs_msg.buttons[5]
        
    # For calculation of angular position based on LaserScan ranges array index
    def getAngle(self, range_index) -> float:
        angle = self.front_minAngle + (range_index * self.front_angIncrement)
        return angle
    
    # For obtaining the translation and rotation information for transforming from source_frame to target_frame
    def getTransform(self, target_frame, source_frame):
        try:
            self.tf_listener.waitForTransform(target_frame, source_frame, rospy.Time(0), rospy.Duration(4.0))
            (trans, rot) = self.tf_listener.lookupTransform("robot_base_link", "robot_front_laser_link", rospy.Time(0))
            # print(f"quaternion: {rot}")
            return trans, rot
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logerr("Transform lookup from LiDAR frame to robot base frame not available")
            return None, None

    # ROS Publisher function to publish the transformed points w.r.t. "robot_base_link"
    def publish_transformed_pointCloud(self):
        # Obtaining transfomation from "LiDAR frame" to the "Base frame"
        trans, rot = self.getTransform("robot_base_link", "robot_front_laser_link")
        if trans is None or rot is None:
            print("Frame transformation skipped. \n")
            return
        transformed_points = []                                             # List of points transformed from the "LiDAR frame" to the "Base frame"
        for i, r in enumerate(self.front_ranges):
            if self.front_minRange < r < self.front_maxRange:
                theta = self.getAngle(i)                                    # Converting Polar coordinates of the point to --
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
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "robot_base_link"
        pcld2_msg = point_cloud2.create_cloud_xyz32(header, transformed_points)
        self.pcld2_pub.publish(pcld2_msg)
        '''print(len(self.front_ranges))
        print(len(self.base_r_values))
        print(len(self.base_theta_values))
        print(f"lidar frame min angle = {math.degrees(self.front_minAngle)} degrees")
        print(f"lidar frame max angle = {math.degrees(self.front_maxAngle)} degrees")
        print(f"base frame min angle = {math.degrees(min(self.base_theta_values))} degrees")
        print(f"base frame max angle = {math.degrees(max(self.base_theta_values))} degrees")
        print(f"Lidar scan FoV = {(math.degrees(self.front_maxAngle))-(math.degrees(self.front_minAngle))} degrees")
        print(f"transformed base frame pointcloud FoV = {((math.degrees(max(self.base_theta_values)))-(math.degrees(min(self.base_theta_values))))} degrees")'''

    # For displaying LIDAR specifications
    def displayLaserSpecs(self):
        rospy.loginfo("*** KAIROS+ Front Laser Specifications *** \n")
        rospy.loginfo("Minimum Angle in degrees: %f \n", math.degrees(self.front_minAngle))
        rospy.loginfo("Maximum Angle in degrees: %f \n", math.degrees(self.front_maxAngle))
        rospy.loginfo("Angle increment in degrees: %f \n", math.degrees(self.front_angIncrement))
        rospy.loginfo("FoV in degrees: %f \n", math.degrees(self.front_FoV))
        rospy.loginfo("Number of scans per sweep: %f \n", self.front_noOfScans)
        rospy.loginfo("Min Range scanned in m: %f \n", self.front_minRange)
        rospy.loginfo("Max Range scanned in m %f \n", self.front_maxRange)
        rospy.loginfo("Range array size: %d \n", len(self.front_ranges))
        rospy.loginfo("***   ***   ***\n")
        
    # For finding and storing the point on of an obstacle marker that is closest to the robot
    def record_closestPoint(self, points):
        cPoint = Point()
        min_distance = 5.5
        for point in points:
            distance = math.sqrt(point.x**2 + point.y**2)
            if distance <= min_distance:
                min_distance = distance
                cPoint = point
        self.closestPoints.append(cPoint)
        
    # For computing the centroid of an obstacle marker published in RViz
    def compute_centroid(self, points):
        sum_x = sum(point.x for point in points)
        sum_y = sum(point.y for point in points)
        n = len(points)
        g_x = (1/n) * sum_x                                                 # Centroid x-coordinate
        g_y = (1/n) * sum_y                                                 # Centroid y-coordinate
        #print(f"g_x = {g_x}")
        #print(f"g_y = {g_y}")
        self.centroids.append(Point(g_x, g_y, 0))
        return Point(g_x, g_y, 0)
    
    # ROS Publisher function to publish the centroids of obstacle markers
    def publish_centroids(self):
        centroid_marker = Marker()
        centroid_marker.header.frame_id = "robot_base_link"                 # Populating the necessary fields for the Marker message
        centroid_marker.type = Marker.POINTS
        centroid_marker.scale.x = 0.05
        centroid_marker.scale.y = 0.05
        centroid_marker.color.r = 0.0
        centroid_marker.color.g = 1.0
        centroid_marker.color.b = 0.0
        centroid_marker.color.a = 1.0
        centroid_marker.points = self.centroids
        centroid_marker.pose.orientation.w = 1.0
        centroid_marker.header.stamp = rospy.Time.now()
        centroid_marker.lifetime = rospy.Duration(self.marker_lifetime)
        self.centroid_pub.publish(centroid_marker)
            
    # For computing the convex hull of the visible obstacle
    def compute_convexhull(self, points):
        if len(points) < 3:
            return points
        points_np = np.array([(p.x, p.y) for p in points])
        cvxhull = ConvexHull(points_np)
        cvxhull_points = [Point(p[0], p[1], 0) for p in points_np[cvxhull.vertices]]
        return cvxhull_points

    # For computing the Region of Influence (roi_ranges[]) of the obstacle based on it's convex hull 
    def compute_roi1(self, points, centroid):
        #cvxhull_points = self.compute_convexhull(points)
        for point in points:                                                # iterate over cvxhull_points if computing convex hull
            dir1_x = point.x - centroid.x                                   # Generating direction vector from the centroid g to the obstacle point p (dir2 = p - g)
            dir1_y = point.y - centroid.y
            length1 = math.sqrt(dir1_x**2 + dir1_y**2)                      # Computing magnitude or norm of the direction vector (||dir1||)
            unit1_x = dir1_x / length1                                      # Generating unit vector of dir1 (u1 = dir1 / ||dir1||)
            unit1_y = dir1_y / length1
            roi1_x = point.x + (unit1_x * self.rho_0)                       # Generating a vector of constant magnitude r0 and direction same as u (u.r0) that starts--
            roi1_y = point.y + (unit1_y * self.rho_0)                       # --from the obstacle point. The end point will be the ROI point for the obstacle point.

            roi1_r = math.sqrt(roi1_x**2 + roi1_y**2)                       # Converting coordinates: Cartesian form (x,y) -> Polar form (r,theta)  
            roi1_theta =  math.atan2(roi1_y, roi1_x)
            if (self.front_minAngle <= roi1_theta <= self.front_maxAngle):
                index = int((roi1_theta - self.front_minAngle)/self.front_angIncrement)
                self.roi1_ranges[index] = roi1_r                            # roi1_ranges[] will have the value equal to polar "r" of the ROI point,--
                                                                            # --at index which depends on polar "theta" of the ROI point.

    # For computing the Region of Influence (roi_ranges[]) of the obstacle based on it's convex hull 
    def compute_roi2(self, points, centroid):
        #cvxhull_points = self.compute_convexhull(points)
        for point in points:                                                # iterate over cvxhull_points if computing convex hull
            dir2_x = centroid.x - point.x                                   # Generating direction vector from the obstacle point P to the centroid g (dir2 = g - p)
            dir2_y = centroid.y - point.y                                   
            length2 = math.sqrt(dir2_x**2 + dir2_y**2)                      # Computing magnitude or norm of the direction vector (||dir2||)
            unit2_x = dir2_x / length2                                      # Generating unit vector of dir2 (u2 = dir2 / ||dir2||)
            unit2_y = dir2_y / length2
            roi2_x = point.x + (unit2_x * self.rho_0)                       # Generating a vector of constant magnitude r0 and direction same as u (u.r0) that starts--
            roi2_y = point.y + (unit2_y * self.rho_0)                       # --from the obstacle point. The end point will be the ROI point for the obstacle point.

            roi2_r = math.sqrt(roi2_x**2 + roi2_y**2)                       # Converting coordinates: Cartesian form (x,y) -> Polar form (r,theta)  
            roi2_theta =  math.atan2(roi2_y, roi2_x)
            if (self.front_minAngle <= roi2_theta <= self.front_maxAngle):
                index = int((roi2_theta - self.front_minAngle)/self.front_angIncrement)
                self.roi2_ranges[index] = roi2_r                            # roi2_ranges[] will have the value equal to polar "r" of the ROI point,--
                                                                            # --at index which depends on polar "theta" of the ROI point.
    
    # ROS Publisher function to publish a region of influence (ROI#1) padding over all the thresholded obstacles using a LaserScan message
    def publish_roi1(self): 
        scan1 = LaserScan()                                 
        scan1.header.frame_id = 'robot_base_link'                           # Populating the necessary fields for the LaserScan message
        scan1.angle_min = self.front_minAngle
        scan1.angle_max = self.front_maxAngle
        scan1.angle_increment = self.front_angIncrement
        scan1.range_min = self.front_minRange
        scan1.range_max = self.front_maxRange
        scan1.ranges = self.roi1_ranges                                     # ranges[] = roi1_ranges[] for ROI#1
        scan1.header.stamp = rospy.Time.now()
        self.scan1_pub.publish(scan1)                                       # Publish ROI#1
            
    # ROS Publisher function to publish a region of influence (ROI#2) padding over all the thresholded obstacles using a LaserScan Message
    def publish_roi2(self):
        scan2 = LaserScan()
        scan2.header.frame_id = 'robot_base_link'                           # Populating the necessary fields for the LaserScan message
        scan2.angle_min = self.front_minAngle
        scan2.angle_max = self.front_maxAngle
        scan2.angle_increment = self.front_angIncrement
        scan2.range_min = self.front_minRange
        scan2.range_max = self.front_maxRange
        scan2.ranges = self.roi2_ranges                                     # ranges[] = roi2_ranges[] for ROI#2
        scan2.header.stamp = rospy.Time.now()
        self.scan2_pub.publish(scan2)                                       # Publish ROI#2
        
    # ROS Publsiher function to publish arrows that indicate the repulsive forces exerted on the robot by obstacles
    def publish_potentialFields(self):
        rep_markerArray = MarkerArray()
        for i, cPoint in enumerate(self.closestPoints):
            rho_xy = math.sqrt(cPoint.x**2 + cPoint.y**2)
            rep_unit_x = -cPoint.x / rho_xy                                 # Calculating unit vector for the repulsive force
            rep_unit_y = -cPoint.y / rho_xy
            # normFactor0 = 0.5333333333333334
            normFactor0 = 1

            # formula for x- and y- gradient of the repulsive potential field of the obstacle
            if ((rho_xy <= self.rho_0) and (rho_xy >= self.rho_cap)):
                rep_x = 0 + (rep_unit_x * (self.k_rep / (rho_xy**2)) * ((1 / rho_xy) - (1 / self.rho_0)) * 1 / normFactor0)
                rep_y = 0 + (rep_unit_y * (self.k_rep / (rho_xy**2)) * ((1 / rho_xy) - (1 / self.rho_0)) * 1 / normFactor0)
            elif (rho_xy < self.rho_cap):
                rep_x = 0 + (rep_unit_x * (self.k_rep / (self.rho_cap**2)) * ((1 / self.rho_cap) - (1 / self.rho_0)) * 1 / normFactor0)
                rep_y = 0 + (rep_unit_y * (self.k_rep / (self.rho_cap**2)) * ((1 / self.rho_cap) - (1 / self.rho_0)) * 1 / normFactor0)
            else:
                rep_x = 0                                                  # The robot should not experience any repulsion if it is outside the ROI
                rep_y = 0
            print(f"distance to obstacle = {rho_xy} m")
            print(f"repulsive force x component = {rep_x}")
            print(f"repulsive force y component = {rep_y}")
            print(f"repulsive force magnitude = {math.sqrt((rep_x)**2 + (rep_y)**2)}")
            rep_point = Point(rep_x, rep_y, 0)
            self.rep_points.append(rep_point)

            rep_marker = Marker()
            rep_marker.header.frame_id = "robot_base_link"                  # Populating the necessary fields for the Marker message
            rep_marker.type = Marker.ARROW
            rep_marker.id = i
            rep_marker.ns = 'repulsive_forces'
            rep_marker.scale.x = 0.05
            rep_marker.scale.y = 0.1
            rep_marker.scale.z = 0.15
            rep_marker.color.r = 0.8
            rep_marker.color.g = 0.2
            rep_marker.color.b = 0.8
            rep_marker.color.a = 1.0
            rep_marker.pose.orientation.w = 1.0
            rep_marker.points.append(Point(0,0,0))                           
            rep_marker.points.append(rep_point)
            rep_marker.header.stamp = rospy.Time.now()
            rep_marker.lifetime = rospy.Duration(self.marker_lifetime)
            rep_markerArray.markers.append(rep_marker)                      # Adding the ARROW marker to the MarkerArray
        self.repForce_pub.publish(rep_markerArray)
    
    # For computing the resultant vector of the input list of vectors
    def compute_resultant(self, points):
        resultant = Point(0,0,0)
        for point in points:
            resultant.x += point.x
            resultant.y += point.y
            resultant.z += point.z
        return resultant
    
    # ROS Publisher function to publish an arrow marker that indicates the resultant repulsive force from all the visible obstacles
    def publish_repulsiveResultant(self):
        if self.rep_from == ALL_OBSTACLES:
            rep_vectors = self.rep_points
            self.rep_resultant = self.compute_resultant(rep_vectors)
            
        if self.rep_from == CLOSEST_OBSTACLE:
            if self.rep_points:
                closest_rep_point = self.rep_points[0]
                closest_rep_magnitude = math.sqrt(closest_rep_point.x**2 + closest_rep_point.y**2)
                for rep_point in self.rep_points:
                    rep_magnitude = math.sqrt(rep_point.x**2 + rep_point.y**2)
                    if rep_magnitude > closest_rep_magnitude:
                        closest_rep_point = rep_point
                self.rep_resultant = closest_rep_point
            
        if (len(self.rep_points)==0):
            normFactor1 = 1
        else:
            # normFactor1 = len(rep_vectors)
            # normFactor1 = len(rep_vectors) * 1.066666666666669
            normFactor1 = 1
        self.rep_resultant.x = self.rep_resultant.x / normFactor1
        self.rep_resultant.y = self.rep_resultant.y / normFactor1
        self.rep_resultant.z = self.rep_resultant.z / normFactor1
        print(f"rep resultant magnitude = {math.sqrt(self.rep_resultant.x**2 + self.rep_resultant.y**2)}")
        resultant_marker = Marker()
        resultant_marker.header.frame_id = "robot_base_link"
        resultant_marker.type = Marker.ARROW
        resultant_marker.ns = "repulsive_resultant"
        resultant_marker.scale.x = 0.05
        resultant_marker.scale.y = 0.1
        resultant_marker.scale.z = 0.15
        resultant_marker.color.r = 1.0
        resultant_marker.color.g = 0.0
        resultant_marker.color.b = 0.0
        resultant_marker.color.a = 1.0
        resultant_marker.pose.orientation.w = 1.0
        resultant_marker.points.append(Point(0,0,0))
        resultant_marker.points.append(self.rep_resultant)
        resultant_marker.header.stamp = rospy.Time.now()
        self.resForce_pub.publish(resultant_marker)
        
    # ROS Publisher function to publish an arrow marker that indicates the input joystick reference signal
    def publish_referenceSignal(self):
        reference_marker = Marker()
        reference_marker.header.frame_id = "robot_base_link"
        reference_marker.type = Marker.ARROW
        reference_marker.scale.x = 0.05
        reference_marker.scale.y = 0.1
        reference_marker.scale.z = 0.15
        reference_marker.color.r = 0.0
        reference_marker.color.g = 0.0
        reference_marker.color.b = 1.0
        reference_marker.color.a = 1.0
        reference_marker.pose.orientation.w = 1.0
        reference_marker.points.append(Point(0,0,0))
        reference_marker.points.append(self.ref_signal)
        reference_marker.header.stamp = rospy.Time.now()
        self.refSignal_pub.publish(reference_marker)
        
    # ROS Publisher to publish an arrow marker that indicated the shared autonomy controller output - final robot velocity
    def publish_finalVelocity_marker(self):
        ref_magnitude = math.sqrt(self.ref_signal.x**2 + self.ref_signal.y**2)
        # if((ref_magnitude != 0) and (self.deadman_switch == 1)):
        if(self.deadman_switch == 1):
            self.vfinal_signal = self.compute_resultant([self.ref_signal, self.rep_resultant])
        else:
            self.vfinal_signal = Point(0,0,0)

        if ((self.rep_resultant.x == 0) and (self.rep_resultant.y == 0)):
            normFactor2 = 1
        else:
            normFactor2 = 2
        self.vfinal_signal.x = self.vfinal_signal.x / normFactor2
        self.vfinal_signal.y = self.vfinal_signal.y / normFactor2
        self.vfinal_signal.z = self.vfinal_signal.z / normFactor2
        print(f"vfinal_x = {self.vfinal_signal.x}")
        print(f"vfinal_y = {self.vfinal_signal.y}")
        vfinal_marker = Marker()
        vfinal_marker.header.frame_id = "robot_base_link"
        vfinal_marker.scale.x = 0.05
        vfinal_marker.scale.y = 0.1
        vfinal_marker.scale.z = 0.15
        vfinal_marker.color.r = 0.0
        vfinal_marker.color.g = 1.0
        vfinal_marker.color.b = 0.0
        vfinal_marker.color.a = 1.0
        vfinal_marker.pose.orientation.w = 1.0
        vfinal_marker.points.append(Point(0,0,0))
        vfinal_marker.points.append(self.vfinal_signal)
        vfinal_marker.header.stamp = rospy.Time.now()
        self.vfinal_marker_pub.publish(vfinal_marker)
        #return vfinal_signal
    
    # ROS Publisher function to publish the new /robot/joy message
    def publish_vfinal_joy(self):
        vfinal_joy_axes = list(self.vfinal_joy.axes)                        # vfinal_joy.axes is an immutable tuple. Need to convert it to list format for manipulation
        vfinal_joy_axes[0] = self.vfinal_signal.y                           # Replacing the axes[0] and axes[1] in vfinal_joy with the vfinal_signal values
        vfinal_joy_axes[1] = self.vfinal_signal.x
        self.vfinal_joy.axes = tuple(vfinal_joy_axes)                       # Re-converting the list into a tuple form for vfinal_joy.axes
        self.vfinal_joy_pub.publish(self.vfinal_joy)
        
    # ROS Publisher function to publish Marker line strips on thresholded obstacles (to be coded into a separate .py file)
    def publish_obstacles(self):
        self.roi1_ranges = [5.5] * len(self.base_r_values)                  # This list will hold the laserScan ranges[] for the ROI of all obstacles captured in one LIDAR scan
        self.roi2_ranges = [6.0] * len(self.base_r_values)                  # Re-initialize all elements of roi1_ranges[] with the ceiling value for the next LIDAR scan                                               
        points = []                                                         # This list will hold the set of points that form a sub-threshold obstacle
        markerNumber = 0                                                    
        for j in range(len(self.base_r_values)):                            # :: LOOP START :: For each ranges element in the subscribed LaserScan message -- 
            if self.base_r_values[j] < self.marker_threshold_range:         # -- CONDITION: is range element sub-threshold? 
                r = self.base_r_values[j]                                   #  - TRUE: -- convert element coordinates from Polar to Cartesian
                #theta = self.getAngl(j)                                    #          -- add cartesian point to points[]
                theta = self.base_theta_values[j]
                marker_point = Point()
                marker_point.x = r * np.cos(theta)
                marker_point.y = r * np.sin(theta)
                marker_point.z = 0.0
                points.append(marker_point)
            else:                                                           # - FALSE: --
                #self.roi1_ranges.append(5.5)                               #          -- set all non-subthreshold range values to a ceiling of 5.5m
                #self.roi2_ranges.append(5.5) 
                if points:                                                  #          -- CONDITION: are we on the first range element in the subscribed message?
                    marker = Marker()                                       #           - FALSE: -- populate necessary fields for the Marker message
                    marker.header.frame_id = 'robot_base_link'              #                    -- publish /obstacle_marker for points[]
                    marker.ns = 'thresholded_laserScan'                     #                    -- compute closest point and the centroid
                    marker.id = markerNumber                                #                    -- compute the obstacle ROI points to be added to roi1_ranges[] and roi2_ranges[]
                    # print(f"obstacle #{marker.id} @inner if loop")          #                    -- make points[] empty again for  capturing the next obstacle
                    marker.type = Marker.LINE_STRIP                         #           - TRUE: -- do nothing
                    marker.action = Marker.ADD                              # :: LOOP END ::               
                    marker.scale.x = 0.02
                    marker.color.r = 1.0
                    marker.color.g = 0.0
                    marker.color.b = 0.0
                    marker.color.a = 1.0
                    marker.pose.orientation.x = 0.0
                    marker.pose.orientation.y = 0.0
                    marker.pose.orientation.z = 0.0
                    marker.pose.orientation.w = 1.0
                    marker.points = points
                    marker.header.stamp = rospy.Time.now()
                    marker.lifetime = rospy.Duration(self.marker_lifetime)
                    self.marker_pub.publish(marker)
                    self.record_closestPoint(points)
                    centroid = self.compute_centroid(points)
                    self.compute_roi1(points, centroid)
                    self.compute_roi2(points, centroid)
                    points = []
                    markerNumber += 1
        if points:                                                          # -- CONDITION: is points[] still not empty? => need to publish markers for this points[]
            marker = Marker()                                               #  - TRUE:  -- populate necessary fields for the Marker message
            marker.header.frame_id = 'robot_front_laser_link'               #           -- publish /obstacle_marker for points[]
            marker.ns = 'thresholded_laserScan'                             #           -- compute the closest point and the centroid
            marker.id = markerNumber                                        #           -- compute the obstacle ROI points to be added to roi1_ranges[] and roi2_ranges[]
            # print(f"obstacle #{marker.id} @outer if loop")                  #  - FALSE: -- do nothing
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.scale.x = 0.02
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            marker.pose.orientation.x = 0.0 
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.points = points
            marker.header.stamp = rospy.Time.now()
            marker.lifetime = rospy.Duration(self.marker_lifetime)
            self.marker_pub.publish(marker)
            self.record_closestPoint(points)
            centroid = self.compute_centroid(points)                        
            self.compute_roi1(points, centroid)                             
            self.compute_roi2(points, centroid)
        markerNumber = 0                                                    # Re-initialize obstacle count to 0

#-------------------------------------------------------------------------------------------------
#   Function Definitions
#-------------------------------------------------------------------------------------------------


#-------------------------------------------------------------------------------------------------
#   Main Function
#-------------------------------------------------------------------------------------------------
 
if __name__ == '__main__':
    try:
        sac = SharedAutonomyController()
        rospy.Subscriber('/robot/front_laser/scan_filtered', LaserScan, sac.frontScan_callback)
        rospy.Subscriber('/robot/joy_bs', Joy, sac.joy_bs_callback)
        sac.displayLaserSpecs()
        loopRate = rospy.Rate(sac.loop_frequency)                           # Loop rate frequency (25 Hz)
        while not rospy.is_shutdown():
            #sac.displayLaserSpecs()
            sac.publish_transformed_pointCloud()                            # To publish a pointCloud of the subscribed laserScan points, transformed to the "robot_base_link"
            sac.publish_obstacles()                                         # To publish obstacle markers over a thresholded laserScan data
            sac.publish_centroids()                                         # To publish the centroids of the visible thresholded obstacles
            sac.publish_roi1()                                              # One LIDAR Scan complete. Publish ROI#1 for obstacles captured in this scan
            sac.publish_roi2()                                              # One LIDAR Scan complete. Publish ROI#2 for obstacles captured in this scan
            sac.publish_potentialFields()                                   # Publish ARROW markers representing the replusive force exerted by obstacle potential fields
            sac.publish_repulsiveResultant()                                # Publish ARROW marker representing the resultant repulsive force experienced by the obstacle
            sac.publish_referenceSignal()                                   # Publish ARROW marker representing the user input joystick reference signal
            sac.publish_finalVelocity_marker()                              # Publish ARROW marker representing the final velocity vector for the robot
            sac.publish_vfinal_joy()                                        # Publish the final Joy message under the /robot/joy topic
            #print(f"x_linear (JOY) = {sac.ref_signal.x}")
            #print(f"y_linear (JOY) = {sac.ref_signal.y}")
            #print(f"x_linear (SAC) = {vfinal_signal.x}")
            #print(f"y_linear (SAC) = {vfinal_signal.y}")
            sac.base_r_values = []                                          # Re-initialize the base_r_values[] list to "empty" for the next LIDAR scan
            sac.base_theta_values = []                                      # Re-initilaize the base_theta_values[] list to "empty" for the next LIDAR scan 
            sac.centroids = []                                              # Re-initialize the centroids[] list to "empty" for the next LIDAR scan
            sac.closestPoints = []                                          # Re-initialize the closestPoints[] list to "empty" for the next LIDAR scan
            sac.rep_points = []                                             # Re-initialize the rep_points[] list to "empty" for the next LIDAR scan
            print("***   ***   ***\n")
            loopRate.sleep()                                                # Sleep for the reqd. amount of time to maintain the loop rate frequency
    except rospy.ROSInterruptException:
        rospy.logerr("*** ERROR *** \n")
        pass
