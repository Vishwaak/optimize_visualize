#! /usr/bin/env python

#-------------------------------------------------------------------------------------------------
#   @file       : ps5_joy_subscriber.py
#   @author     : PARAM D SALUNKHE | UTARI - AIS
#   @comments   : Need to run $rosrun joy joy_node before running this node
#-------------------------------------------------------------------------------------------------
 
#-------------------------------------------------------------------------------------------------
#   Package Imports
#-------------------------------------------------------------------------------------------------

import rospy
import math
import time
from pydualsense import *
from geometry_msgs.msg import Point
from sensor_msgs.msg import Joy, JoyFeedbackArray, JoyFeedback
from visualization_msgs.msg import Marker


#-------------------------------------------------------------------------------------------------
#   Global Variable Declarations
#-------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------
#   Class Definitions
#-------------------------------------------------------------------------------------------------

class PlayStation5Controller:
    #------------------------------------ CLASS CONSTRUCTOR --------------------------------------
    def __init__(self):
        rospy.init_node('ps5_controller_node', anonymous=True)              # Initializing a ROS node
        self.ps5_pub = rospy.Publisher('/ps5_marker', Marker, queue_size=10)
        self.ps5_feedback_pub = rospy.Publisher('/joy/set_feedback', JoyFeedbackArray, queue_size=10)
        self.loop_rate = 25
        self.ps5_refsignal = Point()
        self.sine_vibration_flag = 0
        self.square_vibration_flag = 0
        self.dualsense = pydualsense()
        
    #--------------------------------- CLASS FUNCTION DEFINITIONS --------------------------------
    # ROS Callback function for the PS5 Subscriber
    def ps5_callback(self, ps5_msg):
        self.ps5_refsignal.x = ps5_msg.axes[1]
        self.ps5_refsignal.y = ps5_msg.axes[0]
        self.sine_vibration_flag = ps5_msg.buttons[1]
        self.square_vibration_flag = ps5_msg.buttons[3]
        
    # ROS Publisher to mark the input reference signal from the PS5 controller
    def publish_ps5marker(self):
        ps5_marker = Marker()
        ps5_marker.header.frame_id = "robot_base_link"
        ps5_marker.type = Marker.ARROW
        ps5_marker.scale.x = 0.05
        ps5_marker.scale.y = 0.1
        ps5_marker.scale.z = 0.15
        ps5_marker.color.r = 0.0
        ps5_marker.color.g = 0.0
        ps5_marker.color.b = 1.0
        ps5_marker.color.a = 1.0
        ps5_marker.pose.orientation.w = 1.0
        ps5_marker.points.append(Point(0,0,0))
        ps5_marker.points.append(self.ps5_refsignal)
        ps5_marker.header.stamp = rospy.Time.now()
        self.ps5_pub.publish(ps5_marker)
        return
    
    # ROS Publisher to publish a feedback signal to the PS5 controller
    def publish_ps5feedback(self):
        while (self.sine_vibration_flag == 1):
            print('sine_vibration_flag = 1. Max intensity vibration')
            self.dualsense.light.setColorI(255, 0, 0)
            self.dualsense.setLeftMotor(0)
            self.dualsense.setRightMotor(255)
            feedback_array = JoyFeedbackArray()
            feedback = JoyFeedback()
            feedback.type = JoyFeedback.TYPE_RUMBLE
            feedback.id = 0
            feedback.intensity = 1.0
            feedback_array.array.append(feedback)
            # self.ps5_feedback_pub.publish(feedback_array)
        while (self.square_vibration_flag == 1):
            print('square_vibration_flag = 1. Max intensity vibration')
            self.dualsense.light.setColorI(0, 255, 0)
            self.dualsense.setLeftMotor(255)
            self.dualsense.setRightMotor(0)
            feedback_array = JoyFeedbackArray()
            feedback = JoyFeedback()
            feedback.type = JoyFeedback.TYPE_RUMBLE
            feedback.id = 0
            feedback.intensity = 1.0
            feedback_array.array.append(feedback)
            # self.ps5_feedback_pub.publish(feedback_array)
        self.dualsense.light.setColorI(0, 0, 255)
        self.dualsense.setLeftMotor(0)
        self.dualsense.setRightMotor(0)
        print('vibration_flags = 0. No vibration feedback')
        return
    
#-------------------------------------------------------------------------------------------------
#   Function Definitions
#-------------------------------------------------------------------------------------------------    

#-------------------------------------------------------------------------------------------------
#   Main Function
#-------------------------------------------------------------------------------------------------
 
if __name__ == '__main__':
    try:
        ps5 = PlayStation5Controller()
        ps5.dualsense.init()
        rospy.Subscriber('/joy', Joy, ps5.ps5_callback)
        loopRate = rospy.Rate(ps5.loop_rate)                                # Loop rate frequency (25 Hz)
        while not rospy.is_shutdown():
            ps5.publish_ps5marker()
            ps5.publish_ps5feedback()
            print("***   ***   ***\n")
            loopRate.sleep()                                                # Sleep for the reqd. amount of time to maintain the loop rate frequency
    except rospy.ROSInterruptException:
        rospy.logerr("*** ERROR *** \n")
        pass
