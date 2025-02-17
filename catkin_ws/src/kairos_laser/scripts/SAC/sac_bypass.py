#! /usr/bin/env python3

#-------------------------------------------------------------------------------------------------
#   @file       : sac_bypass.py
#   @author     : PARAM D SALUNKHE | UTARI - AIS
#   @comments   : Need to run this node in order to teleoperate the KAIROS without SAC node
#-------------------------------------------------------------------------------------------------
 
#-------------------------------------------------------------------------------------------------
#   Package Imports
#-------------------------------------------------------------------------------------------------

import rospy
from sensor_msgs.msg import Joy

#-------------------------------------------------------------------------------------------------
#   Global Variable Declarations
#-------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------
#   Class Definitions
#-------------------------------------------------------------------------------------------------
class SACBypass:
    
    #------------------------------------ CLASS CONSTRUCTOR --------------------------------------
    def __init__(self):
        rospy.init_node('sac_bypass_node', anonymous=True)
        self.loop_frequency = 250
        self.joy_pub = rospy.Publisher('/robot/joy', Joy, queue_size=10)
        self.joy = Joy()
        
    #--------------------------------- CLASS FUNCTION DEFINITIONS --------------------------------
    
    # ROS Callback function for the /joy_bs subscriber
    def joy_callback(self, joy_bs_msg):
        self.joy = joy_bs_msg
        
    # ROS Publisher to forward /joy_bs to /joy
    def publish_joy(self):
        self.joy_pub.publish(self.joy)
        
#-------------------------------------------------------------------------------------------------
#   Function Definitions
#-------------------------------------------------------------------------------------------------    

#-------------------------------------------------------------------------------------------------
#   Main Function
#-------------------------------------------------------------------------------------------------
 
if __name__ == '__main__':
    try:
        sbypass = SACBypass()
        rospy.Subscriber('/robot/joy_bs', Joy, sbypass.joy_callback)
        loopRate = rospy.Rate(sbypass.loop_frequency)                       # Loop rate frequency (25 Hz)
        while not rospy.is_shutdown():
            sbypass.publish_joy()
            print("***   ***   ***\n")
            loopRate.sleep()                                                # Sleep for the reqd. amount of time to maintain the loop rate frequency
    except rospy.ROSInterruptException:
        rospy.logerr("*** ERROR *** \n")
        pass
