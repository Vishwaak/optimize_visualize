#! /usr/bin/python3

#-------------------------------------------------------------------------------------------------
#   @file       : joy_repacker.py
#   @author     : PARAM D SALUNKHE | UTARI - AIS
#   @comments   : Need to run $rosrun joy joy_node before running this node
#-------------------------------------------------------------------------------------------------
 
#-------------------------------------------------------------------------------------------------
#   Package Imports
#-------------------------------------------------------------------------------------------------

import rospy
from pydualsense import *
from sensor_msgs.msg import Joy


#-------------------------------------------------------------------------------------------------
#   Global Variable Declarations
#-------------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------
#   Class Definitions
#-------------------------------------------------------------------------------------------------

class Repacker:
    #------------------------------------ CLASS CONSTRUCTOR --------------------------------------
    def __init__(self):
        rospy.init_node('ps5_repacker_node', anonymous=True)              # Initializing a ROS node
        self.repack_pub = rospy.Publisher('/joy_ps5', Joy, queue_size=10)
        self.dualsense = pydualsense()
        self.new_msg = Joy()
        self.loop_rate = 25
        
    #--------------------------------- CLASS FUNCTION DEFINITIONS --------------------------------
    # ROS Callback function for the PS5 Subscriber
    def joy_repack_callback(self, ps5_msg):
        self.new_msg = ps5_msg
        # repacking the mismatched message header mappings
        self.new_msg.header.frame_id = "/dev/input/js_base"
        
        # repacking the mismatched message buttons[] mappings
        # btn_list = list(self.new_msg.buttons)
        btn_list = list()
        # btn_list[0] = ps5_msg.buttons[3]
        # btn_list[1] = ps5_msg.buttons[0]
        # btn_list[2] = ps5_msg.buttons[1]
        # btn_list[3] = ps5_msg.buttons[2]
        # # buttons[4] to buttons[10] are mapped correctly
        # btn_list[10] = ps5_msg.buttons[11]
        # btn_list[11] = ps5_msg.buttons[12]
        # btn_list[12] = ps5_msg.buttons[10]
        
        btn_list.append(ps5_msg.buttons[3])
        btn_list.append(ps5_msg.buttons[0])
        btn_list.append(ps5_msg.buttons[1])
        btn_list.append(ps5_msg.buttons[2])
        btn_list.append(ps5_msg.buttons[4])
        btn_list.append(ps5_msg.buttons[5])
        btn_list.append(ps5_msg.buttons[6])
        btn_list.append(ps5_msg.buttons[7])
        btn_list.append(ps5_msg.buttons[8])
        btn_list.append(ps5_msg.buttons[9])
        btn_list.append(ps5_msg.buttons[11])
        btn_list.append(ps5_msg.buttons[12])
        btn_list.append(ps5_msg.buttons[10])
        btn_list.append(0)
        
        self.new_msg.buttons = tuple(btn_list)
        # repacking the mismatched message axes[] mappings
        # axes[0] and axes[1] are mapped correctly
        
        # self.new_msg.axes = [0] * float(0.0)
        ax_list = list()
    
        ax_list.append(ps5_msg.axes[0])
        ax_list.append(ps5_msg.axes[1])
        ax_list.append(ps5_msg.axes[3])
        ax_list.append(ps5_msg.axes[2])
        ax_list.append(ps5_msg.axes[5])
        ax_list.append(ps5_msg.axes[4])
        ax_list.append(0)
        ax_list.append(0)
        ax_list.append(0)
        ax_list.append(ps5_msg.axes[6])
        ax_list.append(ps5_msg.axes[7])
        
        # ax_list[2] = ps5_msg.axes[3]
        # ax_list[3] = ps5_msg.axes[2]
        # ax_list[4] = ps5_msg.axes[5]
        # ax_list[5] = ps5_msg.axes[4]
        # ax_list[9] = ps5_msg.axes[6]
        # ax_list[10] = ps5_msg.axes[7]
        
        self.new_msg.axes = tuple(ax_list)
        
        print(self.new_msg)
        self.repack_pub.publish(self.new_msg)
        return
    
#-------------------------------------------------------------------------------------------------
#   Function Definitions
#-------------------------------------------------------------------------------------------------    

#-------------------------------------------------------------------------------------------------
#   Main Function
#-------------------------------------------------------------------------------------------------
 
if __name__ == '__main__':
    try:
        r = Repacker()
        r.dualsense.init()
        rospy.Subscriber('/joy', Joy, r.joy_repack_callback)
        print("ps5 message repacker running...")
        while not rospy.is_shutdown():
            rospy.spin()
        
    except rospy.ROSInterruptException:
        rospy.logerr("*** ERROR *** \n")
        pass
