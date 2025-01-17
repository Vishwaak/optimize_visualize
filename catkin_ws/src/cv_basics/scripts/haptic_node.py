#! /usr/bin/python3
import rospy
import math
from std_msgs.msg import Float32
from temp_hacptic import haptic_controller
from multiprocessing import Queue
from time import sleep

import threading


class haptic_node:
    def __init__  (self):
        rospy.init_node('haptic_node', anonymous=True)
        self.magnitude_data = rospy.Subscriber('/obstacle_distance', Float32, self.distance_callback)
        self.haptic = haptic_controller()
        self.rate = rospy.Rate(10)
        self.min_dst = 40
        self.max_dst = 70     
        self.scaleL = 50
        self.scaleR = 255
        self.case = 2

        self.dst_dirc = Queue()

        self.haptic_thread = threading.Thread(target=self.run_node, args=(self.dst_dirc,))
    
    def haptic_force(self, x):
        if x < self.min_dst:
            return 0
        elif x > self.max_dst:
            return 0
        if self.case == 1:
            slope = -1 / (self.max_dst - self.min_dst)
            y  = slope * (x - self.min_dst) + 1
            return y
        elif self.case == 2:
            norm = self.max_dst - self.min_dst
            y = ((self.max_dst - x) / norm)**2
            return y

        elif self.case ==3:
            y = 1/math.exp((x**2)*math.pi)
            return y
    
    def directional(self, angle):
        
        input_angle = math.radians(angle)
        right_haptic = 0.5 + (math.sin(input_angle))/2
        left_haptic = 0.5 + (math.sin(-input_angle))/2
        return left_haptic, right_haptic
    
    
    def haptic_feeback(self, distance, direction):
        force = self.haptic_force(distance)
        print(force)
        left, right = self.directional(direction)
        right = right * force
        left = left * force

        return left, right
    
    def distance_callback(self, data):
        distance = data.data
        direction = 0
        self.dst_dirc.put((distance, direction))
    
    def run_node(self, dst_dirc):
        while True:
            print("Here")
            if not dst_dirc.empty():
                dst, dirc = dst_dirc.get()
                dst = dst * 100
                print(dst)
                left_haptic, right_haptic = self.haptic_feeback(dst, dirc)
                print("left",left_haptic*self.scaleL,"right",right_haptic*self.scaleR)
                self.haptic.vibrate_both(intensityL=math.floor(left_haptic*self.scaleL), intensityR=math.floor(right_haptic*self.scaleR))
                print("I came here")
                sleep(0.1)
            else:
                self.haptic.reset_hacptic()
                sleep(0.1)
            # haptic_controller.reset_hacptic()
        haptic_controller.close()
    
    def start_node(self):
        self.haptic_thread.start()
        
            

if __name__ == '__main__':
    print("hello2")
    node = haptic_node()
    try:
        node.start_node()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass