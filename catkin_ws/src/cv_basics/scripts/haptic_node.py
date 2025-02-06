#! /usr/bin/python3
import rospy
import math
from std_msgs.msg import Float32
from temp_hacptic import haptic_controller
from multiprocessing import Queue
from time import sleep

import threading
from kairos_laser.msg import obstacle


class haptic_node:
    def __init__  (self):
        rospy.init_node('haptic_node', anonymous=True)
        self.magnitude_data = rospy.Subscriber('/obstacle_info', obstacle, self.distance_callback, queue_size=1)
        self.haptic = haptic_controller()
        self.rate = rospy.Rate(100)
        self.min_dst = 2
        self.max_dst = 30     
        self.scaleL = 50
        self.scaleR = 255
        self.case = 2

        self.dst_dirc = Queue()

        self.haptic_thread = threading.Thread(target=self.run_node, args=(self.dst_dirc,))

        self.haptic_direction = "vib" #vib or braille

        self.force_dict = []
    
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
        elif self.case == 4:
            norm = self.max_dst - self.min_dst
            x = ((self.max_dst - x) / norm)
            print(x)
            y = -math.log(x**2)
            return y
   

    
    def directional(self, angle):
        
        input_angle = math.radians(angle)
        right_haptic = 0.5 + (math.sin(input_angle))/2
        left_haptic = 0.5 + (math.sin(-input_angle))/2
        return left_haptic, right_haptic
    
    
    def haptic_feeback(self, distance, direction):
        force = self.haptic_force(distance)
        print("force :",force)
        if self.haptic_direction == "vib":
            left, right = self.directional(direction)
        else:
            return force, force
        
        right = right * force
        left = left * force

        return left, right
    
    def distance_callback(self, data):
        distance = data.distance.data
        direction = data.direction.data
        header  =  data.header
        # print("distance", distance, "direction", direction)
        self.dst_dirc.put((distance, direction, header))
    
    def vibrate_pattern(self, forceR, forceL, dir):
        
        if dir > 270 and dir <= 360:
            self.haptic.vibrate_pattern(left=forceL, right=forceR,time=[0.09,0.09], count=1)
        if dir >= 0 and dir <= 90:
            self.haptic.vibrate_pattern(left=forceL, right=forceR,time=[0.09,0.09], count=1)
            self.haptic.vibrate_pattern(left=0, right=forceR,time=[0.09], count=1)
            self.haptic.vibrate_pattern(left=forceL, right=forceR,time=[0.09], count=1)
        if dir > 90 and dir <= 180:
            self.haptic.vibrate_pattern(left=forceL, right=forceR,time=[0.09,0.09], count=1)
        if dir > 180 and dir <= 270:
            self.haptic.vibrate_pattern(left=forceL, right=forceR,time=[0.09,0.09, 0.09], count=1)
    
    def run_node(self, dst_dirc):
        
        while not rospy.is_shutdown():
            if not dst_dirc.empty():
                dst, dirc, header = dst_dirc.get()
                dst = dst * 10
                print("distance", dst, "direction", dirc, "header_time", header.seq)

                left_haptic, right_haptic = self.haptic_feeback(dst, dirc)
                
                print("left",left_haptic*self.scaleL,"right",right_haptic*self.scaleR)

                if self.haptic_direction == "vib":
                    self.haptic.vibrate_both(intensityL=math.floor(left_haptic*self.scaleL), intensityR=math.floor(right_haptic*self.scaleR))
                else:
                    self.vibrate_pattern(math.floor(left_haptic*self.scaleL), math.floor(right_haptic*self.scaleR), dirc)

                if dst < self.max_dst and self.min_dst < dst:
                    sleep(0.03) #0.03
            else:
                self.haptic.reset_hacptic()
                sleep(0.01)
            # haptic_controller.reset_hacptic()
        haptic_controller.close()
    
    def start_node(self):
        self.haptic_thread.start()
        
            

if __name__ == '__main__':
    node = haptic_node()
    try:
        node.start_node()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass