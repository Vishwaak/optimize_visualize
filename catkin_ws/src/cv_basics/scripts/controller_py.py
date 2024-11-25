from time import sleep
import numpy as np
from dualsense_controller import DualSenseController


class force_feedback:

    def __init__(self, inner_radius, outer_radius):
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
    
    def get_force(self, x):
        if x < self.inner_radius:
            return 0
        elif x < self.outer_radius:
            slope = -1 / (self.outer_radius - self.inner_radius)
            return (slope * (x - self.inner_radius) + 1) * 255
        else:
            return 0
    
    def pattern_vibration(self, x):
        sqaure_pattern = [0.01, 0.05, 0.01, 0.05]
        force = self.get_force(x)
        return [force, sqaure_pattern]

class haptic_controller:

    def __init__(self):
        if len(DualSenseController.enumerate_devices()) < 1:
            raise Exception('No DualSense Controller available.')
        
        print("herllo")
        self.controller = DualSenseController()
        # self.controller.activate()

    def stop(self):
        self.controller.deactivate()

    def activate(self):
        self.controller.activate()
    
    def send_rumble(self, left_rumble, right_rumble):
        self.controller.left_rumble.set(left_rumble)
        self.controller.right_rumble.set(right_rumble)
    
    def button_press(self):
        if self.controller.btn_cross.pressed:
            return 1
        elif self.controller.btn_triangle.pressed:
            return -1
        else:
            return 0


if __name__ == '__main__':
    haptic = force_feedback(5, 10)
    controller = haptic_controller()

   

    controller.activate()
    distance = 0
   
    while True:
        distance = distance + controller.button_press()
        force, pattern = haptic.pattern_vibration(distance)
        print(distance)
        if force > 0:
            for i in range(len(pattern)):
                print(distance)
                controller.send_rumble(0, force)
                print("current pattern: ", pattern[i], "current force: ", force)
                sleep(pattern[i])
       
        sleep(0.01)
      
        
        
        
    
    

