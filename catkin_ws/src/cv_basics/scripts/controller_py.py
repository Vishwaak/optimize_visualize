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
        
       
        self.controller = DualSenseController()
        # self.controller.activate()

    def stop(self):
        self.controller.deactivate()

    def activate(self):
        self.controller.activate()
    
    def send_rumble(self, left_rumble, right_rumble):
        self.controller.left_rumble.set(left_rumble)
        self.controller.right_rumble.set(right_rumble)
        self.controller.left_trigger.effect.machine(start_position=1,end_position=9,amplitude_a=2,amplitude_b=7,frequency=5,period=3) 
    
    def button_press(self):
        if self.controller.btn_cross.pressed:
            return 1
        elif self.controller.btn_triangle.pressed:
            return -1
        else:
            return 0


if __name__ == '__main__':
    haptic = force_feedback(5, 10)
    controller1 = haptic_controller()

   

    controller1.activate()
    distance = 0
    # force = 0
    while True:
        force = 500
        controller1.send_rumble(force, force)
        print(distance)
        sleep(0.001)
      
        
        
        
    
    

