from time import sleep
from pydualsense import pydualsense

class haptic_controller:

    def __init__(self):
        self.ds = pydualsense()
        self.ds.init()
        self.ds.light.setColorI(255,0,0)
        self.counter = 0
    
    def vibrate_left(self, intensity=255):
        self.ds.leftMotor = intensity
        self.ds.rightMotor = 0
    
    def virbate_right(self, intensity=255):
        self.ds.leftMotor = 0
        self.ds.rightMotor = intensity
    
    def vibrate_both(self, intensityL=255, intensityR=255):
        self.ds.leftMotor = self.check_haptic_limit(intensityL)
        self.ds.rightMotor = self.check_haptic_limit(intensityR)
    
    def virbate_direction(self, direction, intensity):
        if direction == 'left':
            self.vibrate_left(intensity[0])
        elif direction == 'right':
            self.virbate_right(intensity[0])
        else:
            self.vibrate_both(intensity[0], intensity[1])

    def vibrate_pattern(self, left, right, time=[], count=10):
        curr_count=0
        while curr_count <= count:
            for index,t in enumerate(time):
                if index%2 == 0:
                    self.vibrate_left(left)
                else:
                    self.virbate_right(right)
                sleep(t)
            # print("current count", curr_count)
            curr_count += 1

    def reset_hacptic(self):
        self.ds.leftMotor = 0
        self.ds.rightMotor = 0
    
    def close(self):
        self.ds.close()


    def check_haptic_limit(self, value):
        if value > 255:
            return 255
         
        return value
       
