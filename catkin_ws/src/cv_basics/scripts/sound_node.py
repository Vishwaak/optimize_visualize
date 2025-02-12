#! /usr/bin/python3
import rospy

from multiprocessing import Queue
from time import sleep

from kairos_laser.msg import obstacle
from cv_basics.msg import object_type

import sounddevice as sd
import soundfile as sf
import os
import threading

from time import sleep

class sound_node:

    def __init__(self):

        self.min_dst = 2
        self.max_dst = 30
        self.mid_dst = (self.min_dst + self.max_dst) / 2
        rospy.init_node('sound_node', anonymous=True)
        self.sound_dict ={
            "near": "sounds/near.wav",
            "far": "sounds/far.wav",
            "left": "sounds/left.wav",
            "right": "sounds/right.wav",
            "front": "sounds/front.wav",
            "back": "sounds/back.wav",
            "box": "sounds/obstacle.wav",
            "human": "sounds/human.wav",
        }

        self.obsatcle_type = rospy.Subscriber('/object_type', object_type, self.play_sound)
        self.obstacle_info = rospy.Subscriber('/obstacle_info', obstacle, self.play_sound)

        self.rate = rospy.Rate(100) 

        self.obs_info = None
        self.obs_type = None

        self.lock = threading.Lock()

#info contains the distance
#type contains the type of object

    def refresh_rate(self, x):
        limit = ((x - self.min_dst) / (self.max_dst - self.min_dst)) * 100
        if limit < 25:
            return 0.9
        elif limit < 50:
            return 1.5
        elif limit < 75:
            return 2
        else:
            return 2
    

    def play_sound(self, data):
        with self.lock:
            if isinstance(data, obstacle):
                self.obs_info = data
            elif isinstance(data, object_type):
                self.obs_type= data
            if self.obs_info and self.obs_type:
                self.choose_sound(self.obs_info, self.obs_type)

    
    def get_dir_sound(self, direction):
        if direction > 180 and direction <= 270:
            return self.sound_dict["back"]
        elif direction > 90 and direction <= 180:
            return self.sound_dict["right"]
        elif direction > 0 and direction <= 90:
            return self.sound_dict["front"]
        elif direction > 270 and direction <= 360:
            return self.sound_dict["left"]
        else:
            return None

            

    def choose_sound(self, obstacle_info, obstacle_type):

        distance = obstacle_info.distance.data * 10 
        direction = obstacle_info.direction.data

        print("Distance", distance)

        if obstacle_type.human == True:
            obs_type = "human"
        elif obstacle_type.box == True:
            obs_type = "box"
        else:
            obs_type = "none"
        
        sounds_play = []
        
        #approaching obstacle on left
        #warning left

        #TODO: add more directions
        #TODO: Help direction if ideal for more time spot the free direction.
        #TODO: use closed obstacle to determine the direction or go fancy and use the camera
        #TODO: Ask how many times they need assitance
        #TODO: set common parameters

        if distance < self.max_dst:
            if self.min_dst < distance and distance < self.mid_dst:
                sounds_play.append(self.sound_dict["near"])
            elif self.mid_dst < distance:
                sounds_play.append(self.sound_dict["far"])
            if obs_type != "none":
                sounds_play.append(self.get_dir_sound(direction))
                sounds_play.append(self.sound_dict[obs_type])

        print("Sounds to play", sounds_play)


        if len(sounds_play) > 1:
            for sounds in sounds_play:
                filename = sounds
                base_path = os.path.dirname(os.path.abspath(__file__))
                filename = os.path.join(base_path, filename)
                data, fs = sf.read(filename, dtype='float32')
                print("Playing sound")
                sd.play(data, fs)
                sleep(self.refresh_rate(distance))


if __name__ == "__main__":
    
    sound_node()
    rospy.spin()