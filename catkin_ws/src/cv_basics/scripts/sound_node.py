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

import math

class sound_node:

    def __init__(self):

        self.min_dst = 2
        self.max_dst = 15
        self.mid_dst = (self.min_dst + self.max_dst) / 2
        rospy.init_node("sound_node", anonymous=True)
        self.sound_dict = {
            "near": "sounds/near.wav",
            "far": "sounds/far.wav",
            "left": "sounds/left.wav",
            "right": "sounds/right.wav",
            "front": "sounds/front.wav",
            "back": "sounds/back.wav",
            "box": "sounds/obstacle.wav",
            "human": "sounds/human.wav",
        }

        # self.obsatcle_type = rospy.Subscriber(
        #     "/object_type", object_type, self.play_sound, queue_size=1
        # )
        self.obstacle_info = rospy.Subscriber(
            "/obstacle_info", obstacle, self.play_sound, queue_size=1
        )

        self.rate = rospy.Rate(100)

        self.obs_info = None
        self.obs_type = None

        self.lock = threading.Lock()

    # info contains the distance
    # type contains the type of object

    def refresh_rate(self, x):
        

        norm = self.max_dst - self.min_dst
        x = 1 - (x - self.min_dst) / norm
        func = round(1/math.exp((x**2) * math.pi),1)

        if func < 0.3:
            return 0.3
        elif func > 1:
            return 1
        else:
            return func
        
    def play_sound(self, data):
        with self.lock:
            if isinstance(data, obstacle):
                self.obs_info = data
            # elif isinstance(data, object_type):
            #     self.obs_type = data
            if self.obs_info :
                curr_distance = self.obs_info.distance.data
                if curr_distance < self.max_dst and curr_distance > 0:
                    self.choose_sound(self.obs_info)
    
    def get_dir_sound(self, direction):

        dir_lr = math.cos(math.radians(direction))
        dir_fb = math.sin(math.radians(direction))
        direction = []

        if dir_lr > 0:
            direction.append(self.sound_dict["right"])
        else:
            direction.append(self.sound_dict["left"])
        
        if dir_fb > 0:
            direction.append(self.sound_dict["front"])
        else:
            direction.append(self.sound_dict["back"])
        
        return direction
    
   

    def choose_sound(self, obstacle_info):

        distance = obstacle_info.distance.data * 10
        direction = obstacle_info.direction.data

        print("Distance", distance)

        # if obstacle_type.human == True:
        #     obs_type = "human"
        # elif obstacle_type.box == True:
        #     obs_type = "box"
        # else:
        #     obs_type = "none"

        sounds_play = []

        # approaching obstacle on left
        # warning left

        # TODO: add more directions
        # TODO: Help direction if ideal for more time spot the free direction.
        # TODO: use closed obstacle to determine the direction or go fancy and use the camera
        # TODO: Ask how many times they need assitance
        # TODO: set common parameters

        if distance < self.max_dst:
            if distance < self.mid_dst:
                sounds_play.append(self.sound_dict["near"])
            elif self.mid_dst < distance:
                sounds_play.append(self.sound_dict["far"])
           
            sounds_play.extend(self.get_dir_sound(direction))
            sounds_play.append(self.sound_dict["box"])

        print("Sounds to play", sounds_play)

        if len(sounds_play) > 1:
            for sounds in sounds_play:
                filename = sounds
                base_path = os.path.dirname(os.path.abspath(__file__))
                filename = os.path.join(base_path, filename)
                data, fs = sf.read(filename, dtype="float32")
                print("Playing sound")
                sd.play(data, fs)
                print("current refesh rate", self.refresh_rate(distance))
                sleep(self.refresh_rate(distance))


if __name__ == "__main__":

    sound_node()
    rospy.spin()
