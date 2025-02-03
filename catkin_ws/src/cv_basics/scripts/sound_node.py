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

    def play_sound(self, data):
        with self.lock:
            if isinstance(data, obstacle):
                print("got obs info")
                self.obs_info = data
            elif isinstance(data, object_type):
                print("got obs type")
                self.obs_type= data

            if self.obs_info and self.obs_type:
                print("im here")
                self.choose_sound(self.obs_info, self.obs_type)

            

    def choose_sound(self, obstacle_info, obstacle_type):

        distance = obstacle_info.distance.data * 10 

        print("Distance", distance)

        if obstacle_type.human == True:
            obs_type = "human"
        elif obstacle_type.box == True:
            obs_type = "box"
        else:
            obs_type = "none"
        
        sounds_play = []
        
        if distance < self.max_dst:
            if self.min_dst < distance and distance < self.mid_dst:
                sounds_play.append(self.sound_dict["near"])
            elif self.mid_dst < distance:
                sounds_play.append(self.sound_dict["far"])
            if obs_type != "none":
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
                sleep(0.9)


if __name__ == "__main__":
    
    sound_node()
    rospy.spin()