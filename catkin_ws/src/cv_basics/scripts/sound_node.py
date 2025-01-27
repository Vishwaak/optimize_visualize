#! /usr/bin/python3
import rospy
import math
from std_msgs.msg import Float32
from temp_hacptic import haptic_controller
from multiprocessing import Queue
from time import sleep

import threading
from kairos_laser.msg import obstacle

import sounddevice as sd
import soundfile as sf

class sound_node:

    def __init__(self):

        self.min_dst = 2
        self.max_dst = 30
        self.mid_dst = (self.min_dst + self.max_dst) / 2
        rospy.init_node('sound_node', anonymous=True)
        self.sound_dict ={
            "near": "sound/near.wav",
            "far": "sound/far.wav",
            "left": "sound/left.wav",
            "right": "sound/right.wav",
            "front": "sound/front.wav",
            "back": "sound/back.wav",
            "obsatcle": "sound/obsatcle.wav",
            "human": "sound/human.wav",
        }

        self.obsatcle_info = rospy.Subscriber('/obstacle_type', obstacle, self.play_sound)
        self.rate = rospy.Rate(100) 


    def choose_sound(self, data):
        distance = data.distance.data  
        obs_type = data.type.data
        sounds_play = []
        if distance < self.max_dst:
            if self.min_dst < distance and distance > self.min_dst:
                sounds_play.append(self.sound_dict["near"])
            if self.max_dst > distance:
                sounds_play.append(self.sound_dict["far"])
            sounds_play.append(self.sound_dict[obs_type])
            
        for sounds in sounds_play:
            filename = sounds
            data, fs = sf.read(filename, dtype='float32')
            sd.play(data, fs)


if __name__ == "__main__":
    
    sound_node()
    rospy.spin()