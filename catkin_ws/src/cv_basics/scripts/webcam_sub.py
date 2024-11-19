#! /usr/bin/python3


# rospy for the subscriber
import rospy
# ROS Image message
from sensor_msgs.msg import Image
# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError
# OpenCV2 for saving an image
import cv2

import imutils
# from seg_models import model_inference
from seg_models import model_inference_hg
from seg_models import onnx_infernce
from seg_models import Visualizer

from PIL import Image as PILImage

import numpy as np

from multiprocessing import Queue

import threading
import time

import multiprocessing

# Publisher and Subscriber
#robot/front_rgbd_camera/rgb/image_raw

class InferenceNode:
    def __init__(self, flag="onnx"):
        rospy.init_node('inference_node')
        self.subscriber = rospy.Subscriber("/img", Image, self.data_callback)
        self.publisher = rospy.Publisher('/seg_img_1', Image, queue_size=10)
        # self.poly_pub = rospy.Publisher('/poly_img', Image, queue_size=100)

        # self.call_model = model_inference_hg("mask2former", flag)
        self.bridge = CvBridge()
      
        self.data_queue = []
        self.depth_queue = []
        self.lock = threading.Lock()
        self.running = True
        self.processing_thread = threading.Thread(target=self.run_inference)
        
        
        self.spinner = rospy.Rate(100)
        self.frame_count = 0
        self.frame_skip = 2

        self.flag = flag


        self.onnx_model = onnx_infernce()
        self.visualization = Visualizer()

        self.segment_queue = Queue()

        print("InferenceNode initialized")
        self.visualization_thread = threading.Thread(target=self.run_visualization)
      

    def data_callback(self, msg):
        with self.lock:
            self.frame_count += 1
            if self.frame_count % self.frame_skip == 0:
                self.data_queue.append(msg)
            

    def run_inference(self):
        print("inference thread started")
        while self.running:
            with self.lock:
                if self.data_queue:
                    input_data = self.data_queue.pop(0)
                else:
                    input_data = None

            if input_data is not None:
                # print("Processing data")
                input_image = self.bridge.imgmsg_to_cv2(input_data, "bgr8")
                input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
                input_image = PILImage.fromarray(input_image)
                output_data = self.perform_inference(input_image)
                # print("input queue: ",len(self.data_queue))
                self.segment_queue.put([output_data,input_image]) 
                # print("segment queue: ",self.segment_queue.qsize())

    def run_visualization(self):
        # print("starting visualization thread")
        dynamic_frame_skip = 1
        while self.running:
            if not self.segment_queue.empty():
                for i in range(dynamic_frame_skip):
                    if not self.segment_queue.empty():
                        self.segment_queue.get()
                
                seg,img = self.segment_queue.get()
                start = time.time()
                seg_img, poly_vis = self.visualization.visual(seg, img)
                seg_vis = cv2.cvtColor(seg_img, cv2.COLOR_RGB2BGR)
                # poly_img = cv2.cvtColor(poly_vis, cv2.COLOR_RGB2BGR)

                end = time.time()
                processing_time = end - start
                fps = 1.0 / processing_time
                fps_text = f"FPS: {fps:.2f}"
                seg_vis = self.bridge.cv2_to_imgmsg(seg_vis, "bgr8")
                # poly_img = self.bridge.cv2_to_imgmsg(poly_img, "bgr8")

                # print("+++++++++++++++++++Publishing image++++++++++++++++++++", end-start)
                self.publisher.publish(seg_vis)
                # self.poly_pub.publish(poly_img)
                
                print(fps_text)
                
                if processing_time > 0.1:
                    dynamic_frame_skip = 3
                else:
                    dynamic_frame_skip = 1
                    # Calculate FPS
                    


    def perform_inference(self, image):
        print("Performing inference") 
        if self.flag == "onnx":
            output = self.onnx_model.run(image)
        else: 
            output = self.call_model.predict_segmentaion(image)

        return output


    def start(self):
        self.processing_thread.start()
        self.visualization_thread.start()
     
    def stop(self):
        self.running = False



if __name__ == '__main__':
    #withtout onnx 0.1
    node = InferenceNode("onnx")
    try:
        node.start()
        rospy.spin()  
    except rospy.ROSInterruptException:
        node.stop()



