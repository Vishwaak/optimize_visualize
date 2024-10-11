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
from PIL import Image as PILImage

import numpy as np


import threading
import time

class InferenceNode:
    def __init__(self):
        rospy.init_node('inference_node')
        # Publisher and Subscriber
        self.subscriber = rospy.Subscriber("/img", Image, self.data_callback)
        self.depth_subscriber = rospy.Subscriber("/depth_img", Image, self.depth_callback)

        self.publish_infer = rospy.Publisher("/output_infer",Image, queue_size=10)
        self.publish_depth = rospy.Publisher("/output_depth",Image, queue_size=10)
        self.publish_combined = rospy.Publisher('/combined', Image, queue_size=10)

        self.bridge = CvBridge()
        # Thread control
        self.data_queue = []
        self.depth_queue = []
        self.lock = threading.Lock()
        self.running = True
        # self.one_former_predict = model_inference()
        self.call_model = model_inference_hg("oneformer")

        self.processing_thread = threading.Thread(target=self.run_inference)
        # self.depth_thread = threading.Thread(target=self.depth_processing)

        # self.depth_thread.start()
        self.processing_thread.start()
        
        self.spinner = rospy.Rate(100)
        self.frame_count = 0
        self.frame_skip = 3
        self.depth_frame_count = 0
        
        print("InferenceNode initialized")

    def data_callback(self, msg):
        with self.lock:
            self.frame_count += 1
            if self.frame_count % self.frame_skip == 0:
                self.data_queue.append(msg)
            # print("Data received")
        
    def depth_callback(self, msg):
        with self.lock:
            self.depth_queue.append(msg)

    def run_inference(self):
        print("inference thread started")
        while self.running:
            with self.lock:
                if self.data_queue:
                    # print("Queue length",len(self.data_queue))
                    input_data = self.data_queue.pop(0)
                    input_depth = self.depth_queue.pop(0)
                else:
                    input_data = None

            if input_data is not None:
                print("Processing data")
                input_image = self.bridge.imgmsg_to_cv2(input_data, "bgr8")
                input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
                input_image = PILImage.fromarray(input_image)
                output_data = self.perform_inference(input_image)
                
                depth_image = self.bridge.imgmsg_to_cv2(input_depth, "pasthrough")
                depth_image = np.nan_to_num(depth_image, nan=0.0)
                depth_image = cv2.medianBlur(depth_image, 5)
                no_depth_mask_far = (depth_image < 70)
                depth_image_color = cv2.cvtColor(depth_image, cv2.COLOR_GRAY2BGR)
                depth_image_color[no_depth_mask_far] = [0, 255, 0]
                
                _, depth_mask = cv2.threshold(depth_image_color, 50, 255, cv2.THRESH_BINARY)
                
                # Convert inference output to binary mask (assuming output_data is a segmentation map)
                _, inference_mask = cv2.threshold(output_data, 127, 255, cv2.THRESH_BINARY)
                
                intersection = np.logical_and(inference_mask, depth_mask)
                
                print(f"Intersection: {np.sum(intersection)}")
                
                output_image_msg = self.bridge.cv2_to_imgmsg(intersection.astype(np.uint8), "mono8")
                self.publisher.publish(output_image_msg)
                print("Published intersection image")
             
                
                
                print(len(self.data_queue))
               
                output_inf_msg = self.bridge.cv2_to_imgmsg(output_data, "bgr8")
                self.publish_infer.publish(output_image_msg)
                
                output_dpth_msg = self.bridge.cv2_to_imgmsg(depth_image, "bgr8")
                self.publish_depth.publish(output_dpth_msg)
                
                print("Published output image") 


    def perform_inference(self, image_data):
        print("Performing inference") 
        image = image_data
        output = self.call_model.predict_segmentaion(image)
        result = self.call_model.visualization(output, image)
        # output = self.one_former_predict.predict_segmentaion(image)

        return result

    def depth_processing(self, depth_data):
        depth_image = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
        depth_image = np.clip(depth_image, 0, 255).astype(np.uint8)
        filtered_image = cv2.medianBlur(depth_image, 3)
        no_depth_mask_far = (filtered_image < 70)
        depth_image_color = cv2.cvtColor(filtered_image, cv2.COLOR_GRAY2BGR)
        depth_image_color[no_depth_mask_far] = [0, 255, 0]
        return depth_image


    def calculate_iou(self, infernce_output, depth_image):
        intersection = np.logical_and(inference_output, depth_image)
        union = np.logical_or(inference_output, depth_image)
        iou = np.sum(intersection) / np.sum(union)

        return iou

    def start(self):
        threading.Thread(target=self.run_inference).start()
        # threading.Thread(target=self.depth_processing).start()
        print("Thread started")

    def stop(self):
        self.running = False

class depth_image:

    def __init__ (self):
        rospy.init_node('depth_image')
        self.subscriber = rospy.Subscriber("/depth", Image, self.depth_callback)
        self.publisher = rospy.Publisher('/output_depth_img', Image, queue_size=10)
        self.bridge = CvBridge()

    def depth_callback(self, msg):
        try:
            depth_image = self.bridge.imgmsg_to_cv2(msg, "passthrough")
            depth_image = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
            depth_image = np.clip(depth_image, 0, 255).astype(np.uint8)
            filtered_image = cv2.medianBlur(depth_image, 3)
            no_depth_mask_far = (filtered_image < 70)
            # no_depth_mask_near = (filtered_image > 50)

            depth_image_color = cv2.cvtColor(filtered_image, cv2.COLOR_GRAY2BGR)
            depth_image_color[no_depth_mask_far] = [0, 255, 0]
            # depth_image_color[no_depth_mask_near] = [0, 0, 255]

        except CvBridgeError as e:
            print(e)
        else:
            depth_image_color = depth_image_color.astype(np.uint8)
            depth_image_color = self.bridge.cv2_to_imgmsg(depth_image_color, "bgr8")
            self.publisher.publish(depth_image_color)   


if __name__ == '__main__':
    node = InferenceNode()
    # node = depth_image()
    try:
        node.start()
        rospy.spin()  
    except rospy.ROSInterruptException:
        node.stop()



