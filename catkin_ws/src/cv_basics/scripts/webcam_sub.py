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




# # Instantiate CvBridge



# def image_callback(msg):
#     print("Received an image!")

#     try:
#         # Convert your ROS Image message to OpenCV2
#         cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")
#         image = imutils.resize(cv2_img, width=640)
#         predict_thread = multiprocessing.Process(target=predict_segmentaion,args=[image, predictor, metadata])
#         predict_thread.start()
#     except CvBridgeError as e:
#         print(e)
#     else:
#         # Save your OpenCV2 image as a jpeg 
#         cv2.imshow("live_video", cv2_img)
        
#         cv2.waitKey(3)

# def main():
#     rospy.init_node('image_listener')
#     # Define your image topic
#     image_topic = "/img"
#     # Set up your subscriber and define its callback
#     rospy.Subscriber(image_topic, Image, image_callback)
#     # Spin until ctrl + c
#     rospy.spin()

#!/usr/bin/env python


import threading
import time

class InferenceNode:
    def __init__(self):
        rospy.init_node('inference_node')
        # Publisher and Subscriber
        self.subscriber = rospy.Subscriber("/img", Image, self.data_callback)
        self.publisher = rospy.Publisher('/output_data', Image, queue_size=10)
        self.bridge = CvBridge()
        # Thread control
        self.data_queue = []
        self.lock = threading.Lock()
        self.running = True
        # self.one_former_predict = model_inference()
        self.call_model = model_inference_hg("mask2former")

        self.processing_thread = threading.Thread(target=self.run_inference)
        self.processing_thread.start()
        
        self.spinner = rospy.Rate(100)  
        print("InferenceNode initialized")

    def data_callback(self, msg):
        with self.lock:
            self.data_queue.append(msg)
            # print("Data received")

    def run_inference(self):
        print("inference thread started")
        while self.running:
            with self.lock:
                if self.data_queue:
                    # print("Queue length",len(self.data_queue))
                    input_data = self.data_queue.pop(0)
                else:
                    input_data = None

            if input_data is not None:
                print("Processing data")
                input_image = self.bridge.imgmsg_to_cv2(input_data, "bgr8")
                input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
                input_image = PILImage.fromarray(input_image)
                output_data = self.perform_inference(input_image)
                print(len(self.data_queue))
                output_image_msg = self.bridge.cv2_to_imgmsg(output_data, "bgr8")
                self.publisher.publish(output_image_msg)
                print("Published output image") 

            

    def perform_inference(self, image_data):
        print("Performing inference") 
        # image = imutils.resize(image_data, width=640)
        image = image_data
        output = self.call_model.predict_segmentaion(image)
        result = self.call_model.visualization(output, image)
        # output = self.one_former_predict.predict_segmentaion(image)

        return result

    def start(self):
        threading.Thread(target=self.run_inference).start()
        print("Thread started")

    def stop(self):
        self.running = False

if __name__ == '__main__':
    node = InferenceNode()
    try:
        node.start()
        rospy.spin()  
    except rospy.ROSInterruptException:
        node.stop()



