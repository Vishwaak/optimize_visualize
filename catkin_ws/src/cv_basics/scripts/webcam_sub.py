#! /usr/bin/python3


# rospy for the subscriber
import rospy

# ROS Image message
from sensor_msgs.msg import Image

# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError

# OpenCV2 for saving an image
import cv2

# from seg_models import model_inference

from seg_models import onnx_infernce
from seg_models import Visualizer

from PIL import Image as PILImage

import numpy as np

from multiprocessing import Queue

import threading
import time

import multiprocessing as mp
from trt_py import trt_infernce, visualize_image

import pycuda.driver as cuda

# Publisher and Subscriber
# robot/front_rgbd_camera/rgb/image_raw


class InferenceNode:
    def __init__(self, flag="onnx"):
        rospy.init_node("inference_node")
        self.subscriber = rospy.Subscriber("/img", Image, self.data_callback)
        self.publisher = rospy.Publisher(
            "/seg_img_1", Image, queue_size=100, latch=True
        )
       

        self.bridge = CvBridge()

        self.data_queue = Queue()
        self.segment_queue = Queue()

        self.lock = threading.Lock()
        self.running = True

        self.spinner = rospy.Rate(100)
        self.frame_count = 0
        self.frame_skip = 2

        self.visualization = Visualizer()

        print("InferenceNode initialized")
        self.predict_prc = mp.Process(
            target=self.run_inference,
            args=(self.data_queue, self.segment_queue, self.lock),
        )
        self.visul_prc = threading.Thread(
            target=self.run_visualization, args=(self.segment_queue,)
        )

    def data_callback(self, msg):
        with self.lock:
            self.frame_count += 1
            # if self.frame_count % self.frame_skip == 0:
            self.data_queue.put(msg)

    def run_inference(self, data_queue, segment_queue, lock):
        print("inference thread started")
        cuda.init()
        device = cuda.Device(0)
        ctx = device.make_context()
        self.predict_prc = trt_infernce(
            "/home/developer/Desktop/tao/mask2former.engine"
        )
        while self.running:
            with lock:
                print("data queue: ", data_queue.qsize())
                if data_queue:

                    input_data = data_queue.get()

                else:
                    input_data = None

            if input_data is not None:

                input_image = self.bridge.imgmsg_to_cv2(input_data, "bgr8")
                input_image = PILImage.fromarray(
                    cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
                )
                class_logits, mask_logits = self.predict_prc.predict(input_image)
                segment_queue.put([class_logits, mask_logits, input_image])

        ctx.pop()

    def run_visualization(self, segment_queue):
        # print("starting visualization thread")
        print("visualization thread started")
        dynamic_frame_skip = 1
        k = 0
        while self.running:
            if not segment_queue.empty():
                for i in range(dynamic_frame_skip):
                    if not segment_queue.empty():
                        segment_queue.get()

                seg, mask, img = segment_queue.get()
                start = time.time()
                seg_img = visualize_image(seg, mask, img)
                seg_vis = cv2.cvtColor(seg_img, cv2.COLOR_RGB2BGR)

                end = time.time()
                processing_time = end - start

                k += 1
                seg_vis = self.bridge.cv2_to_imgmsg(seg_vis, "bgr8")

                print("segment queue: ", segment_queue.qsize())
                self.publisher.publish(seg_vis)
                print("published")

                if processing_time > 0.1:
                    dynamic_frame_skip = 3
                else:
                    dynamic_frame_skip = 1
                    # Calculate FPS
                print("FPS: ", 1.0 / processing_time)

    def perform_inference(self, image):
        print("Performing inference")
        if self.flag == "onnx":
            output = self.onnx_model.run(image)
        else:
            output = self.call_model.predict_segmentaion(image)

        return output

    def start(self):
        self.predict_prc.start()
        self.visul_prc.start()

    def stop(self):
        self.running = False


if __name__ == "__main__":
    # withtout onnx 0.1
    node = InferenceNode("onnx")
    try:
        node.start()
        rospy.spin()
    except rospy.ROSInterruptException:
        node.stop()
