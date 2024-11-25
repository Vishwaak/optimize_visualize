#! /usr/bin/python3


# rospy for the subscriber
import rospy

# ROS Image message
from sensor_msgs.msg import Image

from cv_basics.msg import poly

# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError

# OpenCV2 for saving an image
import cv2

# from seg_models import model_inference


from PIL import Image as PILImage

import numpy as np

from multiprocessing import Queue

import threading
import time

import multiprocessing as mp
from trt_py import trt_infernce, SegmentVisual

import pycuda.driver as cuda

# Publisher and Subscriber
# robot/front_rgbd_camera/rgb/image_raw


class InferenceNode:
    def __init__(self, flag="onnx"):
        rospy.init_node("inference_node")
        self.subscriber = rospy.Subscriber("/img", Image, self.data_callback)
        self.publisher = rospy.Publisher(
            "/seg_img", Image, queue_size=100, latch=True
        )
        self.poly_data_publisher = rospy.Publisher(
           "/poly_pre", poly, queue_size=100, latch=True
         )
        self.bridge = CvBridge()

        self.data_queue = Queue()
        self.segment_queue = Queue()
        self.poly_queue = Queue()

        self.lock = threading.Lock()
        self.running = True

        self.spinner = rospy.Rate(100)
        self.frame_count = 0
        self.frame_skip = 2

        self.visual_output = SegmentVisual()

        print("InferenceNode initialized")
        self.predict_prc = mp.Process(
            target=self.run_inference,
            args=(self.data_queue, self.segment_queue, self.lock),
        )
        self.seg_prc = threading.Thread(
            target=self.run_visualization, args=(self.segment_queue,)
        )
        self.poly_thrt = threading.Thread(
            target=self.run_poly_visualization, args=(self.poly_queue,)
        )
    def data_callback(self, msg):
        with self.lock:
            self.frame_count += 1
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
                start = time.time()
                class_logits, mask_logits = self.predict_prc.predict(input_image)
                end = time.time()
                # print(" ML time: ", end - start)
                segment_queue.put([class_logits, mask_logits, input_image])

        ctx.pop()

    def run_visualization(self, segment_queue):
        # print("starting visualization thread")
        print("visualization thread started")
        dynamic_frame_skip = 1
       
        while self.running:
            if not segment_queue.empty():
                for i in range(dynamic_frame_skip):
                    if not segment_queue.empty():
                        segment_queue.get()

                seg, mask, img = segment_queue.get()
                start = time.time()
                seg_img, poly_seg = self.visual_output.segment_visual(seg, mask, img)
                seg_vis = cv2.cvtColor(seg_img, cv2.COLOR_RGB2BGR)

                end = time.time()

                processing_time = end - start
                seg_vis = self.bridge.cv2_to_imgmsg(seg_vis, "bgr8")
                poly_seg_msg = self.bridge.cv2_to_imgmsg(poly_seg)
                img= np.array(img)
                poly_seg_img = self.bridge.cv2_to_imgmsg(img, "bgr8")
                
                poly_msg = poly()
                poly_msg.segment_image = poly_seg_msg
                poly_msg.frame = poly_seg_img
                
                self.publisher.publish(seg_vis)
                self.poly_data_publisher.publish(poly_msg)
                print("published")

                if processing_time > 0.1:
                    dynamic_frame_skip = 3
                else:
                    dynamic_frame_skip = 1
                    # Calculate FPS
                # print("FPS: ", 1.0 / processing_time)
    
    def run_poly_visualization(self, poly_queue):
        print("starting poly visualization thread")
        output_queue = Queue()
        self.poly_proc = mp.Process(target=self.visual_output.poly_visual, args=(poly_queue, output_queue))
        self.poly_proc.start()
        while self.running:
            if not output_queue.empty():
               
                poly_image = output_queue.get()
                poly_vis = cv2.cvtColor(poly_image, cv2.COLOR_RGB2BGR)
                poly_vis = self.bridge.cv2_to_imgmsg(poly_vis, "bgr8")
                self.poly_publisher.publish(poly_vis)
                print("publishing poly image")

    def perform_inference(self, image):
        print("Performing inference")
        if self.flag == "onnx":
            output = self.onnx_model.run(image)
        else:
            output = self.call_model.predict_segmentaion(image)

        return output

    def start(self):
        self.predict_prc.start()
        self.seg_prc.start()
        # self.poly_thrt.start()

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
