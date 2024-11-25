#! /usr/bin/python3

import rospy

from sensor_msgs.msg import Image
from cv_basics.msg import poly

from trt_py import SegmentVisual
from cv_bridge import CvBridge, CvBridgeError

import cv2

class polyNode:
    def __init__(self):
        rospy.init_node("poly_node")
        self.subscriber = rospy.Subscriber("/poly_pre", poly, self.data_callback)
        self.poly_publisher = rospy.Publisher("/poly_vis", Image, queue_size=100, latch=True)
        self.data_queue = []
        self.bridge = CvBridge()

        self.visual_output = SegmentVisual()
        rospy.Rate(15)
    
    def data_callback(self, data):
        self.data_queue.append(data)
        print("data received")

    def run(self):
        while not rospy.is_shutdown():
            if self.data_queue:
                print("data processing")
                data = self.data_queue.pop(0)
                seg_img = self.bridge.imgmsg_to_cv2(data.segment_image)
                image = self.bridge.imgmsg_to_cv2(data.frame)
                poly_image = self.visual_output.poly_visual(seg_img, image)
                print(poly_image)
                poly_vis = cv2.cvtColor(poly_image, cv2.COLOR_RGB2BGR)
                poly_vis = self.bridge.cv2_to_imgmsg(poly_vis, "bgr8")
                self.poly_publisher.publish(poly_vis)
            else:
                print("no data")
        print("shutting down")

if __name__ == "__main__":
    node = polyNode()
    node.run()
    rospy.spin()


    