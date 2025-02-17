#! /usr/bin/python3

import numpy as np
from scipy.spatial import KDTree
from matplotlib import pyplot as plt

import rospy
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
from sensor_msgs import point_cloud2

import open3d as o3d

def median_filter_point_cloud(points, k=2):

    points = np.asarray(points)
    if points.ndim == 1:
        points = points[:, np.newaxis]

    filtered_points = np.zeros_like(points)
    tree = KDTree(points)

    for i, point in enumerate(points):
        _, indices = tree.query(point, k=k)
        neighbors = points[indices]
        filtered_points[i] = np.median(neighbors, axis=0)

    return filtered_points

def lowes_ratio_filtering(points, threshold=0.75):
    points = np.asarray(points)
    distance = np.linalg.norm(points[:, np.newaxis]- points, axis=2)

    closest_neighbours = np.sort(distance, axis=1)[:,1:3]

    valid_indices = [
        i for i,(d1,d2) in enumerate(closest_neighbours) if d1/d2 < threshold
    ]

    filtered_points = points[valid_indices]
    
    return filtered_points

def icp_filtering(points, threshold=0.02):
    front_pcl = o3d.geometry.PointCloud()
    back_pcl = o3d.geometry.PointCloud()
    front_pcl.points = o3d.utility.Vector3dVector(points[:len(points)//2])
    back_pcl.points = o3d.utility.Vector3dVector(points[len(points)//2:])

    threshold = 0.02

    icp_result = o3d.pipelines.registration.registration_icp(
        front_pcl, back_pcl, threshold, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )

    front_pcl.transform(icp_result.transformation)
    filtered_points = np.asarray(front_pcl.points) + np.asarray(back_pcl.points)
    return filtered_points
    
def plot_point_cloud(points, ax, title=""):
    ax.scatter(points[:, 0], points[:, 1], c='b', s=1)
    ax.set_title(title)
    ax.set_aspect('equal', 'box')
    ax.axis('off')
    # plt.savefig(f'{title}.png')

class plot_point_cloud:
    def __init__(self):
        rospy.init_node("haptic_node", anonymous=True)
        self.pcld_pts = rospy.Subscriber("/merged_transformed_pcld2", PointCloud2, self.laser_callback, queue_size=1)
        self.filtred_points_median = rospy.Publisher("/filtered_point_cloud_m", PointCloud2, queue_size=1)
        self.filtred_points_lowes = rospy.Publisher("/filtered_point_cloud_l", PointCloud2, queue_size=1)
        self.filtred_points_icp = rospy.Publisher("/filtered_point_cloud_icp", PointCloud2, queue_size=1)

    def laser_callback(self, data):
        cloud_points = []
        for p in point_cloud2.read_points(data, field_names=("x", "y","z")):
            cloud_points.append(p)

   
        filtered_cloud_m = median_filter_point_cloud(cloud_points, k=65)
        filtered_cloud_l = lowes_ratio_filtering(cloud_points,0.65)
        filtered_cloud_i = icp_filtering(cloud_points)
        
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "robot_base_link"
        pcl2_msg_m =  point_cloud2.create_cloud_xyz32(header, filtered_cloud_m)
        pcl2_msg_l =  point_cloud2.create_cloud_xyz32(header, filtered_cloud_l)
        pcl2_msg_i =  point_cloud2.create_cloud_xyz32(header, filtered_cloud_i)
        self.filtred_points_median.publish(pcl2_msg_m)
        self.filtred_points_lowes.publish(pcl2_msg_l)
        self.filtred_points_icp.publish(pcl2_msg_i)
        print("publishing filtered data")
        # plot_point_cloud(cloud, ax=plt.subplot(1, 2, 1), title="Original Point Cloud")
        # plot_point_cloud(filtered_cloud, ax=plt.subplot(1, 2, 2), title="Filtered Point Cloud")

if __name__ == '__main__':
    # Example usage:
    # np.random.seed(42)
    # cloud = np.random.rand(100, 2)  # Generate a random point cloud
    # filtered_cloud = median_filter_point_cloud(cloud, k=2)

    # print("Original Point Cloud (first 5 points):\n", cloud[:5])
    # print("Filtered Point Cloud (first 5 points):\n",filtered_cloud[:5])

    # plot_point_cloud(cloud, ax=plt.subplot(1, 2, 1), title="Original Point Cloud")
    # plot_point_cloud(filtered_cloud, ax=plt.subplot(1, 2, 2), title="Filtered Point Cloud")

    test = plot_point_cloud()
    rospy.spin()