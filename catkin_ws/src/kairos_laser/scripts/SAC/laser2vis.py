#! /usr/bin/python3

import numpy as np
from scipy.spatial import KDTree
from matplotlib import pyplot as plt

import rospy
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
from sensor_msgs import point_cloud2

import open3d as o3d

import statsmodels.api as sm


def median_filter_point_cloud(points, k=65):

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

def lowes_filtering(points):
    points = np.asarray(points)
    lowess = sm.nonparametric.lowess
    x = points[:, 0]
    y = points[:, 1]
    w = lowess(y, x, frac=0.1)
    return w


def split_points(points, threshold_distance=0.1):

    overlap_points_left = []
    overlap_points_right = []
    non_overlap = []
   
    
    front_points = np.asarray(points[:len(points)//2])
    back_points = np.asarray(points[len(points)//2:])

    front_pcl = o3d.geometry.PointCloud()
    back_pcl = o3d.geometry.PointCloud()

    front_pcl.points = o3d.utility.Vector3dVector(front_points)
    back_pcl.points = o3d.utility.Vector3dVector(back_points)

    front_pcl_tree = o3d.geometry.KDTreeFlann(front_pcl)
    back_pcl_tree = o3d.geometry.KDTreeFlann(back_pcl)


    for fpoint, bpoint in zip(front_points, back_points):
        _, idxb, _ = back_pcl_tree.search_knn_vector_3d(fpoint, 1)
        _, idxf, _ = front_pcl_tree.search_knn_vector_3d(bpoint, 1)

        closest_point_back = back_points[idxb[0]]
        closest_point_front = front_points[idxf[0]]

        distance_back = np.linalg.norm(fpoint - closest_point_back)
        distance_front = np.linalg.norm(bpoint - closest_point_front)

        if distance_back > threshold_distance:
            non_overlap.append(fpoint)
        else:
            if fpoint[0] < 0:
                overlap_points_right.append(fpoint)
            else:
                overlap_points_left.append(fpoint)
           

        if distance_front > threshold_distance:
            non_overlap.append(bpoint)
        else:
            if bpoint[0] < 0:
                overlap_points_right.append(bpoint)
            else:
                overlap_points_left.append(bpoint)
            
    filtered_points = {"overlap_left": overlap_points_left, "overlap_right": overlap_points_right, "non_overlap": non_overlap}

   
    return filtered_points


    

class filter_point_cloud:
    def __init__(self):
        rospy.init_node("plotting_node", anonymous=True)
        self.pcld_pts = rospy.Subscriber("/merged_transformed_pcld2", PointCloud2, self.laser_callback, queue_size=1)
        self.filtred_points_median = rospy.Publisher("/filtered_point_cloud_m", PointCloud2, queue_size=1)
        self.filtred_points_lowes = rospy.Publisher("/filtered_point_cloud_l", PointCloud2, queue_size=1)
        self.filtred_points_icp = rospy.Publisher("/filtered_point_cloud_icp", PointCloud2, queue_size=1)
        self.plot_counter = 1000
    
    

    def save_points(self, points = [], names=["filtered_points"]):
        for name, type_point in zip(names, points):
            np.save(f"plot_points/{name}.npy", type_point)

    
    def call_filter(self, points, call_func = ""):

        func_names = {"median": median_filter_point_cloud, "lowes": lowes_ratio_filtering, "lowes2": lowes_filtering}
        filtered_points = np.concatenate((func_names[call_func](points["overlap_left"]), func_names[call_func](points["overlap_right"])), axis=0)

        return filtered_points

    def laser_callback(self, data):
        cloud_points = []
        for p in point_cloud2.read_points(data, field_names=("x", "y","z")):
            cloud_points.append(p)

        filtered_cloud = split_points(cloud_points)

        
        # add filter calls here
        filtered_cloud_median = self.call_filter(filtered_cloud, "median")
        # filtered_cloud_lowes = self.call_filter(filtered_cloud, "lowes")
        filtered_cloud_lowes2 = self.call_filter(filtered_cloud, "lowes2")

        

        if self.plot_counter % 1000 == 0:
            self.save_points(
                [filtered_cloud_median, filtered_cloud["overlap_left"],filtered_cloud_lowes2],
                ["median_points", "overlap_left","lowes_points"]
                )
            self.plot_counter = 1
        else:
            self.plot_counter += 1 

        
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "robot_base_link"

        pcl2_msg_m =  point_cloud2.create_cloud_xyz32(header, filtered_cloud_median)
        pcl2_msg_i =  point_cloud2.create_cloud_xyz32(header, filtered_cloud_median)

        self.filtred_points_median.publish(pcl2_msg_m)
        self.filtred_points_icp.publish(pcl2_msg_i)


        print("publishing filtered data")
       





if __name__ == '__main__':
    
    test = filter_point_cloud()
    rospy.spin()
    

  

    




    # threshold = 0.5

    # print("Overlap Front: ", len(overlap_front))
    # print("Overlap Back: ", len(overlap_back))
    # print("Non Overlap: ", len(non_overlap))

    # front_pcl.points = o3d.utility.Vector3dVector(overlap_front)
    # back_pcl.points = o3d.utility.Vector3dVector(overlap_back)

    # icp_result = o3d.pipelines.registration.registration_icp(
    #     front_pcl, back_pcl, threshold, np.eye(4),
    #     o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    #     o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1000)
    # )

    # front_pcl = front_pcl.transform(icp_result.transformation)

    # filtered_points = non_overlap
    # filtered_points = np.asarray(front_pcl.points)
    # filtered_points = np.asarray(back_pcl.points)
    # filtered_points = np.concatenate((filtered_points, np.asarray(back_pcl.points)), axis=0)
    # filtered_points = np.concatenate((filtered_points, np.asarray(non_overlap)), axis=0)
    # filtered_points = median_filter_point_cloud(filtered_points, k=10)
    
    # filtered_points = np.concatenate((overlap_front, overlap_back), axis=0)


