import numpy as np
import open3d as o3d
import utils.utils as utils
import spconv
from spconv.pytorch.utils import PointToVoxel
from spconv.pytorch.core import SparseConvTensor
import torch
import cv2
import os
from natsort import natsorted
from scipy.spatial.transform import Rotation

resolution = 0.1

SEQUENCE_NAMES = ['IRL', 'ECOT', 'RUSTANDY']

#load the training folders
dataset_dir = '/home/brendan/spot_data/dataset/'
raw_data = 'raw/'

for sequence_name in SEQUENCE_NAMES:
    
    #get poses form files
    odom_files = [s for s in os.path.join(dataset_dir, raw_data, sequence_name, 'odometry')]
    odom_files = natsorted(odom_files)
    trans_files = [s for s in os.path.join(dataset_dir, raw_data, sequence_name, 'transforms')]
    trans_files = natsorted(trans_files)

    # iterate through each pose
    for i, _ in enumerate(odom_files):
        print(house_name, " " ,pose_files[i], " ", rot_files[i])
        pcd = o3d.io.read_point_cloud(os.path.join(dataset_dir, raw_data, sequence_name, sequence_name, '.pcd'))
        pose = np.loadtxt("/home/arpg/Documents/habitat-lab/full_training_data/" + house_name +"/" + pose_files[i])
        rot = np.loadtxt("/home/arpg/Documents/habitat-lab/full_training_data/" + house_name +"/" + rot_files[i])
        #create the homogeneous transform matrix
        #load rotation to quaternions
        rotation_obj = Rotation.from_rotvec(rot)
        hm_tx_mat = utils.homogeneous_transform(pose, rotation_obj.as_quat())
        #transform the pcd into the robot frame
        pcd.transform(utils.inverse_homogeneous_transform(hm_tx_mat))
        house_points = np.asarray(pcd.points)
        #get the local points
        local_points = utils.points_within_distance(0,0,house_points, 2)
        #remove the lower floors
        local_points = local_points[local_points[:,1] > -1.4]
        #remove the celing
        local_points = local_points[local_points[:,1] < 0.9]
        pcd_local = o3d.geometry.PointCloud()
        pcd_local.points = o3d.utility.Vector3dVector(local_points)
        pointmap = utils.pc_to_pointmap(local_points,
                                         voxel_size = 0.1,
                                         x_y_bounds = [-2, 2],
                                          z_bounds = [-1.4, 0.9])
        print(pointmap.shape)
        np.save("data/full_gt_pointmaps/" + house_name + "_" + str(i) + ".npy", pointmap) 
    # break
    # o3d.visualization.draw_geometries([pcd])