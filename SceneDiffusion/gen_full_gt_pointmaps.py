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

#load the training folders
training_dirs = os.listdir("/home/arpg/Documents/habitat-lab/full_training_data/")
print(training_dirs)
for house_name in training_dirs:
    #load the training poses
    files = os.listdir("/home/arpg/Documents/habitat-lab/full_training_data/" + house_name )
    #get poses form files
    pose_files = [s for s in files if s.startswith("pose")]
    pose_files = natsorted(pose_files)
    rot_files = [s for s in files if s.startswith("rot")]
    rot_files = natsorted(rot_files)

    # iterate through each pose
    for i, val in enumerate(pose_files):
        print(house_name, " " ,pose_files[i], " ", rot_files[i])
        pcd = o3d.io.read_point_cloud("/home/arpg/Documents/habitat-lab/full_training_data/" + house_name + "/occ_pc_out.pcd")
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