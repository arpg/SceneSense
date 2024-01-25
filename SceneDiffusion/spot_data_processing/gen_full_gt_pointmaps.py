import numpy as np
import open3d as o3d
import SceneDiffusion.utils as utils
import spconv
from spconv.pytorch.utils import PointToVoxel
from spconv.pytorch.core import SparseConvTensor
import torch
import cv2
import os
from natsort import natsorted
from scipy.spatial.transform import Rotation
import yaml
from copy import deepcopy


resolution = 0.1

SEQUENCE_NAMES = ['IRL', 'ECOT', 'RUSTANDY']

#load the training folders
dataset_dir = '/home/brendan/spot_data/dataset/'
raw_data = 'raw/'
trajs_dir = 'trajs/'

for sequence_name in SEQUENCE_NAMES:
    
    #get poses form files
    odom_files = [s for s in os.listdir(os.path.join(dataset_dir, raw_data, sequence_name, 'odometry/'))]
    odom_files = natsorted(odom_files)
    trans_files = [s for s in os.listdir(os.path.join(dataset_dir, raw_data, sequence_name, 'transforms/'))]
    trans_files = natsorted(trans_files)
    
    sequence_path = os.path.join(dataset_dir, raw_data, sequence_name)
    pcd_ = o3d.io.read_point_cloud(os.path.join(sequence_path, 'IRL_lab_and_below.pcd'))
    # iterate through each pose
    for i, _ in enumerate(odom_files):
        print(sequence_name, " " , odom_files[i], " ", trans_files[i])
        
        
        
        pcd = deepcopy(pcd_)
        odometry = np.load(os.path.join(sequence_path, 'odometry', odom_files[i]), allow_pickle=True)
        pose = odometry[:3]
        transformations = yaml.safe_load(open(os.path.join(sequence_path, 'transforms', trans_files[i]), 'r'))
        rot = transformations['transform']['rotation']
        rot = [rot[k] for k in ['x', 'y', 'z', 'w']]
        #create the homogeneous transform matrix
        #load rotation to quaternions
        rotation_obj = Rotation.from_quat(rot)
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
        np.save(os.path.join(dataset_dir, trajs_dir, sequence_name, 'gt_point_maps', odom_files[i].split('.')[0] + '.npy'), pointmap) 
    # break
    # o3d.visualization.draw_geometries([pcd])