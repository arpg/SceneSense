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

#SEQUENCE_NAMES = ['IRL', 'ECOT', 'RUSTANDY']
SEQUENCE_NAMES = ['doncey_full']

#load the training folders
dataset_dir = '/home/brendan/realsense_data/dataset/'
raw_data = 'raw/'
trajs_dir = 'trajs/'


def read_pose_txt(filename):
    # Step 1: Read the transformation matrix from the file
    with open(filename, 'r') as file:
        # Read the single line containing the matrix
        matrix_str = file.readline()
 
        numbers_str = matrix_str.replace('[', '').replace(']', '').split()
        # Convert to float and create a numpy array
        numbers = np.array([float(num) for num in numbers_str])
 
        # Step 2: Reshape the flat array to a 4x4 matrix
        transform_matrix = numbers.reshape(4, 4)
 
    return transform_matrix


for sequence_name in SEQUENCE_NAMES:
    
    # odom_path = 
    # os.makedirs(odom_path, exist_ok=True)
    # transforms_path = 
    # os.makedirs(transforms_path, exist_ok=True)
    #get poses form files
    odom_files = [s for s in os.listdir(os.path.join(dataset_dir, raw_data, sequence_name, 'poses/'))]
    odom_files = natsorted(odom_files)
    # trans_files = [s for s in os.listdir(os.path.join(dataset_dir, raw_data, sequence_name, 'transforms/'))]
    # trans_files = natsorted(trans_files)
    
    sequence_path = os.path.join(dataset_dir, raw_data, sequence_name)
    pcd_ = o3d.io.read_point_cloud(os.path.join(sequence_path, 'full_octomap.pcd'))
    
    gt_path = os.path.join(dataset_dir, trajs_dir, sequence_name, 'gt_point_maps')
    os.makedirs(gt_path, exist_ok=True)
    # iterate through each pose
    for i, _ in enumerate(odom_files):
        #print(sequence_name, " " , odom_files[i], " ", trans_files[i])
        
        
        
        pcd = deepcopy(pcd_)
        # pcd = np.asarray(pcd.points)
        # print(pcd.shape)
        # print(pcd.min(axis=0), pcd.max(axis=0))
        # exit()
        rot_matrix = read_pose_txt(os.path.join(dataset_dir, raw_data, sequence_name, 'poses/', odom_files[i]))
        #transformations = yaml.safe_load(open(os.path.join(sequence_path, 'transforms', trans_files[i]), 'r'))
        # rot = [rot[k] for k in ['x', 'y', 'z', 'w']]
        #create the homogeneous transform matrix
        #load rotation to quaternions
        #transform the pcd into the robot frame
        
        pcd.transform(utils.inverse_homogeneous_transform(rot_matrix))
        house_points = np.asarray(pcd.points)
        #get the local points
        local_points = utils.points_within_distance2(0,0,house_points, 2)
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
        np.save(os.path.join(gt_path, odom_files[i].split('.')[0] + '.npy'), pointmap) 
    # break
    # o3d.visualization.draw_geometries([pcd])