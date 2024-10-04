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
#from pypcd4 import PointCloud


def full_pc_to_local(full_pc):
    #print(full_pc.min(axis=1))
    radial_distance = np.linalg.norm(full_pc[:, 0:2], axis=1)
    #print((-1.4 < full_pc[:, 2]).shape, (full_pc[:, 2] < 0.9).shape)
    floor, ceil = (-1.4 < full_pc[:, 2]), (full_pc[:, 2] < 0.9)
    vert = floor * ceil
    #print(vert.shape)
    idx = np.argwhere((np.abs(radial_distance) < 7) * vert).squeeze()
    #print(idx.shape)
    return full_pc[idx, :]


resolution = 0.1

#load the training folders
SEQUENCE_NAMES = ['IRL', 'ECOT', 'RUSTANDY']
SEQUENCE_NAMES = ['IRL']
dataset_dir = '/home/brendan/spot_data/dataset/'
raw_data = 'raw/'
trajs_dir = 'trajs/'

gen = PointToVoxel(vsize_xyz=[0.05, 0.05, 0.05],
                        coors_range_xyz=[-10, -10, -10, 10, 10, 10],
                        num_point_features=3,
                        max_num_voxels=65536,
                        max_num_points_per_voxel=1)

# pc_files = [s for s in os.listdir(os.path.join(dataset_dir, raw_data, SEQUENCE_NAMES[0], 'point_clouds/'))]
# pc = o3d.io.read_point_cloud(os.path.join(dataset_dir, raw_data, SEQUENCE_NAMES[0], 'point_clouds',pc_files[0] ))
# print(np.asarray(pc.points).shape)
# o3d.visualization.draw_geometries([pc])



for sequence_name in SEQUENCE_NAMES:
    #load the rgb data
    odom_files = [s for s in os.listdir(os.path.join(dataset_dir, raw_data, sequence_name, 'odometry/'))]
    odom_files = natsorted(odom_files)
    trans_files = [s for s in os.listdir(os.path.join(dataset_dir, raw_data, sequence_name, 'transforms/'))]
    trans_files = natsorted(trans_files)
    pc_files = [s for s in os.listdir(os.path.join(dataset_dir, raw_data, sequence_name, 'point_clouds/'))]
    pc_files = natsorted(pc_files)
    #get poses form files
    cond_path = os.path.join(dataset_dir, trajs_dir, sequence_name, 'conditioning')
    os.makedirs(cond_path, exist_ok=True)
    for idx, pc_file in enumerate(pc_files):
        print(sequence_name, " ", pc_file)
        #load the image data
        pc = o3d.io.read_point_cloud(os.path.join(dataset_dir, raw_data, sequence_name, 'point_clouds', pc_file))
        pc = np.asarray(pc.points)
        print(pc.shape)

        local_pc = full_pc_to_local(pc)
        print(local_pc.shape)
        #load the point data
        # pc = np.loadtxt("data/full_conditioning_pcs/" + house_name + "_pc_" + str(idx) + ".npy")
        # colors = image.reshape(-1, image.shape[2])

        # #create the full rgbd pc
        # rgbd_pc = np.append(pc.T, colors, axis = 1)
        
        

        voxels_th, indices_th, num_p_in_vx_th = gen(torch.tensor(pc), empty_mean = True)
        voxels_np = voxels_th.numpy() 
        conditioning_voxel_points = np.reshape(voxels_np, (-1,3))
        #get average point location per voxel
        #avereaged_voxels = np.mean(voxels_np, axis = 1)

        np.save(os.path.join(dataset_dir, trajs_dir, sequence_name, 'conditioning', pc_file.split('.')[0] + '.npy'), conditioning_voxel_points)
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(conditioning_voxel_points[:,0:3])
        # pcd.colors = o3d.utility.Vector3dVector(conditioning_voxel_points[:,3:6]/255)
        # o3d.visualization.draw_geometries([pcd])