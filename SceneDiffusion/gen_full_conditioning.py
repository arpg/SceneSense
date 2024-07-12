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
gen = PointToVoxel(vsize_xyz=[0.01, 0.01, 0.01],
                        coors_range_xyz=[-10, -10, -10, 10, 10, 10],
                        num_point_features=6,
                        max_num_voxels=65536,
                        max_num_points_per_voxel=1)

for house_name in training_dirs:
    #load the rgb data
    files = os.listdir("/home/arpg/Documents/habitat-lab/full_training_data/" + house_name )
    #get poses form files
    rgb_files = [s for s in files if s.startswith("rgb")]
    rgb_files = natsorted(rgb_files)
    for idx, rgb_file in enumerate(rgb_files[0:-1]):
        print(house_name, " ", rgb_file)
        #load the image data
        image = cv2.imread("/home/arpg/Documents/habitat-lab/full_training_data/" + house_name + "/" + rgb_file)
        #load the point data
        pc = np.loadtxt("data/full_conditioning_pcs/" + house_name + "_pc_" + str(idx) + ".npy")
        colors = image.reshape(-1, image.shape[2])

        #create the full rgbd pc
        rgbd_pc = np.append(pc.T, colors, axis = 1)
        
        

        voxels_th, indices_th, num_p_in_vx_th = gen(torch.tensor(rgbd_pc), empty_mean = True)
        voxels_np = voxels_th.numpy() 
        conditioning_voxel_points = np.reshape(voxels_np, (-1,6))
        #get average point location per voxel
        # avereaged_voxels = np.mean(voxels_np, axis = 1)

        np.save("data/full_conditioning_rgbd/" + house_name + "_rgbd_" + str(idx) + ".npy", conditioning_voxel_points)
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(conditioning_voxel_points[:,0:3])
        # pcd.colors = o3d.utility.Vector3dVector(conditioning_voxel_points[:,3:6]/255)
        # o3d.visualization.draw_geometries([pcd])