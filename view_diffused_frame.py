from dataclasses import dataclass
from diffusers import UNet2DConditionModel
import torch
from diffusers import DDPMScheduler
import torch.nn.functional as F
from pointnet2_scene_diffusion import get_model
import os
from natsort import natsorted
import numpy as np
import copy
import open3d as o3d
from tqdm.auto import tqdm
import wandb
import random
from huggingface_hub import login
from diffusers.optimization import get_cosine_schedule_with_warmup
import utils.utils as utils
from scipy.spatial.transform import Rotation

def homogeneous_transform(translation, rotation):
    """
    Generate a homogeneous transformation matrix from a translation vector
    and a quaternion rotation.

    Parameters:
    - translation: 1D NumPy array or list of length 3 representing translation along x, y, and z axes.
    - rotation: 1D NumPy array or list of length 4 representing a quaternion rotation.

    Returns:
    - 4x4 homogeneous transformation matrix.
    """

    # Ensure that the input vectors have the correct dimensions
    translation = np.array(translation, dtype=float)
    rotation = np.array(rotation, dtype=float)

    if translation.shape != (3,) or rotation.shape != (4,):
        raise ValueError("Translation vector must be of length 3, and rotation quaternion must be of length 4.")

    # Normalize the quaternion to ensure it is a unit quaternion
    rotation /= np.linalg.norm(rotation)

    # Create a rotation matrix from the quaternion using scipy's Rotation class
    rotation_matrix = Rotation.from_quat(rotation).as_matrix()

    # Create a 4x4 homogeneous transformation matrix
    homogeneous_matrix = np.eye(4)
    homogeneous_matrix[:3, :3] = rotation_matrix
    homogeneous_matrix[:3, 3] = translation

    return homogeneous_matrix

model = UNet2DConditionModel.from_pretrained("alre5639/diff_unet_512_arpg")
# model = UNet2DConditionModel.from_pretrained("alre5639/diff_unet")
conditioning_model = get_model()
conditioning_model.load_state_dict(torch.load("/home/arpg/Documents/SceneDiffusion/full_conditioning_weights/full_cond_model" + str(249)))
# conditioning_model.load_state_dict(torch.load("/home/arpg/Documents/SceneDiffusion/conditioning_model_weights/cond_model" + str(217)))

#make sure all the data moves through the network correctly
sample_noise_start = torch.randn(1,22,30, 30)
sample_noise_target = torch.randn(1,22,30, 30)
sample_pc_in = torch.randn(1, 3, 65536)
#input to pointnet needs to be shape: 1, 3, 65536
sample_conditioning = conditioning_model(sample_pc_in)
#need to swap axis 1 and 2 to get it in the right shape
sample_conditioning = sample_conditioning.swapaxes(1, 2)
#output from pointnet neeeds to be shape: 1,n, channels
print(sample_conditioning.shape)
print("Unet output shape:", model(sample_noise_start, timestep=1.0, encoder_hidden_states=sample_conditioning).sample.shape)


f = open("/home/arpg/Documents/habitat-lab/out_training_data/sample_octomap_running.txt", "r")

final_pointcloud = np.zeros((1,3,65536), dtype=np.single)

node_count = 0

for x in f:
    if x[0:4] == "NODE":
        if node_count == 1:
            final_pointcloud = copy.deepcopy(pointcloud)
        elif node_count == 0:
            pass
        #add a check that stops the conditoning building when it is the same size as the gt
        elif node_count == 15:
            break
        else:
            final_pointcloud = np.append(final_pointcloud, pointcloud, axis = 0)
        
        node_count += 1
        pc_count = 0
        #make empty pointcloud to fill  
        pointcloud = np.zeros((1,3,65536), dtype=np.single)
        print(final_pointcloud.shape)
    else:
        coord = np.fromstring(x, dtype=np.single, sep=' ')
        pointcloud[0,:,pc_count] = coord
        pc_count += 1

        
# #   print(x)

#need to transform arr into the same format as the input points
#basicly un volxelize it

# print(np.max(arr, axis=0))
# print(np.min(arr, axis=0))

#####################################
#This works for the gt pointmaps 
#########################################

# # load GT local PC:
gt_points = np.load("/home/arpg/Documents/open3d_from_habitat/training_pointmaps/pointmap_" + str(14) + ".npy")
print(gt_points.shape)
gt_arr = np.empty((0,3), float)
for idx_x,x in enumerate(gt_points):
    for idx_y, y in enumerate(x):
        for idx_z, z in enumerate(y):
            if z == 1:
                gt_arr = np.append(gt_arr, np.array([[idx_x,idx_y,idx_z]]), axis=0)

#i flipped these values and might have messed it up...
gt_arr[:,0] = abs(gt_arr[:,0]-29)

idx_vals = 1/(30 - 1)
idx_vals_Z = 1/(22 - 1)

#scale output pc:
gt_arr = gt_arr/10

#load the pose and rot:
curr_pose = np.loadtxt("/home/arpg/Documents/habitat-lab/out_training_data/pose_" + str(14) + ".txt")
curr_rot = np.loadtxt("/home/arpg/Documents/habitat-lab/out_training_data/rot_" + str(14) + ".txt")
flipped_gt_points = copy.deepcopy(gt_arr)
# flipped_gt_points[:,0] = gt_arr[:,2]
#z is for sure the second term
#this is the z values which are +1.5
flipped_gt_points[:,0] = gt_arr[:,0] # - 1.5
flipped_gt_points[:,1] = gt_arr[:,2] #- curr_pose[1]
flipped_gt_points[:,2] = gt_arr[:,1] #- 1.5
# flipped_gt_points[:,0] = abs(gt_arr[:,0])
# rot = rotation_val.as_quat()
rotation_val = Rotation.from_rotvec(curr_rot)
pose_mat = homogeneous_transform([1.5,curr_pose[1], 1.5],rotation_val.as_quat())

# points = np.loadtxt("/home/arpg/Documents/open3d_from_habitat/training_pointclouds/local_pc_" + str(14) + ".txt")
# #load the pose:


###########################
# this is for the diffused map
######################################

final_pointcloud = torch.from_numpy(final_pointcloud)
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
#do the denoising:
post_model_conditioning_batch = conditioning_model(final_pointcloud)
post_model_conditioning_batch = post_model_conditioning_batch.swapaxes(1, 2)
print(post_model_conditioning_batch[13].shape)
test_conditioning = post_model_conditioning_batch[13]
test_conditioning = test_conditioning[None,:,:]
point_map = utils.denoise_guided_inference(model, noise_scheduler,test_conditioning, 30)

#this should go in utils 
arr = np.empty((0,3), float)
for idx_x,x in enumerate(point_map[0]):
    for idx_y, y in enumerate(x):
        for idx_z, z in enumerate(y):
            if z == 1:
                arr = np.append(arr, np.array([[idx_x,idx_y,idx_z]]), axis=0)

#i flipped these values and might have messed it up...
arr[:,1] = abs(arr[:,1]-29)

# #first scale it to the size of the inputs which is 3 meters total
arr = arr/10
# #flip the values to be the same direction as the input
# #during training we fliped y and z
flipped_arr = copy.deepcopy(arr)
# flipped_arr[:,0] = arr[:,2]
#z is for sure the second term
flipped_arr[:,1] = arr[:,0]
flipped_arr[:,0] = arr[:,1]
# #then shift it to the pose?

# pose_mat = homogeneous_transform([-1.5,curr_pose[1], -1.5],rotation_val.as_quat())

#this is the input pointcloud
pcd_in = o3d.geometry.PointCloud()
pcd_in.points = o3d.utility.Vector3dVector(final_pointcloud[13].T)
# o3d.visualization.draw_geometries([pcd_local])

pcd_local = o3d.geometry.PointCloud()
pcd_local.points = o3d.utility.Vector3dVector(flipped_arr)
pcd_in.transform(pose_mat)
o3d.visualization.draw_geometries([pcd_local, pcd_in])

# point_map= np.load("training_pointmaps/pointmap_14.npy")
# print(point_map.shape)

# arr = np.empty((0,3), float)
# for idx_x,x in enumerate(point_map):
#     for idx_y, y in enumerate(x):
#         for idx_z, z in enumerate(y):
#             if z == 1:
#                 arr = np.append(arr, np.array([[idx_x,idx_y,idx_z]]), axis=0)


# pcd_local = o3d.geometry.PointCloud()
# pcd_local.points = o3d.utility.Vector3dVector(arr)
# o3d.visualization.draw_geometries([pcd_local])
