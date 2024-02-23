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
import re
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

torch_device = "cpu"
#heres the short outputs
# model = UNet2DConditionModel.from_pretrained("alre5639/full_rgbd_unet_512_more_pointnet_short", revision = "adde767c4a5238d047434ba5f2464751b30187d5")
# conditioning_model = get_model()
# conditioning_model.load_state_dict(torch.load("/home/arpg/Documents/SceneDiffusion/data/full_sim_pointnet_weights_more_pointnet_short/145"))
#hres the full output
model = UNet2DConditionModel.from_pretrained("alre5639/full_rgbd_unet_512_more_pointnet", revision = "b063adc01ea748b7a4dbfb7e180eedf741aef536")
conditioning_model = get_model()
conditioning_model.load_state_dict(torch.load("/hdd/sceneSense_data/data/full_sim_pointnet_weights_more_pointnet/171"))

#make sure all the data moves through the network correctly
sample_noise_start = torch.randn(1,23,40, 40)
sample_noise_target = torch.randn(1,223,40, 40)
sample_pc_in = torch.randn(1, 6, 800)
#input to pointnet needs to be shape: 1, 3, 65536
sample_conditioning = conditioning_model(sample_pc_in)
#need to swap axis 1 and 2 to get it in the right shape
sample_conditioning = sample_conditioning.swapaxes(1, 2)
#output from pointnet neeeds to be shape: 1,n, channels
print(sample_conditioning.shape)
print("Unet output shape:", model(sample_noise_start, timestep=1.0, encoder_hidden_states=sample_conditioning).sample.shape)

########################
#get gt data
#########################3
gt_dir = "data/full_gt_pointmaps/"
gt_files = natsorted(os.listdir(gt_dir))

# ##################################
# #get the conditioning data
# ##################################
cond_dir = "data/full_conditioning_rgbd/"
cond_files = natsorted(os.listdir(cond_dir))


#make sure the gt and conditioning data are aligned
aligned_gt_files = []
aligned_cond_files = []
for i in range(len(cond_files)):
    split_gt_files = re.split(r'[._]', gt_files[i])
    split_conditioning_files = re.split(r'[._]', cond_files[i])
    if split_gt_files[0] == split_conditioning_files[0] and int(split_gt_files[1]) ==  int(split_conditioning_files[2]):
        # print(split_gt_files," ", split_conditioning_files)
        aligned_gt_files.append(gt_files[i])
        aligned_cond_files.append(cond_files[i])
    else:
        # print(split_gt_files," ", split_conditioning_files)
        gt_files.pop(i)
        # break
for i in range(len(aligned_cond_files)):
    print(aligned_cond_files[i], " ", aligned_gt_files[i])


file_num = 14
####################################
#load a conditioning file and pass through pointnet
#####################################
single_cond_data = np.load(cond_dir + aligned_cond_files[file_num])
##########################3
#view the conditionign data
###############################
# print(single_cond_data.shape)
pcd_in = o3d.geometry.PointCloud()
pcd_in.points = o3d.utility.Vector3dVector(single_cond_data[:,0:3])
# o3d.visualization.draw_geometries([pcd_in])

single_cond_data = single_cond_data.T
single_cond_data = single_cond_data[None, :, :]
single_cond_tensor = conditioning_model(torch.tensor(single_cond_data.astype(np.single)).to(torch_device))
post_model_conditioning_batch = single_cond_tensor.swapaxes(1, 2)
print(post_model_conditioning_batch.shape)
##############################3
# Load GT file
####################################
gt_data = np.load(gt_dir + aligned_gt_files[file_num])
gt_points =utils.pointmap_to_pc(gt_data,
                                         voxel_size = 0.1,
                                         x_y_bounds = [-2, 2],
                                          z_bounds = [-1.4, 0.9])

pcd_gt = o3d.geometry.PointCloud()
pcd_gt.points = o3d.utility.Vector3dVector(gt_points)
colors = np.zeros((len(np.asarray(pcd_gt.points)), 3))
pcd_gt.colors = o3d.utility.Vector3dVector(colors)

# f = open("/home/arpg/Documents/habitat-lab/out_training_data/sample_octomap_running.txt", "r")

# final_pointcloud = np.zeros((1,3,65536), dtype=np.single)

# node_count = 0

# for x in f:
#     if x[0:4] == "NODE":
#         if node_count == 1:
#             final_pointcloud = copy.deepcopy(pointcloud)
#         elif node_count == 0:
#             pass
#         #add a check that stops the conditoning building when it is the same size as the gt
#         elif node_count == 15:
#             break
#         else:
#             final_pointcloud = np.append(final_pointcloud, pointcloud, axis = 0)
        
#         node_count += 1
#         pc_count = 0
#         #make empty pointcloud to fill  
#         pointcloud = np.zeros((1,3,65536), dtype=np.single)
#         print(final_pointcloud.shape)
#     else:
#         coord = np.fromstring(x, dtype=np.single, sep=' ')
#         pointcloud[0,:,pc_count] = coord
#         pc_count += 1

        
# # #   print(x)

# #need to transform arr into the same format as the input points
# #basicly un volxelize it

# # print(np.max(arr, axis=0))
# # print(np.min(arr, axis=0))

# #####################################
# #This works for the gt pointmaps 
# #########################################

# # # load GT local PC:
# gt_points = np.load("/home/arpg/Documents/open3d_from_habitat/training_pointmaps/pointmap_" + str(14) + ".npy")
# print(gt_points.shape)
# gt_arr = np.empty((0,3), float)
# for idx_x,x in enumerate(gt_points):
#     for idx_y, y in enumerate(x):
#         for idx_z, z in enumerate(y):
#             if z == 1:
#                 gt_arr = np.append(gt_arr, np.array([[idx_x,idx_y,idx_z]]), axis=0)

# #i flipped these values and might have messed it up...
# gt_arr[:,0] = abs(gt_arr[:,0]-29)

# idx_vals = 1/(30 - 1)
# idx_vals_Z = 1/(22 - 1)

# #scale output pc:
# gt_arr = gt_arr/10

# #load the pose and rot:
# curr_pose = np.loadtxt("/home/arpg/Documents/habitat-lab/out_training_data/pose_" + str(13) + ".txt")
# curr_rot = np.loadtxt("/home/arpg/Documents/habitat-lab/out_training_data/rot_" + str(13) + ".txt")
# flipped_gt_points = copy.deepcopy(gt_arr)
# # flipped_gt_points[:,0] = gt_arr[:,2]
# #z is for sure the second term
# #this is the z values which are +1.5
# flipped_gt_points[:,0] = gt_arr[:,0] # - 1.5
# flipped_gt_points[:,1] = gt_arr[:,2] #- curr_pose[1]
# flipped_gt_points[:,2] = gt_arr[:,1] #- 1.5
# # flipped_gt_points[:,0] = abs(gt_arr[:,0])
# # rot = rotation_val.as_quat()
# rotation_val = Rotation.from_rotvec(curr_rot)
# pose_mat = homogeneous_transform([1.5,curr_pose[1], 1.5],rotation_val.as_quat())

# # points = np.loadtxt("/home/arpg/Documents/open3d_from_habitat/training_pointclouds/local_pc_" + str(14) + ".txt")
# # #load the pose:


# ###########################
# # this is for the diffused map
# ######################################

# final_pointcloud = torch.from_numpy(final_pointcloud)
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
# #do the denoising:
# post_model_conditioning_batch = conditioning_model(final_pointcloud)
# post_model_conditioning_batch = post_model_conditioning_batch.swapaxes(1, 2)
# # print(post_model_conditioning_batch[13].shape)
# test_conditioning = post_model_conditioning_batch[13]
# test_conditioning = test_conditioning[None,:,:]
print("conditioning shape: ", post_model_conditioning_batch.shape)
# point_map = utils.denoise_guided_inference(model, noise_scheduler,post_model_conditioning_batch, 40)
# point_map_np = point_map.numpy()
# print(point_map_np.shape)
# diffused_pointcloud =utils.pointmap_to_pc(point_map_np[0],
#                                          voxel_size = 0.1,
#                                          x_y_bounds = [-2, 2],
#                                           z_bounds = [-1.4, 0.9])

# pcd_diff = o3d.geometry.PointCloud()
# pcd_diff.points = o3d.utility.Vector3dVector(diffused_pointcloud)
# colors = np.zeros((len(np.asarray(pcd_diff.points)), 3))
# colors[:,0] = 1
# colors[:,1] = 0
# colors[:,2] = 0
# pcd_diff.colors = o3d.utility.Vector3dVector(colors)
# o3d.visualization.draw_geometries([pcd_gt,pcd_in])