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

def points_within_distance(x, y, points, distance):
    """
    Find all 3D points within a specified distance from a given (x, y) location.

    Parameters:
    - x, y: The x and y coordinates of the location.
    - points: NumPy array of shape (num_points, 3) representing 3D points.
    - distance: The maximum distance for points to be considered within.

    Returns:
    - NumPy array of points within the specified distance.
    """

    # Extract x, y coordinates from the 3D points
    #this actually needs to be xz
    xy_coordinates = points[:, [0,2]]

    # Calculate the Euclidean distance from the given location to all points
    distances = np.linalg.norm(xy_coordinates - np.array([x, y]), axis=1)

    # Find indices of points within the specified distance
    within_distance_indices = np.where(distances <= distance)[0]

    # Extract points within the distance
    points_within_distance = points[within_distance_indices]

    return points_within_distance


#load the 100th octomap frame from testing
pcd = o3d.io.read_point_cloud("/home/arpg/Documents/open3d_from_habitat/training_graphs/running_110_frames.pcd")
colors = np.zeros((len(np.asarray(pcd.points)), 3))
# colors[:,0] = colors[:,0]
pcd.colors = o3d.utility.Vector3dVector(colors)

#get the current pose of the platform
coor = o3d.geometry.TriangleMesh.create_coordinate_frame()
f = open("/home/arpg/Documents/habitat-lab/out_training_data/sample_octomap_running.txt", "r")


points = np.loadtxt("/home/arpg/Documents/open3d_from_habitat/training_pointclouds/local_pc_" + str(110) + ".txt")
#load the pose:
curr_pose = np.loadtxt("/home/arpg/Documents/habitat-lab/out_training_data/pose_" + str(111) + ".txt")
#load the rotation
curr_rot = np.loadtxt("/home/arpg/Documents/habitat-lab/out_training_data/rot_" + str(111) + ".txt")
rotation_val = Rotation.from_rotvec(curr_rot)
pose_mat = homogeneous_transform(curr_pose,rotation_val.as_quat())
coor = coor.transform(pose_mat)

pcd_local = o3d.geometry.PointCloud()
pcd_local.points = o3d.utility.Vector3dVector(points)
colors = np.zeros((len(points), 3))
colors[:,0] = colors[:,0] + 1
pcd_local.colors = o3d.utility.Vector3dVector(colors)


#run data through diffusion:
# load the models
model = UNet2DConditionModel.from_pretrained("alre5639/rgbd_unet_512")
conditioning_model = get_model()
conditioning_model.load_state_dict(torch.load("/home/arpg/Documents/SceneDiffusion/rgbd_512_cond_weights/cond_model" + str(249)))


########################
#get gt data
#########################3
gt_dir = "data/gt_pointmaps/"
gt_files = natsorted(os.listdir(gt_dir))

# ##################################
# #get the conditioning data
# ##################################
cond_dir = "data/rgbd_voxelized_data/"
cond_files = natsorted(os.listdir(cond_dir))
i = 110
print(gt_files[i]," | ", cond_files[i])

#get conditioning data:
# cond_data = np.loadtxt(cond_dir + cond_files[i])
# #transpose data for pointnet
# cond_data = cond_data.T
# #add batch dim
# cond_data = cond_data[None,:,:]
# #pass through pointnet
# cond_tensor = conditioning_model(torch.tensor(cond_data.astype(np.single)))
# post_model_conditioning_batch = cond_tensor.swapaxes(1, 2)
# noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
# #run diffusion
# point_map = utils.denoise_guided_inference(model, noise_scheduler,post_model_conditioning_batch, 30)
# returned_pc = utils.pointmap_to_pc(point_map[0])
# predicted_pc = o3d.geometry.PointCloud()
# predicted_pc.points = o3d.utility.Vector3dVector(returned_pc)

#gt_pc 
gt_data_pointmap = np.load(gt_dir + gt_files[i])

gt_data = utils.pointmap_to_pc(gt_data_pointmap)
gt_pc = o3d.geometry.PointCloud()
gt_pc.points = o3d.utility.Vector3dVector(gt_data)
# gt_pc = gt_pc.transform(pose_mat)

#get the local data from the octomap
running_loc_points = points_within_distance(curr_pose[0], curr_pose[2], np.asarray(pcd.points), 1.5)
#set vertical bounds
running_loc_points = running_loc_points[running_loc_points[:,1] > -1.4 + 1.4]
#remove the celing
running_loc_points = running_loc_points[running_loc_points[:,1] < 0.9 + 1.3]

local_110_points = o3d.geometry.PointCloud()
local_110_points.points = o3d.utility.Vector3dVector(running_loc_points)
#need to generate a pointmap from the current pose of the platform
local_110_points = local_110_points.transform(utils.inverse_homogeneous_transform(pose_mat))
# transform the local_points into a pointmapo
print(np.asarray(local_110_points.points).shape)
pointmap_110_local = utils.pc_to_pointmap(np.asarray(local_110_points.points), voxel_size = 0.1, x_y_bounds = [-1.5, 1.5], z_bounds = [-1.4, 0.9])
#unpointmap it to check
# returned_pc = utils.pointmap_to_pc(pointmap)
# predicted_pc = o3d.geometry.PointCloud()
# predicted_pc.points = o3d.utility.Vector3dVector(returned_pc)
# print(pointmap_110_local.shape)
# print(gt_data_pointmap[0].shape)
print(utils.get_IoU(pointmap_110_local,gt_data_pointmap))
####################################
#diffusion inpainting:
####################################
cond_data = np.loadtxt(cond_dir + cond_files[i])
#transpose data for pointnet
cond_data = cond_data.T
#add batch dim
cond_data = cond_data[None,:,:]
#pass through pointnet
cond_tensor = conditioning_model(torch.tensor(cond_data.astype(np.single)))
post_model_conditioning_batch = cond_tensor.swapaxes(1, 2)
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
#run diffusion
denoised_inpainted_prediction = utils.inpainting_pointmaps(model = model,
                                                            noise_scheduler = noise_scheduler,
                                                            conditioning = post_model_conditioning_batch,
                                                            width = 30,
                                                            inpainting_target = pointmap_110_local)

print(utils.get_IoU(denoised_inpainted_prediction[0],gt_data_pointmap))
returned_pc = utils.pointmap_to_pc(denoised_inpainted_prediction[0])
predicted_pc = o3d.geometry.PointCloud()
predicted_pc.points = o3d.utility.Vector3dVector(returned_pc)

o3d.visualization.draw_geometries([predicted_pc])