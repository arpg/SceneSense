import copy
import os
import random
import re
import shutil
import sys
from dataclasses import dataclass

import cv2
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
import utils.utils as utils
import wandb
from cleanfid import fid
from diffusers import DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from huggingface_hub import login
from natsort import natsorted
from pointnet2_scene_diffusion import get_model
from scipy.spatial.transform import Rotation
from spconv.pytorch.utils import PointToVoxel
from tqdm.auto import tqdm


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
    # this actually needs to be xz
    xy_coordinates = points[:, [0, 2]]

    # Calculate the Euclidean distance from the given location to all points
    distances = np.linalg.norm(xy_coordinates - np.array([x, y]), axis=1)

    # Find indices of points within the specified distance
    within_distance_indices = np.where(distances <= distance)[0]

    # Extract points within the distance
    points_within_distance = points[within_distance_indices]

    return points_within_distance


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
        raise ValueError(
            "Translation vector must be of length 3, and rotation quaternion must be of length 4."
        )

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
# model = UNet2DConditionModel.from_pretrained("alre5639/full_rgbd_unet_512_more_pointnet_short", revision = "1f1755c2e9947f51c16a156d34f9a3e58d02bd4a")
# # model = UNet2DConditionModel.from_pretrained("alre5639/diff_unet")
# conditioning_model = get_model()
# conditioning_model.load_state_dict(torch.load("/home/arpg/Documents/SceneDiffusion/data/full_sim_pointnet_weights_more_pointnet_short/145"))
# conditioning_model.load_state_dict(torch.load("/home/arpg/Documents/SceneDiffusion/conditioning_model_weights/cond_model" + str(217)))
model = UNet2DConditionModel.from_pretrained(
    "alre5639/full_rgbd_unet_512_more_pointnet",
    revision="b063adc01ea748b7a4dbfb7e180eedf741aef536",
)
conditioning_model = get_model()
conditioning_model.load_state_dict(
    torch.load("/hdd/sceneSense_data/data/full_sim_pointnet_weights_more_pointnet/171")
)

# loop through all the directories in house 1
house_path = "/hdd/sceneDiff_data/house_2/"
# get all the directories
house_dirs = natsorted(os.listdir(house_path))
# print(house_dirs)nning octomap
pcd = None
gt_pcd = None
print(sys.argv)
curr_dir = sys.argv[1]
diff_steps = sys.argv[2]
s_guide = sys.argv[3]
step = house_dirs[int(curr_dir)]

# load the gt data
gt_file_path = "/home/arpg/Documents/habitat-lab/house_2/occupancy_gt.pcd"
gt_pcd = o3d.io.read_point_cloud(gt_file_path)
# load the occupided data
pcd_file_path = house_path + step + "/running_octomap/running_occ.pcd"
running_occ_pcd = o3d.io.read_point_cloud(pcd_file_path)
# load the pose data
curr_coor = np.loadtxt(house_path + step + "/running_octomap/curr_pose.txt")
curr_rot = np.loadtxt(house_path + step + "/running_octomap/curr_heading.txt")
rotation_obj = Rotation.from_rotvec(curr_rot)
hm_tx_mat = utils.homogeneous_transform(curr_coor, rotation_obj.as_quat())

####################################
# Get the Conditioning
########################################3
# get the local conditioning
# load the training folders
training_dirs = house_path + step + "/running_octomap/"
gen = PointToVoxel(
    vsize_xyz=[0.01, 0.01, 0.01],
    coors_range_xyz=[-10, -10, -10, 10, 10, 10],
    num_point_features=6,
    max_num_voxels=65536,
    max_num_points_per_voxel=1,
)
# o3d.visualization.draw_geometries([local_pcd, pcd])
image = cv2.imread(training_dirs + "rgb_.png")
# load the point data
pc = np.load(training_dirs + "sample_pc.npy")
print(pc.shape)
colors = image.reshape(-1, image.shape[2])

# create the full rgbd pc
# this is also what we will pass through pointnet
rgbd_pc = np.append(pc, colors, axis=1)

conditioning_pcd = o3d.geometry.PointCloud()
conditioning_pcd.points = o3d.utility.Vector3dVector(rgbd_pc[:, 0:3])
conditioning_pcd.colors = o3d.utility.Vector3dVector(rgbd_pc[:, 3:6] / 255)

# transfomr the point could to the robot frame
rotation_obj = Rotation.from_rotvec(curr_rot)
hm_tx_mat = utils.homogeneous_transform(curr_coor, rotation_obj.as_quat())
conditioning_pcd.transform(hm_tx_mat)

# This shows the idea visualization using gt
# o3d.visualization.draw_geometries([conditioning_pcd,local_pcd, pcd])


voxels_th, indices_th, num_p_in_vx_th = gen(torch.tensor(rgbd_pc), empty_mean=True)
voxels_np = voxels_th.numpy()
conditioning_voxel_points = np.reshape(voxels_np, (-1, 6))
print(conditioning_voxel_points.shape)
# add batch size
conditioning_voxel_points = conditioning_voxel_points.T
conditioning_voxel_points = conditioning_voxel_points[None, :, :]
# swap axis 1 and 2

conditioning_voxel_points = torch.tensor(conditioning_voxel_points.astype(np.single))

pointnet_conditioing = conditioning_model(conditioning_voxel_points)
pointnet_conditioing = pointnet_conditioing.swapaxes(1, 2)

###################################
# Generate Local Occupied PC info
################################################
# first shift the local map into the correct frame
running_occ_pcd_shift = copy.deepcopy(running_occ_pcd).transform(
    utils.inverse_homogeneous_transform(hm_tx_mat)
)
local_occ_points = points_within_distance(
    0.0, 0.0, np.asarray(running_occ_pcd_shift.points), 2.0
)
# remove the lower floors
local_occ_points = local_occ_points[local_occ_points[:, 1] > -1.5]
# #remove the celing
local_occ_points = local_occ_points[local_occ_points[:, 1] < 0.8]
# convert it into a pointmap
local_octomap_pm = utils.pc_to_pointmap(
    local_occ_points, voxel_size=0.1, x_y_bounds=[-2.0, 2.0], z_bounds=[-1.5, 0.8]
)
returned_pc = utils.pointmap_to_pc(
    pointmap=local_octomap_pm, voxel_size=0.1, x_y_bounds=[-2, 2], z_bounds=[-1.5, 0.8]
)
reconstructed_pcd = o3d.geometry.PointCloud()
reconstructed_pcd.points = o3d.utility.Vector3dVector(returned_pc)
# colors = np.zeros((len(np.asarray(reconstructed_pcd.points)), 3))
# reconstructed_pcd.colors = o3d.utility.Vector3dVector(colors)
# local_pcd = o3d.geometry.PointCloud()
# local_pcd.points = o3d.utility.Vector3dVector(local_occ_points)
# o3d.visualization.draw_geometries([reconstructed_pcd])

###################################
# Generate Local Unocc PC info
################################################
unoc_pcd_path = house_path + step + "/running_octomap/unoc.pcd"
unoc_pcd = o3d.io.read_point_cloud(unoc_pcd_path)
unoc_pcd.transform(utils.inverse_homogeneous_transform(hm_tx_mat))
unoc_local_points = points_within_distance(0, 0, np.asarray(unoc_pcd.points), 2.0)

# #remove the lower floors
unoc_local_points = unoc_local_points[unoc_local_points[:, 1] > -1.5]
# #remove the celing
unoc_local_points = unoc_local_points[unoc_local_points[:, 1] < 0.8]

unoc_pm = utils.pc_to_pointmap(
    unoc_local_points, voxel_size=0.1, x_y_bounds=[-2.0, 2.0], z_bounds=[-1.5, 0.8]
)
returned_unoc_pc = utils.pointmap_to_pc(
    pointmap=unoc_pm, voxel_size=0.1, x_y_bounds=[-2, 2], z_bounds=[-1.5, 0.8]
)
unoc_recon_pcd = o3d.geometry.PointCloud()
unoc_recon_pcd.points = o3d.utility.Vector3dVector(returned_unoc_pc)
colors = np.zeros((len(np.asarray(unoc_recon_pcd.points)), 3))
colors[:, 0] = 1
unoc_recon_pcd.colors = o3d.utility.Vector3dVector(colors)

# #do the freespace inpainting
# o3d.visualization.draw_geometries([reconstructed_pcd, unoc_recon_pcd])
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

inpained_pm = utils.inpainting_pointmaps_w_freespace(
    model,
    noise_scheduler,
    pointnet_conditioing,
    40,
    local_octomap_pm,
    unoc_pm,
    torch_device="cpu",
    denoising_steps=int(diff_steps),
    guidance_scale=int(s_guide),
    sample_batch_size=1,
)


inpained_points = utils.pointmap_to_pc(
    inpained_pm[0], voxel_size=0.1, x_y_bounds=[-2, 2], z_bounds=[-1.5, 0.8]
)
pcd_inpaint = o3d.geometry.PointCloud()
pcd_inpaint.points = o3d.utility.Vector3dVector(inpained_points)

o3d.visualization.draw_geometries([pcd_inpaint])

o3d.io.write_point_cloud(
    "/hdd/sceneDiff_data/figure_data/" "pointcloud_" + str(2) + ".pcd", pcd_inpaint
)
# #I think the freespace inpainting might be wrong need to convicnce ourselves it is working
# print(inpained_pm.shape)
