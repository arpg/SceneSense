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
from spconv.pytorch.utils import PointToVoxel
import cv2
from cleanfid import fid

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
# model = UNet2DConditionModel.from_pretrained("alre5639/full_rgbd_unet_512_more_pointnet_short", revision = "1f1755c2e9947f51c16a156d34f9a3e58d02bd4a")
# # model = UNet2DConditionModel.from_pretrained("alre5639/diff_unet")
# conditioning_model = get_model()
# conditioning_model.load_state_dict(torch.load("/home/arpg/Documents/SceneDiffusion/data/full_sim_pointnet_weights_more_pointnet_short/145"))
# conditioning_model.load_state_dict(torch.load("/home/arpg/Documents/SceneDiffusion/conditioning_model_weights/cond_model" + str(217)))
model = UNet2DConditionModel.from_pretrained("alre5639/full_rgbd_unet_512_more_pointnet", revision = "b063adc01ea748b7a4dbfb7e180eedf741aef536")
conditioning_model = get_model()
conditioning_model.load_state_dict(torch.load("/hdd/sceneSense_data/data/full_sim_pointnet_weights_more_pointnet/171"))

#get the running octomap
pcd_file_path = '/home/arpg/Documents/habitat-lab/running_octomap/running_occ.pcd'
pcd = o3d.io.read_point_cloud(pcd_file_path)
colors = np.zeros((len(np.asarray(pcd.points)), 3))
# pcd.colors = o3d.utility.Vector3dVector(colors)
#get the gt prediction
gt_file_path = '/home/arpg/Documents/habitat-lab/running_octomap/gt_occ_point.pcd'
gt_pcd = o3d.io.read_point_cloud(gt_file_path)
#load just the points at the current pose
gt_points = np.asarray(gt_pcd.points)
#get the current pose of the robot
curr_coor = np.loadtxt("/home/arpg/Documents/habitat-lab/running_octomap/curr_pose.txt")
curr_rot= np.loadtxt("/home/arpg/Documents/habitat-lab/running_octomap/curr_heading.txt")

rotation_obj = Rotation.from_rotvec(curr_rot)
hm_tx_mat = utils.homogeneous_transform(curr_coor, rotation_obj.as_quat())

gt_points_shift = copy.deepcopy(gt_pcd).transform(utils.inverse_homogeneous_transform(hm_tx_mat))
#get the local data
local_gt_points = points_within_distance(0.0,0.0,np.asarray(gt_points_shift.points),2.0)
#remove the lower floors
local_gt_points = local_gt_points[local_gt_points[:,1] > -1.4]
#remove the celing
local_gt_points = local_gt_points[local_gt_points[:,1] < 0.9]

local_pcd = o3d.geometry.PointCloud()
local_pcd.points = o3d.utility.Vector3dVector(local_gt_points)
# #add color to the points 
# # colors = np.zeros((len(np.asarray(local_pcd.points)), 3))
# # # colors[:,0] = colors[:,1] + 1
# # local_pcd.colors = o3d.utility.Vector3dVector(colors)
R = pcd.get_rotation_matrix_from_xyz((np.pi/2, 0, 0))
# # pcd.rotate(R, center=(0, 0, 0))
coor = o3d.geometry.TriangleMesh.create_coordinate_frame()
# # coor.rotate(R, center=(0, 0, 0))
# # # plot as voxels
# # # pcd_vox = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.1)
# o3d.visualization.draw_geometries([local_pcd, coor])
# #okay so we are going to need to take slices from the top, the color part is weird so maybe we just normalize to 255

local_octomap_pm = utils.pc_to_pointmap(np.asarray(local_pcd.points), 
                                        voxel_size = 0.1,
                                         x_y_bounds = [-2.0, 2.0],
                                          z_bounds = [-1.4, 0.9])
returned_pc = utils.pointmap_to_pc(pointmap = local_octomap_pm,
                                         voxel_size = 0.1,
                                         x_y_bounds = [-2, 2],
                                          z_bounds = [-1.4, 0.9])
print(returned_pc.shape)

reconstructed_pcd = o3d.geometry.PointCloud()
reconstructed_pcd.points = o3d.utility.Vector3dVector(returned_pc)
colors = np.zeros((len(np.asarray(reconstructed_pcd.points)), 3))
reconstructed_pcd.colors = o3d.utility.Vector3dVector(colors)

# o3d.visualization.draw_geometries([reconstructed_pcd, coor])
# print(local_octomap_pm.shape)

#now that is a pointmap slice it into individual pngs
for i, img in enumerate(local_octomap_pm):
    #normalize the outputs to 255 in each pixel
    output = copy.deepcopy(img) * 255
    #dupicate it to be an image
    output = np.repeat(output[:, :, np.newaxis], 3, axis=2)
    #save it as a cv2 image
    cv2.imwrite("fid_data/gt/" + str(i) + ".png", output )

#need to shift this I think
current_map_shift = copy.deepcopy(pcd).transform(utils.inverse_homogeneous_transform(hm_tx_mat))
local_octomap_points = points_within_distance(0,
                        0,
                        np.asarray(current_map_shift.points),
                        2.0)
# local_octomap_points = local_octomap_points[local_octomap_points[:,1] < 0.8]
local_octomap_points = local_octomap_points[local_octomap_points[:,1] > -1.4]
#remove the celing
local_octomap_points = local_octomap_points[local_octomap_points[:,1] < 0.9]

local_octomap_pcd = o3d.geometry.PointCloud()
local_octomap_pcd.points = o3d.utility.Vector3dVector(local_octomap_points)

# local_octomap_pcd.rotate(R, center=(0, 0, 0))
# o3d.visualization.draw_geometries([local_octomap_pcd, coor])
local_octomap_pm = utils.pc_to_pointmap(np.asarray(local_octomap_points), 
                                        voxel_size = 0.1,
                                         x_y_bounds = [-2.0, 2.0],
                                          z_bounds = [-1.4, 0.9])
# #now that is a pointmap slice it into individual pngs
for i, img in enumerate(local_octomap_pm):
    #normalize the outputs to 255 in each pixel
    output = copy.deepcopy(img) * 255
    #dupicate it to be an image
    output = np.repeat(output[:, :, np.newaxis], 3, axis=2)
    #save it as a cv2 image
    cv2.imwrite("fid_data/predicted/" + str(i) + ".png", output )

# #compute the fid
fid_val = fid.compute_fid("fid_data/predicted/", "fid_data/gt/")
print(fid_val)