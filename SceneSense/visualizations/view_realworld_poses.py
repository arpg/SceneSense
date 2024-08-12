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
from natsort import natsorted
import shutil
from io import StringIO

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

def string_to_floats(input_str):
    # Replace '][' with a space
    formatted_str = input_str.replace('] [', ' ')
    # Remove the leading and trailing brackets if present
    formatted_str = formatted_str.strip('[]')
    # Convert the string to a NumPy array of floats
    float_array = np.fromstring(formatted_str, sep=' ')
    # Return the array
    return float_array
def remove_distant_points(flipped_cond_points):
    # Calculate the Euclidean distance from the origin for each point
    distances = np.sqrt(np.sum(flipped_cond_points**2, axis=1))
    
    # Filter points where the distance is less than or equal to 5
    close_points = flipped_cond_points[distances <= 5]
    
    return close_points

torch_device = "cpu"
# model = UNet2DConditionModel.from_pretrained("alre5639/full_rgbd_unet_512_more_pointnet_short", revision = "1f1755c2e9947f51c16a156d34f9a3e58d02bd4a")
# # model = UNet2DConditionModel.from_pretrained("alre5639/diff_unet")
# conditioning_model = get_model()
# conditioning_model.load_state_dict(torch.load("/home/arpg/Documents/SceneDiffusion/data/full_sim_pointnet_weights_more_pointnet_short/145"))
# conditioning_model.load_state_dict(torch.load("/home/arpg/Documents/SceneDiffusion/conditioning_model_weights/cond_model" + str(217)))
model = UNet2DConditionModel.from_pretrained("alre5639/full_rgbd_unet_512_more_pointnet", revision = "b063adc01ea748b7a4dbfb7e180eedf741aef536")
conditioning_model = get_model()
conditioning_model.load_state_dict(torch.load("/hdd/sceneSense_data/data/full_sim_pointnet_weights_more_pointnet/171"))


gt_file_path =  '/hdd/sceneDiff_data/real_data/pointmaps/step_686/running_occ.pcd'
gt_pcd = o3d.io.read_point_cloud(gt_file_path)
# gt_file_path_sim =  '/home/arpg/Documents/habitat-lab/running_octomap/gt_occ_point.pcd'
# gt_pcd_sim = o3d.io.read_point_cloud(gt_file_path_sim)

#flip points 
#FLIP TO CORRECT ORRIENTATION
flipped_pcd = copy.deepcopy(gt_pcd)
flipped_points = np.asarray(flipped_pcd.points)
flipped_points[:, [1, 2]] = flipped_points[:, [2, 1]]
flipped_points[:,0] = -flipped_points[:,0]
flipped_pcd = o3d.geometry.PointCloud()
flipped_pcd.points = o3d.utility.Vector3dVector(flipped_points)
# gt_points[:, [1, 2]] = gt_points[:, [2, 1]]
# gt_points[:,0] = -gt_points[:,0]
# gt_pcd = o3d.geometry.PointCloud()
# gt_pcd.points = o3d.utility.Vector3dVector(gt_points)
# unoc_gt_path = '/hdd/sceneDiff_data/real_data/unocc_pcd.pcd'
# unoc_gt = o3d.io.read_point_cloud(unoc_gt_path)

coor = o3d.geometry.TriangleMesh.create_coordinate_frame()
# o3d.visualization.draw_geometries([gt_pcd, coor])

#get poses
poses_path = "/hdd/sceneDiff_data/real_data/full_running_octomaps/poses/"
#get all the directories
pose_dirs = natsorted(os.listdir(poses_path))
# loop through all the poses and get the data
hm_tx_poses =  arr = np.empty((0,4,4), float)
for file in pose_dirs[395:475]:
    s = open(poses_path + file).read()
    curr_pose = string_to_floats(s)
    curr_pose = np.reshape(curr_pose, (4,4))
    # correct_pose = curr_pose
    # correct_pose[[1,2], 3]  = correct_pose[[2,1], 3]  
    # correct_pose[0,3] = -correct_pose[0,3]
    hm_tx_poses = np.append(hm_tx_poses, curr_pose[None,:,:], axis = 0)

# positions_arr =  arr = np.empty((0,3), float)
positions_arr = hm_tx_poses[:,0:3,3]
# positions_arr[:, [1, 2]] = positions_arr[:, [2, 1]]
# positions_arr[:,0] = -positions_arr[:,0]
pose_pcd = o3d.geometry.PointCloud()
pose_pcd.points = o3d.utility.Vector3dVector(positions_arr)
colors = np.zeros((len(np.asarray(pose_pcd.points)), 3))
pose_pcd.colors = o3d.utility.Vector3dVector(colors)
o3d.visualization.draw_geometries([pose_pcd, gt_pcd])