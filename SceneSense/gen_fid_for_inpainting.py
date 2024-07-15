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
pcd.colors = o3d.utility.Vector3dVector(colors)
#get the gt prediction
gt_file_path = '/home/arpg/Documents/habitat-lab/running_octomap/gt_occ_point.pcd'
gt_pcd = o3d.io.read_point_cloud(gt_file_path)
#load just the points at the current pose
gt_points = np.asarray(gt_pcd.points)
#get the current pose of the robot
curr_coor = np.loadtxt("/home/arpg/Documents/habitat-lab/running_octomap/curr_pose.txt")
curr_rot= np.loadtxt("/home/arpg/Documents/habitat-lab/running_octomap/curr_heading.txt")
local_gt_points = points_within_distance(curr_coor[0],curr_coor[2],gt_points,2.0)
#remove the lower floors
local_gt_points = local_gt_points[local_gt_points[:,1] > -1.4]
#remove the celing
local_gt_points = local_gt_points[local_gt_points[:,1] < 0.9]
local_pcd = o3d.geometry.PointCloud()
local_pcd.points = o3d.utility.Vector3dVector(local_gt_points)
#add color to the points 
colors = np.zeros((len(np.asarray(local_pcd.points)), 3))
# colors[:,0] = colors[:,1] + 1
local_pcd.colors = o3d.utility.Vector3dVector(colors)

####################################
#now do the diffusion
########################################3
#get the local conditioning
#load the training folders
training_dirs = "/home/arpg/Documents/habitat-lab/running_octomap/"
gen = PointToVoxel(vsize_xyz=[0.01, 0.01, 0.01],
                        coors_range_xyz=[-10, -10, -10, 10, 10, 10],
                        num_point_features=6,
                        max_num_voxels=65536,
                        max_num_points_per_voxel=1)
# o3d.visualization.draw_geometries([local_pcd, pcd])
image = cv2.imread(training_dirs + "rgb_.png")
#load the point data
pc = np.load(training_dirs + "sample_pcnpy.npy")
print(pc.shape)
colors = image.reshape(-1, image.shape[2])

#create the full rgbd pc
#this is also what we will pass through pointnet 
rgbd_pc = np.append(pc, colors, axis = 1)

conditioning_pcd = o3d.geometry.PointCloud()
conditioning_pcd.points = o3d.utility.Vector3dVector(rgbd_pc[:,0:3])
conditioning_pcd.colors = o3d.utility.Vector3dVector(rgbd_pc[:,3:6]/255)

#transfomr the point could to the robot frame
rotation_obj = Rotation.from_rotvec(curr_rot)
hm_tx_mat = utils.homogeneous_transform(curr_coor, rotation_obj.as_quat())
conditioning_pcd.transform(hm_tx_mat)

# This shows the idea visualization using gt
# o3d.visualization.draw_geometries([conditioning_pcd,local_pcd, pcd])



voxels_th, indices_th, num_p_in_vx_th = gen(torch.tensor(rgbd_pc), empty_mean = True)
voxels_np = voxels_th.numpy() 
conditioning_voxel_points = np.reshape(voxels_np, (-1,6))
print(conditioning_voxel_points.shape)
# add batch size
conditioning_voxel_points = conditioning_voxel_points.T
conditioning_voxel_points = conditioning_voxel_points[None, :,:]
#swap axis 1 and 2

conditioning_voxel_points = torch.tensor(conditioning_voxel_points.astype(np.single))

pointnet_conditioing = conditioning_model(conditioning_voxel_points)
pointnet_conditioing = pointnet_conditioing.swapaxes(1,2)
# ###########################
# # this is for the diffused map
# ######################################

# # final_pointcloud = torch.from_numpy(final_pointcloud)
# noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
# # #do the denoising:
# # post_model_conditioning_batch = conditioning_model(final_pointcloud)
# # post_model_conditioning_batch = post_model_conditioning_batch.swapaxes(1, 2)
# # # print(post_model_conditioning_batch[13].shape)
# # test_conditioning = post_model_conditioning_batch[13]
# # test_conditioning = test_conditioning[None,:,:]
# print(pointnet_conditioing.shape)
# point_map = utils.denoise_guided_inference(model, noise_scheduler,pointnet_conditioing, 40)
# point_map_np = point_map.numpy()
# print(point_map_np.shape)
# diffused_pointcloud =utils.pointmap_to_pc(point_map_np[0],
#                                          voxel_size = 0.1,
#                                          x_y_bounds = [-2, 2],
#                                           z_bounds = [-1.4, 0.9])

# pcd_diff = o3d.geometry.PointCloud()
# pcd_diff.points = o3d.utility.Vector3dVector(diffused_pointcloud)
# colors = np.zeros((len(np.asarray(pcd_diff.points)), 3))
# colors[:,0] = 0
# colors[:,1] = 0
# colors[:,2] = 0
# pcd_diff.colors = o3d.utility.Vector3dVector(colors)
# #transform to fit room 
# pcd_diff.transform(hm_tx_mat)
# # o3d.visualization.draw_geometries([pcd_diff,pcd])

###################################
# heres inpainting without freespace
################################################
coor = o3d.geometry.TriangleMesh.create_coordinate_frame()
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
#get the local pointmap
local_octomap_points = points_within_distance(curr_coor[0],
                        curr_coor[2],
                        np.asarray(pcd.points),
                        2.0)
#remove lower floor and ground
#remove the lower floors
local_octomap_points = local_octomap_points[local_octomap_points[:,1] > -1.4]
#remove the celing
local_octomap_points = local_octomap_points[local_octomap_points[:,1] < 2.0]
local_octomap_pcd = o3d.geometry.PointCloud()
local_octomap_pcd.points = o3d.utility.Vector3dVector(local_octomap_points)

# colors = np.zeros((len(np.asarray(local_octomap_pcd.points)), 3))
local_octomap_pcd.colors = o3d.utility.Vector3dVector(colors)
o3d.visualization.draw_geometries([pcd,local_octomap_pcd])
#transfomr the local points to 0,0
zeroed_local_pcd = copy.deepcopy(local_octomap_pcd)
zeroed_local_pcd.transform(utils.inverse_homogeneous_transform(hm_tx_mat))
#turn the points into a pointmap
local_octomap_pm = utils.pc_to_pointmap(np.asarray(zeroed_local_pcd.points), 
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
#transform back to location
#there is a shift I am messing up here or something
 
#do the inpainting
# inpained_pm = utils.inpainting_pointmaps(model,
#                                         noise_scheduler,
#                                         pointnet_conditioing,
#                                         40,
#                                         local_octomap_pm)
#convert to pointcloud
# inpained_points = utils.pointmap_to_pc(inpained_pm[0],
#                                          voxel_size = 0.1,
#                                          x_y_bounds = [-2, 2],
#                                           z_bounds = [-1.4, 0.9])
# pcd_inpaint = o3d.geometry.PointCloud()

# print("inpainted shape: ", inpained_points.shape)
# pcd_inpaint.points = o3d.utility.Vector3dVector(inpained_points)
# colors = np.zeros((len(np.asarray(pcd_inpaint.points)), 3))
# colors[:,0] = 1
# colors[:,1] = 0
# colors[:,2] = 0
# pcd_inpaint.colors = o3d.utility.Vector3dVector(colors)
# #transform to fit room 
# pcd_inpaint.transform(hm_tx_mat)
# voxel_grid_octo = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,
#                                                               voxel_size=0.1)
# voxel_grid_diff = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_inpaint,
#                                                               voxel_size=0.1)                                                              
# o3d.visualization.draw_geometries([voxel_grid_octo,voxel_grid_diff])

#do the inpainting w/ freespace
#load the freespace octomap

unoc_pcd_path = '/home/arpg/Documents/habitat-lab/test_unoc.pcd'
unoc_pcd = o3d.io.read_point_cloud(unoc_pcd_path)
unoc_pcd.transform(utils.inverse_homogeneous_transform(hm_tx_mat))
unoc_local_points = points_within_distance(0,0,np.asarray(unoc_pcd.points),2.0)

# #remove the lower floors
unoc_local_points = unoc_local_points[unoc_local_points[:,1] > -1.4]
# #remove the celing
unoc_local_points = unoc_local_points[unoc_local_points[:,1] < 0.9]

unoc_local_pcd = o3d.geometry.PointCloud()
unoc_local_pcd.points = o3d.utility.Vector3dVector(unoc_local_points) 

# o3d.visualization.draw_geometries([unoc_local_pcd, coor])

unoc_pm = utils.pc_to_pointmap(unoc_local_points, 
                                        voxel_size = 0.1,
                                         x_y_bounds = [-2.0, 2.0],
                                          z_bounds = [-1.4, 0.9])
returned_unoc_pc = utils.pointmap_to_pc(pointmap = unoc_pm,
                                         voxel_size = 0.1,
                                         x_y_bounds = [-2, 2],
                                          z_bounds = [-1.4, 0.9])
unoc_recon_pcd = o3d.geometry.PointCloud()
unoc_recon_pcd.points = o3d.utility.Vector3dVector(returned_unoc_pc)
colors = np.zeros((len(np.asarray(unoc_recon_pcd.points)), 3))
colors[:,0] = 1
unoc_recon_pcd.colors = o3d.utility.Vector3dVector(colors)

#do the freespace inpainting
# o3d.visualization.draw_geometries([unoc_recon_pcd, reconstructed_pcd])
inpained_pm = utils.inpainting_pointmaps_w_freespace(model,
                                        noise_scheduler,
                                        pointnet_conditioing,
                                        40,
                                        local_octomap_pm,
                                        unoc_pm)
inpained_points = utils.pointmap_to_pc(inpained_pm[0],
                                         voxel_size = 0.1,
                                         x_y_bounds = [-2, 2],
                                          z_bounds = [-1.4, 0.9])
pcd_inpaint = o3d.geometry.PointCloud()

# # print("inpainted shape: ", inpained_points.shape)
pcd_inpaint.points = o3d.utility.Vector3dVector(inpained_points)
colors = np.zeros((len(np.asarray(pcd_inpaint.points)), 3))
colors[:,0] = 1
colors[:,1] = 0
colors[:,2] = 0
pcd_inpaint.colors = o3d.utility.Vector3dVector(colors)
pcd_inpaint.transform(hm_tx_mat)
# o3d.visualization.draw_geometries([pcd_inpaint, pcd])
#transform pcd to origin
#there is an offset thing going on

# #I think the freespace inpainting might be wrong need to convicnce ourselves it is working
print(inpained_pm.shape)
for i, img in enumerate(np.asarray(inpained_pm[0])):
    #normalize the outputs to 255 in each pixel
    output = copy.deepcopy(img) * 255
    #dupicate it to be an image
    output = np.repeat(output[:, :, np.newaxis], 3, axis=2)
    print(output.shape)
    #save it as a cv2 image
    cv2.imwrite("fid_data/ss_predicted/" + str(i) + ".png", output )


#compute the fid
fid_val = fid.compute_fid("fid_data/ss_predicted/", "fid_data/gt/")
print(fid_val)

o3d.visualization.draw_geometries([pcd_inpaint, pcd])
