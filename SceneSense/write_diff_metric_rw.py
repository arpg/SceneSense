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
import re
def get_trailing_number(s):
    m = re.search(r'\d+$', s)
    return int(m.group()) if m else None
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
def remove_distant_points(flipped_cond_points, dist):
    # Calculate the Euclidean distance from the origin for each point
    distances = np.sqrt(np.sum(flipped_cond_points**2, axis=1))
    
    # Filter points where the distance is less than or equal to 5
    close_points = flipped_cond_points[distances <= dist]
    idx = np.argwhere(distances <=dist)
    return close_points, idx

torch_device = "cpu"
# model = UNet2DConditionModel.from_pretrained("alre5639/full_rgbd_unet_512_more_pointnet_short", revision = "1f1755c2e9947f51c16a156d34f9a3e58d02bd4a")
# # model = UNet2DConditionModel.from_pretrained("alre5639/diff_unet")
# conditioning_model = get_model()
# conditioning_model.load_state_dict(torch.load("/home/arpg/Documents/SceneDiffusion/data/full_sim_pointnet_weights_more_pointnet_short/145"))
# conditioning_model.load_state_dict(torch.load("/home/arpg/Documents/SceneDiffusion/conditioning_model_weights/cond_model" + str(217)))
model = UNet2DConditionModel.from_pretrained("alre5639/full_rgbd_unet_512_more_pointnet", revision = "b063adc01ea748b7a4dbfb7e180eedf741aef536")
conditioning_model = get_model()
conditioning_model.load_state_dict(torch.load("/hdd/sceneSense_data/data/full_sim_pointnet_weights_more_pointnet/171"))


gt_file_path = '/hdd/sceneDiff_data/real_data/pointmaps/step_686/running_occ.pcd'
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

gen = PointToVoxel(vsize_xyz=[0.01, 0.01, 0.01],
                        coors_range_xyz=[-10, -10, -10, 10, 10, 10],
                        num_point_features=6,
                        max_num_voxels=65536,
                        max_num_points_per_voxel=1)
for file in pose_dirs:
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
# o3d.visualization.draw_geometries([pose_pcd, gt_pcd])

#load the conditioning pointcloud
#get poses
cond_path = "/hdd/sceneDiff_data/real_data/full_running_octomaps/input_pcs/"
#get all the directories
cond_files = natsorted(os.listdir(cond_path))
# coor_idx = 0

#load the running occupancy info
running_occ_path = "/hdd/sceneDiff_data/real_data/pointmaps/"
running_occ_files = natsorted(os.listdir(running_occ_path))

# def key_callback(vis):
#     global coor_idx
#     try: coor_idx
#     except NameError: coor_idx = 0
#     print(coor_idx)

#     ###############################
#     # load and move gt data
#     ##################################
#     flipped_pcd = copy.deepcopy(gt_pcd)
#     flipped_pcd.transform(utils.inverse_homogeneous_transform(hm_tx_poses[coor_idx * 15, :,:]))
#     flipped_points = np.asarray(flipped_pcd.points)

#     flipped_points[:, [0, 2]] = flipped_points[:, [2, 0]]
#     flipped_points[:,0] = -flipped_points[:,0]
#     flipped_points[:,1] = -flipped_points[:,1]
#     flipped_points[:,2] = -flipped_points[:,2]

#     #get local flipped points
#     flipped_points = points_within_distance(0.0,0.0,flipped_points, 2.0)

#     #remove lower flooersfloor
#     flipped_points = flipped_points[flipped_points[:,1] > -1.8]
#     #remove the celing
#     flipped_points = flipped_points[flipped_points[:,1] < 0.8]

#     flipped_pcd = o3d.geometry.PointCloud()
#     flipped_pcd.points = o3d.utility.Vector3dVector(flipped_points)

#     ##############################
#     #get the local occ info:
#     ############################
#     running_occ_pcd = o3d.io.read_point_cloud(running_occ_path + running_occ_files[coor_idx] +"/running_occ.pcd" )
#     running_occ_pcd.transform(utils.inverse_homogeneous_transform(hm_tx_poses[coor_idx, :,:]))
#     flipped_running_occ_points = np.asarray(running_occ_pcd.points)

#     flipped_running_occ_points[:, [0, 2]] = flipped_running_occ_points[:, [2, 0]]
#     flipped_running_occ_points[:,0] = -flipped_running_occ_points[:,0]
#     flipped_running_occ_points[:,1] = -flipped_running_occ_points[:,1]
#     flipped_running_occ_points[:,2] = -flipped_running_occ_points[:,2]

#     #get local flipped points
#     local_running_occ_points = points_within_distance(0.0,0.0,flipped_running_occ_points, 2.0)
#     #remove lower flooersfloor
#     local_running_occ_points = local_running_occ_points[local_running_occ_points[:,1] > -1.8]
#     #remove the celing
#     local_running_occ_points = local_running_occ_points[local_running_occ_points[:,1] < 0.8]
#     local_running_occ_pcd = o3d.geometry.PointCloud()
#     local_running_occ_pcd.points = o3d.utility.Vector3dVector(local_running_occ_points)


#     #####################################
#     #load the approate cond filke
#     ############################################
#     cond_pcd = o3d.io.read_point_cloud(cond_path + cond_files[coor_idx])
#     # cond_pcd.transform(hm_tx_poses[coor_idx, :,:])
#     flipped_cond_pcd = copy.deepcopy(cond_pcd)
#     flipped_cond_points = np.asarray(flipped_cond_pcd.points)
    
#     flipped_cond_points[:, [0, 2]] = flipped_cond_points[:, [2, 0]]
#     flipped_cond_points[:,0] = -flipped_cond_points[:,0]
#     flipped_cond_points[:,1] = -flipped_cond_points[:,1]
#     flipped_cond_points[:,2] = -flipped_cond_points[:,2]
#     #remove far points
#     # flipped_cond_points = remove_distant_points(flipped_cond_points)

#     flipped_cond_pcd = o3d.geometry.PointCloud()
#     flipped_cond_pcd.points = o3d.utility.Vector3dVector(flipped_cond_points)
#     flipped_cond_pcd.colors = cond_pcd.colors
#     vis.clear_geometries()

#     vis.add_geometry(flipped_cond_pcd)
#     # vis.add_geometry(gt_pcd)
#     # vis.add_geometry(flipped_pcd)
#     vis.add_geometry(local_running_occ_pcd)
#     vis.add_geometry(coor)

#     coor_idx += 1

# # ctr  = self.o3d_visualizer.get_view_control()
# # view_param =ctr.convert_to_pinhole_camera_parameters()

# vis = o3d.visualization.VisualizerWithKeyCallback()
# vis.create_window()
# vis.add_geometry(gt_pcd)
# vis.add_geometry(coor)
# vis.add_geometry(pose_pcd)
# vis.register_key_callback(65, key_callback)
# # vis.register_animation_callback(key_callback)
# # Set the timer interval (in milliseconds)
# # vis.get_timer().set_interval(1000)

# vis.run()
# vis.destroy_window()

    


for coor_idx in list(range(len(running_occ_files)))[440:450]:
    # if not(os.path.isdir(running_occ_path + running_occ_files[coor_idx] + "/fid_data/")):
    #     os.mkdir(running_occ_path + running_occ_files[coor_idx] + "/fid_data/")
    # if not(os.path.isdir(running_occ_path + running_occ_files[coor_idx] + "/fid_data/gt")):
    #     os.mkdir(running_occ_path + running_occ_files[coor_idx] + "/fid_data/gt")
    # if not(os.path.isdir(running_occ_path + running_occ_files[coor_idx] + "/fid_data/predicted")):
    #     os.mkdir(running_occ_path + running_occ_files[coor_idx] + "/fid_data/predicted")
    # if not(os.path.isdir(running_occ_path + running_occ_files[coor_idx] + "/fid_data/ss_predicted")):    
    #     os.mkdir(running_occ_path + running_occ_files[coor_idx] + "/fid_data/ss_predicted")
    #get the correct index based on the file name

    
    correct_nums = get_trailing_number(running_occ_files[coor_idx])
    # check if folder is empty 
    if len(os.listdir(running_occ_path + running_occ_files[coor_idx])) != 0:
        print("running occ file: ", running_occ_files[coor_idx])
        print("pose: ", correct_nums)
        print("conditioning_file: ", cond_files[correct_nums])
        ###############################
        # load and move gt data
        ##################################
        flipped_pcd = copy.deepcopy(gt_pcd)
        flipped_pcd.transform(utils.inverse_homogeneous_transform(hm_tx_poses[correct_nums, :,:]))
        flipped_points = np.asarray(flipped_pcd.points)

        flipped_points[:, [0, 2]] = flipped_points[:, [2, 0]]
        flipped_points[:,0] = -flipped_points[:,0]
        flipped_points[:,1] = -flipped_points[:,1]
        flipped_points[:,2] = -flipped_points[:,2]

        #get local flipped points
        flipped_points = points_within_distance(0.0,0.0,flipped_points, 2.0)

        #remove lower flooersfloor
        flipped_points = flipped_points[flipped_points[:,1] > -1.5]
        #remove the celing
        flipped_points = flipped_points[flipped_points[:,1] < 0.8]

        flipped_pcd = o3d.geometry.PointCloud()
        flipped_pcd.points = o3d.utility.Vector3dVector(flipped_points)

        ##############################
        #get the local occ info:
        ############################
        running_occ_pcd = o3d.io.read_point_cloud(running_occ_path + running_occ_files[coor_idx] +"/running_occ.pcd" )
        running_occ_pcd.transform(utils.inverse_homogeneous_transform(hm_tx_poses[correct_nums, :,:]))
        flipped_running_occ_points = np.asarray(running_occ_pcd.points)

        flipped_running_occ_points[:, [0, 2]] = flipped_running_occ_points[:, [2, 0]]
        flipped_running_occ_points[:,0] = -flipped_running_occ_points[:,0]
        flipped_running_occ_points[:,1] = -flipped_running_occ_points[:,1]
        flipped_running_occ_points[:,2] = -flipped_running_occ_points[:,2]

        #get local flipped points
        local_running_occ_points = points_within_distance(0.0,0.0,flipped_running_occ_points, 2.0)
        #remove lower flooersfloor
        local_running_occ_points = local_running_occ_points[local_running_occ_points[:,1] > -1.5]
        #remove the celing
        local_running_occ_points = local_running_occ_points[local_running_occ_points[:,1] < 0.8]
        local_running_occ_pcd = o3d.geometry.PointCloud()
        local_running_occ_pcd.points = o3d.utility.Vector3dVector(local_running_occ_points)
        colors = np.zeros((len(np.asarray(local_running_occ_pcd.points)), 3))
        # colors[:,0] = colors[:,0]
        local_running_occ_pcd.colors = o3d.utility.Vector3dVector(colors)

        ##############################3
        # get the local unocc info
        #####################################
        running_unoc_pcd = o3d.io.read_point_cloud(running_occ_path + running_occ_files[coor_idx] +"/running_unoc.pcd" )
        running_unoc_pcd.transform(utils.inverse_homogeneous_transform(hm_tx_poses[correct_nums, :,:]))
        flipped_running_unoc_points = np.asarray(running_unoc_pcd.points)

        flipped_running_unoc_points[:, [0, 2]] = flipped_running_unoc_points[:, [2, 0]]
        flipped_running_unoc_points[:,0] = -flipped_running_unoc_points[:,0]
        flipped_running_unoc_points[:,1] = -flipped_running_unoc_points[:,1]
        flipped_running_unoc_points[:,2] = -flipped_running_unoc_points[:,2]

        # #get local flipped points
        local_running_unoc_points = points_within_distance(0.0,0.0,flipped_running_unoc_points, 2.0)
        #remove lower flooersfloor
        local_running_unoc_points = local_running_unoc_points[local_running_unoc_points[:,1] > -1.5]
        #remove the celing
        local_running_unoc_points = local_running_unoc_points[local_running_unoc_points[:,1] < 0.8]
        local_running_unoc_pcd = o3d.geometry.PointCloud()
        local_running_unoc_pcd.points = o3d.utility.Vector3dVector(local_running_unoc_points)

        # o3d.visualization.draw_geometries([running_unoc_pcd])


        ############################################3
        #load the approate cond filke
        ############################################
        cond_pcd = o3d.io.read_point_cloud(cond_path + cond_files[correct_nums])
        # o3d.visualization.draw_geometries([cond_pcd])

        # cond_pcd.transform(hm_tx_poses[coor_idx, :,:])
        flipped_cond_pcd = copy.deepcopy(cond_pcd)
        flipped_cond_points = np.asarray(flipped_cond_pcd.points)
        
        flipped_cond_points[:, [0, 2]] = flipped_cond_points[:, [2, 0]]
        flipped_cond_points[:,0] = -flipped_cond_points[:,0]
        flipped_cond_points[:,1] = -flipped_cond_points[:,1]
        flipped_cond_points[:,2] = -flipped_cond_points[:,2]
        print(flipped_cond_points.shape)
        #remove far points
        flipped_cond_points, remaining_idx = remove_distant_points(flipped_cond_points, dist = 10)

        print(flipped_cond_points.shape)
        print(remaining_idx.shape)
        
        ds_colors = np.asarray(copy.deepcopy(cond_pcd.colors))
        ds_colors = ds_colors[remaining_idx[:,0]]
        print(flipped_cond_points.shape)
        print(ds_colors.shape)
        # flipped_cond_pcd = o3d.geometry.PointCloud()
        # flipped_cond_pcd.points = o3d.utility.Vector3dVector(flipped_cond_points)
        # flipped_cond_pcd.colors = o3d.utility.Vector3dVector(ds_colors)
        flipped_cond_points = np.append(flipped_cond_points,ds_colors,axis = 1)
        print(flipped_cond_points.shape)
        # o3d.visualization.draw_geometries([flipped_cond_pcd])
        voxels_th, indices_th, num_p_in_vx_th = gen(torch.tensor(flipped_cond_points), empty_mean = True)
        voxels_np = voxels_th.numpy() 
        conditioning_voxel_points = np.reshape(voxels_np, (-1,6))
        print(conditioning_voxel_points.shape)
        # add batch size
        conditioning_voxel_points = conditioning_voxel_points.T
        conditioning_voxel_points = conditioning_voxel_points[None, :,:]

        conditioning_voxel_points = torch.tensor(conditioning_voxel_points.astype(np.single))

        pointnet_conditioing = conditioning_model(conditioning_voxel_points)
        pointnet_conditioing = pointnet_conditioing.swapaxes(1,2)

        running_occ_pm = utils.pc_to_pointmap(np.asarray(local_running_occ_pcd.points), 
                                                voxel_size = 0.1,
                                                x_y_bounds = [-2.0, 2.0],
                                                z_bounds = [-1.5, 0.8])

        running_unoc_pm = utils.pc_to_pointmap(np.asarray(local_running_unoc_pcd.points), 
                                                    voxel_size = 0.1,
                                                    x_y_bounds = [-2.0, 2.0],
                                                    z_bounds = [-1.5, 0.8])
        noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

        inpained_pm = utils.inpainting_pointmaps_w_freespace(model,
                                                noise_scheduler,
                                                pointnet_conditioing,
                                                40,
                                                running_occ_pm,
                                                running_unoc_pm,
                                                torch_device = "cpu",
                                                denoising_steps = int(30),
                                                guidance_scale = int(1),
                                                sample_batch_size = 1)


        inpained_points = utils.pointmap_to_pc(inpained_pm[0],
                                                voxel_size = 0.1,
                                                x_y_bounds = [-2, 2],
                                                z_bounds = [-1.5, 0.8])

        # inpainted_pc = utils.pointmap_to_pc(inpained_pm[0],
        #                                         voxel_size = 0.1,
        #                                         x_y_bounds = [-2, 2],
        #                                         z_bounds = [-1.5, 0.8])
        pcd_inpaint = o3d.geometry.PointCloud()
        pcd_inpaint.points = o3d.utility.Vector3dVector(inpained_points)
        # o3d.visualization.draw_geometries([pcd_inpaint,local_running_occ_pcd,cond_pcd])
        
        o3d.io.write_point_cloud("/hdd/sceneDiff_data/real_data/diff_pcds_s_1/" + str(correct_nums)+ "p_c.pcd", pcd_inpaint)
        for i, img in enumerate(np.asarray(inpained_pm[0])):
            #normalize the outputs to 255 in each pixel
            output = copy.deepcopy(img) * 255
            #dupicate it to be an image
            output = np.repeat(output[:, :, np.newaxis], 3, axis=2)
            print(output.shape)
            #save it as a cv2 image
            cv2.imwrite("/hdd/sceneDiff_data/real_data/diff_images_s_1/" + str(coor_idx) + "_z_" + str(i)  + ".png", output )