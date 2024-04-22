import os
from natsort import natsorted
import numpy as np
import copy
import open3d as o3d
import utils.utils as utils
from scipy.spatial.transform import Rotation
import cv2

def scale_value(old_value, old_min, old_max, new_min, new_max):
    return ((old_value - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min

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
def is_within_distance(point, local_occ_points, d):
    """
    Check if a 3D point is within distance 'd' of any point in a list of points.
    
    Parameters:
    - point: A NumPy array of shape (3,) representing the 3D point to check.
    - local_occ_points: A NumPy array of shape (n, 3) representing the list of points.
    - d: The distance threshold.
    
    Returns:
    - True if 'point' is within 'd' of any point in 'local_occ_points', False otherwise.
    """
    # Calculate the Euclidean distances from 'point' to each point in 'local_occ_points'
    distances = np.linalg.norm(local_occ_points - point, axis=1)
    
    # Check if any distance is within 'd'
    return np.any(distances <= d)

def scale_value(old_value, old_min, old_max, new_min, new_max):
    return ((old_value - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min

#load diffused frames
house_path = "/hdd/sceneDiff_data/house_2/"
#get all the directories
house_dirs = natsorted(os.listdir(house_path))



######################################
#set the step
step = 15
#######################################


# diff_path = "/hdd/sceneDiff_data/figure_data/" "pointcloud_" + str(0)+ ".pcd"
#for house 2:
diff_path = "/hdd/sceneDiff_data/combined_image_data_house_2/diff/pointcloud_" + str(step) + ".pcd"
running_occ_path = "/hdd/sceneDiff_data/house_2/step_" + str(step) + "/running_octomap/running_occ.pcd"
#ground truth
#load the gt data
# gt_file_path = '/home/arpg/Documents/habitat-lab/running_octomap/gt_occ_point.pcd'
gt_file_path =  '/home/arpg/Documents/habitat-lab/house_2/occupancy_gt.pcd'
gt_pcd = o3d.io.read_point_cloud(gt_file_path)
#load the pose data
curr_coor = np.loadtxt( "/hdd/sceneDiff_data/house_2/step_" + str(step) + "/running_octomap/curr_pose.txt")
curr_rot = np.loadtxt( "/hdd/sceneDiff_data/house_2/step_" + str(step) + "/running_octomap/curr_heading.txt")
rotation_obj = Rotation.from_rotvec(curr_rot)
hm_tx_mat = utils.homogeneous_transform(curr_coor, rotation_obj.as_quat())
coor = o3d.geometry.TriangleMesh.create_coordinate_frame()

#get local gt
gt_points_shift = copy.deepcopy(gt_pcd).transform(utils.inverse_homogeneous_transform(hm_tx_mat))
#get the local data
local_gt_points = points_within_distance(0.0,0.0,np.asarray(gt_points_shift.points),2.0)
#remove the lower floors
local_gt_points = local_gt_points[local_gt_points[:,1] > -1.6]
#remove the celing
local_gt_points = local_gt_points[local_gt_points[:,1] < 0.9]

local_pcd = o3d.geometry.PointCloud()
local_pcd.points = o3d.utility.Vector3dVector(local_gt_points)

dif_pcd = o3d.io.read_point_cloud(diff_path)
#you only need to do this in house 1 for some reasons
# dif_pcd.transform(utils.inverse_homogeneous_transform(hm_tx_mat))
# o3d.visualization.draw_geometries([local_pcd])
# o3d.visualization.draw_geometries([dif_pcd])

occ_pcd = o3d.io.read_point_cloud(running_occ_path)
#get local gt
occ_point_shift = copy.deepcopy(occ_pcd).transform(utils.inverse_homogeneous_transform(hm_tx_mat))
#get the local data
local_occ_points = points_within_distance(0.0,0.0,np.asarray(occ_point_shift.points),2.0)
#remove the lower floors
local_occ_points = local_occ_points[local_occ_points[:,1] > -1.6]
#remove the celing
local_occ_points = local_occ_points[local_occ_points[:,1] < 0.9]
local_occ_pcd = o3d.geometry.PointCloud()
local_occ_pcd.points = o3d.utility.Vector3dVector(local_occ_points)

#lets go through tall the dif fpoints and remove anything that is super close to a local point
diff_points = np.asarray(dif_pcd.points)

pruned_diff_points = np.empty((0,3), float)
for point in diff_points:
    if not(is_within_distance(point, local_occ_points, 0.09)):
        pruned_diff_points = np.append(pruned_diff_points, point[None,:], axis = 0)

print(diff_points.shape)
print(pruned_diff_points.shape)

#this just shifts stuff a little bit for the viewer
pruned_diff_points[:,0] = pruned_diff_points[:,0] + 0.05
pruned_diff_points[:,2] = pruned_diff_points[:,2] + -0.05
#create pruned pcd
pruned_diff_pcd = o3d.geometry.PointCloud()
pruned_diff_pcd.points = o3d.utility.Vector3dVector(pruned_diff_points)

########################
# Color stuff
#########################
#make diff a color gradiant:
colors = np.zeros((len(np.asarray(pruned_diff_pcd.points)), 3))
print(np.max(np.asarray(pruned_diff_pcd.points)[:,1], axis = 0))
print(np.min(np.asarray(pruned_diff_pcd.points)[:,1], axis = 0))
max_z = np.max(np.asarray(pruned_diff_pcd.points)[:,1], axis = 0)
min_z = np.min(np.asarray(pruned_diff_pcd.points)[:,1], axis = 0)
#create scaler constant
# colors[:,0] = ((np.asarray(pruned_diff_pcd.points)[:,1] + 1.45)/(max_z - min_z))
colors[:,0] = ((np.asarray(pruned_diff_pcd.points)[:,1] - min_z)/(max_z - min_z))*(1 - 0.3) + 0.3
# colors[:,1] = 0.0 # ((np.asarray(pruned_diff_pcd.points)[:,1] + 1.45)/(max_z - min_z))
colors[:,2] = colors[:,2]
pruned_diff_pcd.colors = o3d.utility.Vector3dVector(colors)

#try a voxel grid
diff_vox = o3d.geometry.VoxelGrid.create_from_point_cloud(pruned_diff_pcd, voxel_size=0.1)
# o3d.visualization.draw_geometries([pcd_vox])

#lets do the same colors for the running
colors = np.zeros((len(np.asarray(local_occ_pcd.points)), 3))
max_z = np.max(np.asarray(local_occ_pcd.points)[:,1], axis = 0)
min_z = np.min(np.asarray(local_occ_pcd.points)[:,1], axis = 0)
#create scaler constant
# colors[:,0] = ((np.asarray(local_occ_pcd.points)[:,1] + 1.45)/(max_z - min_z))
colors[:,1] = ((np.asarray(local_occ_pcd.points)[:,1] - min_z)/(max_z - min_z))*(1 - 0.3) + 0.3
# colors[:,2] = colors[:,2] + 0.7
# colors[:,1] = colors[:,1] + 0.5
local_occ_pcd.colors = o3d.utility.Vector3dVector(colors)
local_occ_vox = o3d.geometry.VoxelGrid.create_from_point_cloud(local_occ_pcd, voxel_size=0.1)


#lets do the same colors for the running
colors = np.zeros((len(np.asarray(local_pcd.points)), 3))
max_z = np.max(np.asarray(local_pcd.points)[:,1], axis = 0)
min_z = np.min(np.asarray(local_pcd.points)[:,1], axis = 0)
#create scaler constant
# colors[:,0] = ((np.asarray(local_pcd.points)[:,1] + 1.45)/(max_z - min_z))
colors[:,0] = ((np.asarray(local_pcd.points)[:,1] - min_z)/(max_z - min_z))*(1 - 0.3) + 0.3
colors[:,1] = ((np.asarray(local_pcd.points)[:,1] - min_z)/(max_z - min_z))*(1 - 0.3) + 0.3

# colors[:,2] = colors[:,2] + 0.7
# colors[:,1] = colors[:,1] + 0.5
local_pcd.colors = o3d.utility.Vector3dVector(colors)
local_gt_vox = o3d.geometry.VoxelGrid.create_from_point_cloud(local_pcd, voxel_size=0.1)

#get view data 
# ctr = o3d.visualization.get_view_control()
# parameters = o3d.io.read_pinhole_camera_parameters("ScreenCamera_2024-02-29-10-16-09.json")
# ctr.convert_from_pinhole_camera_parameters(parameters)

# o3d.visualization.draw_geometries([local_occ_vox, diff_vox])


def key_callback(vis):
    # print('key')
    pcd_file_path = '/home/arpg/Documents/habitat-lab/running_octomap/running_occ.pcd'
    #need to add to load the gt file so we can get the gt predictions
    gt_file_path = '/home/arpg/Documents/habitat-lab/running_octomap/gt_occ_point.pcd'
    gt_pcd = o3d.io.read_point_cloud(gt_file_path)
    #load just the points at the current pose
    gt_points = np.asarray(gt_pcd.points)
    #get the current pose of the robot
    curr_coor = np.loadtxt("/home/arpg/Documents/habitat-lab/running_octomap/curr_pose.txt")
    curr_rot= np.loadtxt("/home/arpg/Documents/habitat-lab/running_octomap/curr_heading.txt")

    local_gt_points = points_within_distance(curr_coor[0],curr_coor[2],gt_points,2.0)
    #remove celling and lower floors
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

    pcd = o3d.io.read_point_cloud(pcd_file_path)
    ctr  = vis.get_view_control()
    view_param =ctr.convert_to_pinhole_camera_parameters()
    vis.clear_geometries()

    vis.add_geometry(local_occ_vox)
    vis.add_geometry(diff_vox)
    
    
    parameters = o3d.io.read_pinhole_camera_parameters("ScreenCamera_2024-02-29-10-16-09.json")
    ctr.convert_from_pinhole_camera_parameters(parameters)


# ctr  = self.o3d_visualizer.get_view_control()
# view_param =ctr.convert_to_pinhole_camera_parameters()

pcd_file_path = '/home/arpg/Documents/habitat-lab/running_octomap/running_occ.pcd'
pcd = o3d.io.read_point_cloud(pcd_file_path)

vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window()
vis.add_geometry(pcd)
vis.register_key_callback(65, key_callback)
# vis.register_animation_callback(key_callback)
# Set the timer interval (in milliseconds)
# vis.get_timer().set_interval(1000)

vis.run()
vis.destroy_window()
