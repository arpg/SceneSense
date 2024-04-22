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

path = "/hdd/hm3d_glb/00027-cVppJowrUqs/cVppJowrUqs.glb"

mesh = o3d.io.read_triangle_mesh(path,enable_post_processing=True)
mesh.compute_vertex_normals()


#get all the directories
house_dirs = natsorted(os.listdir(house_path))
coor_arr = np.empty((0,3), float)
for house_step in house_dirs[0:150]:
    curr_coor = np.loadtxt( house_path + house_step + "/running_octomap/curr_pose.txt")
    #need to swap y and z i think
    curr_coor[[1,2]] = curr_coor[[2,1]]
    curr_coor[1] = -curr_coor[1]
    curr_coor = curr_coor[None,:]
    

    # print(curr_coor)
    coor_arr = np.append(coor_arr, curr_coor, axis=0)

#create a spline so it looks better
# print(coor_arr[:,0])
# tck = interpolate.splrep(coor_arr[:,0], coor_arr[:,1])
# spline = interpolate.splev(0.1, tck)
# print(spline.shape)
#turn the path into a pointcloud
coor_pcd = o3d.geometry.PointCloud()
coor_pcd.points = o3d.utility.Vector3dVector(coor_arr)
def key_callback(vis):
    ######################################
    #set the step
    # step = 64
    #######################################
    global step
    try: step
    except NameError: step = 0
    
    print("step: ", step)

    pruned_point_arr = coor_arr[0:step,:]
    updated_coor_pcd = o3d.geometry.PointCloud()
    updated_coor_pcd.points = o3d.utility.Vector3dVector(pruned_point_arr)
    colors = np.zeros((len(np.asarray(updated_coor_pcd.points)), 3))
    updated_coor_pcd.colors = o3d.utility.Vector3dVector(colors)
    # diff_path = "/hdd/sceneDiff_data/house_2/step_" + str(step) + "/diffused_pc_30.pcd"
    #for house 2:
    diff_path = "/hdd/sceneDiff_data/combined_image_data_house_2/diff/pointcloud_" + str(step) + ".pcd"
    running_occ_path = "/hdd/sceneDiff_data/house_2/step_" + str(step) + "/running_octomap/running_occ.pcd"
    #ground truth
    #load the gt data
    #for house 2
    gt_file_path = "/home/arpg/Documents/habitat-lab/house_2/occupancy_gt.pcd"
    # gt_file_path = '/home/arpg/Documents/habitat-lab/running_octomap/gt_occ_point.pcd'
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
    # local_occ_points = points_within_distance(0.0,0.0,np.asarray(occ_point_shift.points),2.0)
    local_occ_points =np.asarray(occ_point_shift.points)
    #remove the lower floors
    local_occ_points = local_occ_points[local_occ_points[:,1] > -1.7]
    #remove the celing
    local_occ_points = local_occ_points[local_occ_points[:,1] < 0.9]

    #########################3
    #masage for overlay
    #############################3
    x_90_rot = np.eye(4)
    rot_mat = [ [1.0000000,  0.0000000,  0.0000000],
                [0.0000000,  0.0000000, -1.0000000],
                [0.0000000,  1.0000000,  0.0000000 ]]
    x_90_rot[0:3,0:3] = rot_mat

    # local_occ_points[:, [1, 2]] = local_occ_points[:, [2, 1]]
    # local_occ_points[:,0] = -local_occ_points[:,0] 
    # local_occ_points[:,0] = 1 + local_occ_points[:,0] 


    local_occ_pcd = o3d.geometry.PointCloud()
    local_occ_pcd.points = o3d.utility.Vector3dVector(local_occ_points)
    local_occ_pcd.transform(hm_tx_mat)
    #get the points, pass by ref so i can update them
    tx_local_occ_points = np.asarray(local_occ_pcd.points)
    tx_local_occ_points[:, [1, 2]] = tx_local_occ_points[:, [2, 1]]
    tx_local_occ_points[:,1] = -tx_local_occ_points[:,1]
    tx_local_occ_points[:,2] = tx_local_occ_points[:,2] + 0.2

    #lets go through tall the dif fpoints and remove anything that is super close to a local point
    diff_points = np.asarray(dif_pcd.points)

    pruned_diff_points = np.empty((0,3), float)
    for point in diff_points:
        # if not(is_within_distance(point, local_occ_points, 0.09)):
        pruned_diff_points = np.append(pruned_diff_points, point[None,:], axis = 0)

    # print(diff_points.shape)
    # print(pruned_diff_points.shape)

    #this just shifts stuff a little bit for the viewer
    pruned_diff_points[:,0] = pruned_diff_points[:,0] + 0.05
    pruned_diff_points[:,2] = pruned_diff_points[:,2] + -0.08
    #create pruned pcd
    pruned_diff_pcd = o3d.geometry.PointCloud()
    pruned_diff_pcd.points = o3d.utility.Vector3dVector(pruned_diff_points)

    #prune gt data
    #lets go through tall the dif fpoints and remove anything that is super close to a local point
    pcd_gt_points = np.asarray(local_pcd.points)

    pruned_gt_points = np.empty((0,3), float)
    for point in pcd_gt_points:
        if not(is_within_distance(point, local_occ_points, 0.09)):
            pruned_gt_points = np.append(pruned_gt_points, point[None,:], axis = 0)

    # print(diff_points.shape)
    # print(pruned_diff_points.shape)

    #this just shifts stuff a little bit for the viewer
    # pruned_gt_points[:,0] = pruned_gt_points[:,0] + 0.05
    # pruned_gt_points[:,2] = pruned_gt_points[:,2] + -0.08
    #create pruned pcd
    pruned_gt_pcd = o3d.geometry.PointCloud()
    pruned_gt_pcd.points = o3d.utility.Vector3dVector(pruned_gt_points)
    ########################
    # Color stuff
    #########################
    #make diff a color gradiant:
    colors = np.zeros((len(np.asarray(pruned_diff_pcd.points)), 3))
    # print(np.max(np.asarray(pruned_diff_pcd.points)[:,1], axis = 0))
    # print(np.min(np.asarray(pruned_diff_pcd.points)[:,1], axis = 0))
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
    colors = np.zeros((len(np.asarray(pruned_gt_pcd.points)), 3))
    max_z = np.max(np.asarray(pruned_gt_pcd.points)[:,1], axis = 0)
    min_z = np.min(np.asarray(pruned_gt_pcd.points)[:,1], axis = 0)
    #create scaler constant
    # colors[:,0] = ((np.asarray(pruned_gt_pcd.points)[:,1] + 1.45)/(max_z - min_z))
    colors[:,0] = ((np.asarray(pruned_gt_pcd.points)[:,1] - min_z)/(max_z - min_z))*(1 - 0.3) + 0.3
    colors[:,1] = ((np.asarray(pruned_gt_pcd.points)[:,1] - min_z)/(max_z - min_z))*(1 - 0.3) + 0.3

    # colors[:,2] = colors[:,2] + 0.7
    # colors[:,1] = colors[:,1] + 0.5
    pruned_gt_pcd.colors = o3d.utility.Vector3dVector(colors)
    local_gt_vox = o3d.geometry.VoxelGrid.create_from_point_cloud(pruned_gt_pcd, voxel_size=0.1)

    # o3d.visualization.draw_geometries([local_gt_vox, local_occ_vox])

    ctr  = vis.get_view_control()
    view_param =ctr.convert_to_pinhole_camera_parameters()
    vis.clear_geometries()

    vis.add_geometry(mesh)
    vis.add_geometry(local_occ_vox)
    vis.add_geometry(updated_coor_pcd)
    # vis.add_geometry(coor)
    # parameters = o3d.io.read_pinhole_camera_parameters("ScreenCamera_2024-03-04-17-17-19.json")
    ctr.convert_from_pinhole_camera_parameters(view_param)
    num_local_occ = len(np.asarray(local_occ_pcd.points))
    num_diff_point = len(np.asarray(pruned_diff_pcd.points))
    print("local_occ: ", num_local_occ)
    print("num diff: ", num_diff_point)
    print("diff per occ: ", num_diff_point/num_local_occ)
    step += 1
# ctr  = self.o3d_visualizer.get_view_control()
# view_param =ctr.convert_to_pinhole_camera_parameters()


vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window()
vis.add_geometry(mesh)
vis.register_key_callback(65, key_callback) #65 is a
vis.run()
vis.destroy_window()