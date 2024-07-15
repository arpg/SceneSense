import numpy as np
import open3d as o3d
import utils.utils as utils
import spconv
from spconv.pytorch.utils import PointToVoxel
from spconv.pytorch.core import SparseConvTensor
import torch
import cv2
# this script will take the running txt file that has the poses and associated points
# and generate a pointmap

resolution = 0.1
x_grid = 30
y_grid = 30
z_grid = 22

txt_path = "/home/arpg/Documents/habitat-lab/out_training_data/sample_octomap_running.txt"

for i in range(350):
    print(i)
    pose_num = i
    pose, conditioning_pc = utils.get_pose_and_pc_from_running_txt(txt_path, pose_num)

    conditoning_save_pth = "data/rgbd_voxelized_data/"
    pointmap_save_pth = "data/gt_pointmaps/"

    #view loaded pc
    pcd_cond = o3d.geometry.PointCloud()
    pcd_cond.points = o3d.utility.Vector3dVector(conditioning_pc)
    # o3d.visualization.draw_geometries([pcd_cond])

    #load the octomap generated house
    pcd = o3d.io.read_point_cloud("/home/arpg/Documents/octomap/sample_house.pcd")
    # transform house data to robot frame
    pcd.transform(utils.inverse_homogeneous_transform(pose))
    house_points = np.asarray(pcd.points)
    # o3d.visualization.draw_geometries([pcd_cond, pcd])

    #get the local points
    local_points = utils.points_within_distance(0,0,house_points, 1.5)
    #remove the lower floors
    local_points = local_points[local_points[:,1] > -1.4]
    #remove the celing
    local_points = local_points[local_points[:,1] < 0.9]
    pcd_local = o3d.geometry.PointCloud()
    pcd_local.points = o3d.utility.Vector3dVector(local_points)
    # o3d.visualization.draw_geometries([pcd_cond, pcd_local])


    #load the associated colors
    img = cv2.imread("/home/arpg/Documents/habitat-lab/out_training_data/rgb_" + str(pose_num + 1) + ".png")
    colors = img.reshape(-1, img.shape[2])

    #create the full rgbd pc
    rgbd_pc = np.append(conditioning_pc, colors, axis = 1)
    print("\n\nRGBD shape: ", rgbd_pc.shape)

    #voxelize the conditioning pc (its unessicariloy dense)
    #only one point per voxel stops it from going crazy
    gen = PointToVoxel(vsize_xyz=[0.05, 0.05, 0.05],
                        coors_range_xyz=[-0, -10, -10, 10, 10, 10],
                        num_point_features=6,
                        max_num_voxels=10000,
                        max_num_points_per_voxel=1)

    voxels_th, indices_th, num_p_in_vx_th = gen(torch.tensor(rgbd_pc), empty_mean = True)
    voxels_np = voxels_th.numpy() 
    conditioning_voxel_points = np.reshape(voxels_np, (-1,6))
    #get average point location per voxel
    # avereaged_voxels = np.mean(voxels_np, axis = 1)
    # print(avereaged_voxels)
    # print(avereaged_voxels.shape)
    # print(np.max(avereaged_voxels, axis = 0))
    # print(np.min(avereaged_voxels, axis = 0))
    pcd_local_vox = o3d.geometry.PointCloud()
    pcd_local_vox.points = o3d.utility.Vector3dVector(conditioning_voxel_points[:,0:3])
    pcd_local_vox.colors = o3d.utility.Vector3dVector(conditioning_voxel_points[:,3:6]/255)
    # o3d.visualization.draw_geometries([pcd_local_vox])
    # o3d.visualization.draw_geometries([pcd_local_vox, pcd_local])

    #Convert Local GT data to a pointmap
    #range is going to be :
    # x = [-1.5, 1.5]
    # y = [-1.4, 0.9]
    # z = [-1.5 1.5]
    # @ 0.1 m resolution 
    #remember that y is actually vertical
    pointmap = utils.pc_to_pointmap(local_points, voxel_size = 0.1, x_y_bounds = [-1.5, 1.5], z_bounds = [-1.4, 0.9])

    # print(pointmap.shape)

    returned_pc = utils.pointmap_to_pc(pointmap)
    # print(returned_pc.shape)

    # reconstructed_pcd = o3d.geometry.PointCloud()
    # reconstructed_pcd.points = o3d.utility.Vector3dVector(returned_pc)
    # o3d.visualization.draw_geometries([reconstructed_pcd,pcd_local])
    np.save(pointmap_save_pth+"pointmap_" + str(i) + ".npy", pointmap)
    np.savetxt(conditoning_save_pth+"conditioning" + str(i) + ".txt", conditioning_voxel_points)

