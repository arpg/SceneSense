import os
import open3d as o3d
from natsort import natsorted
import numpy as np
import time
import SceneDiffusion.utils as utils

SEQUENCE_NAMES = ['IRL', 'ECOT', 'RUSTANDY']

#load the training folders
dataset_dir = '/home/brendan/spot_data/dataset/'
raw_data = 'raw/'
trajs_dir = 'trajs/'


if __name__ == '__main__':
    sequence_name = 'IRL'

    pcd = o3d.io.read_point_cloud('/home/brendan/spot_data/dataset/raw/IRL/IRL.pcd')
    points = np.asarray(pcd.points)
    points = points[points[:, 2] < 1]
    points = points[points[:, 2] > -1.2]
    pcd_cropped = o3d.geometry.PointCloud()
    pcd_cropped.points = o3d.utility.Vector3dVector(points)
    
    
    coor = o3d.geometry.TriangleMesh.create_coordinate_frame()
    o3d.visualization.draw_geometries([pcd_cropped, coor])




    # gt_files = [s for s in os.listdir(os.path.join(dataset_dir, trajs_dir, sequence_name, 'gt_point_maps/'))]
    # gt_files= natsorted(gt_files)
    # cond_files = [s for s in os.listdir(os.path.join(dataset_dir, trajs_dir, sequence_name, 'conditioning/'))]
    # cond_files = natsorted(cond_files)

    # x = np.random.randint(0, np.random.choice(1000000))
    # np.random.seed(x)
    # gt_file = np.random.choice(gt_files)
    # np.random.seed(x)
    # cond_file = np.random.choice(cond_files)

    # # cond_file = cond_files[350]
    # # gt_file = gt_files[350]
    # gt_points = np.load(os.path.join(dataset_dir, trajs_dir, sequence_name, 'gt_point_maps', gt_file))
    # cond_points = np.load(os.path.join(dataset_dir, trajs_dir, sequence_name, 'conditioning/', cond_file))
    
    # print(cond_file, gt_file)
  
    # gt_points = utils.pointmap_to_pc(gt_points)
    # gt = o3d.geometry.PointCloud()
    # gt.points = o3d.utility.Vector3dVector(gt_points)
    # cond = o3d.geometry.PointCloud()
    # cond.points = o3d.utility.Vector3dVector(cond_points)

    # colors = np.zeros((len(np.asarray(cond.points)), 3))
    # # colors[:,0] = colors[:,0]
    # cond.colors = o3d.utility.Vector3dVector(colors)


    # o3d.visualization.draw_geometries([gt, cond])

