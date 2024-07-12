import os
import open3d as o3d
from natsort import natsorted
import numpy as np
import time
import SceneDiffusion.utils as utils
from scipy.spatial.transform import Rotation as R

SEQUENCE_NAMES = ['IRL', 'ECOT', 'RUSTANDY']

#load the training folders
dataset_dir = '/home/brendan/realsense_data/dataset/'
raw_data = 'raw/'
trajs_dir = 'trajs/'



def read_pose_txt(filename):
    # Step 1: Read the transformation matrix from the file
    with open(filename, 'r') as file:
        # Read the single line containing the matrix
        matrix_str = file.readline()
 
        numbers_str = matrix_str.replace('[', '').replace(']', '').split()
        # Convert to float and create a numpy array
        numbers = np.array([float(num) for num in numbers_str])
 
        # Step 2: Reshape the flat array to a 4x4 matrix
        transform_matrix = numbers.reshape(4, 4)
 
    return transform_matrix

if __name__ == '__main__':
    sequence_name = 'doncey_full'

    # pcd = o3d.io.read_point_cloud('/home/brendan/realsense_data/dataset/raw/doncey_full/doncey_full.pcd')
    # points = np.asarray(pcd.points)
    # # points = points[points[:, 2] < 1]
    # # points = points[points[:, 2] > -1.2]
    # pcd_cropped = o3d.geometry.PointCloud()
    # pcd_cropped.points = o3d.utility.Vector3dVector(points)
    
    
    # coor = o3d.geometry.TriangleMesh.create_coordinate_frame()
    # o3d.visualization.draw_geometries([pcd_cropped, coor])

    gt_files = [s for s in os.listdir(os.path.join(dataset_dir, trajs_dir, sequence_name, 'gt_point_maps/'))]
    gt_files= natsorted(gt_files)
    cond_files = [s for s in os.listdir(os.path.join(dataset_dir, trajs_dir, sequence_name, 'conditioning/'))]
    cond_files = natsorted(cond_files)
    pose_files = [s for s in os.listdir(os.path.join(dataset_dir, raw_data, sequence_name, 'poses/'))]
    pose_files = natsorted(pose_files)
    x = np.random.randint(0, np.random.choice(1000000))
    np.random.seed(x)

    print(gt_files[0], cond_files[0], pose_files[0])

    frame_index = np.random.randint(0, len(pose_files))
    frame_index = 0

    
    #gt_file = gt_files[frame_index]
    gt_file = os.path.join(dataset_dir, raw_data, sequence_name, 'full_octomap.pcd')
    cond_file = cond_files[frame_index]
    pose_file = pose_files[frame_index]

    t625_frame_list = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
    cond_list = o3d.geometry.PointCloud()
    frame = 0
    for frame_index in range(0, 3000, 50):
        # print(frame)
        cond_file = cond_files[frame_index]
        pose_file = pose_files[frame_index]
        print(f"con: {cond_file}, pose: {pose_file}")
        # with open(os.path.join(dataset_dir, raw_data, sequence_name, 'poses/', pose_file)) as file:
        #     pose = file.readline()
        #     pose = np.array([float(x) for x in pose.split(' ')])

        # rotation_obj = R.from_euler('zyx', pose[3:])
        # hm_tx_mat = utils.homogeneous_transform(pose[:3], rotation_obj.as_quat())

        # position = pose[:3]
        # #rotation = R.from_euler('xyz', pose[3:], degrees=False)
        # rotation = R.from_rotvec(pose[:3])
        # transformation_matrix = np.eye(4)  # Initialize a 4x4 identity matrix
        # transformation_matrix[:3, :3] = rotation.as_matrix() # Set the rotation part
        # transformation_matrix[:3, 3] = position  # Set the translation part
    
        # Create a frame in Open3D and apply the transformation to the frame
        rot_matrix = read_pose_txt(os.path.join(dataset_dir, raw_data, sequence_name, 'poses/', pose_file))
        t625_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)  # size controls the scale of the frame
        t625_frame.transform(rot_matrix)

        cond_points = np.load(os.path.join(dataset_dir, trajs_dir, sequence_name, 'conditioning/', cond_file))
        
        # print(cond_file, gt_file)
    
        # gt_points = utils.pointmap_to_pc(gt_points)
        # gt = o3d.geometry.PointCloud()
        # gt.points = o3d.utility.Vector3dVector(gt_points)
        cond = o3d.geometry.PointCloud()
        print(cond_points.shape)
        cond.points = o3d.utility.Vector3dVector(cond_points[:, :3])
        cond.colors = o3d.utility.Vector3dVector(cond_points[:, 3:])
        cond.transform(rot_matrix)

        #colors = np.zeros((len(np.asarray(cond.points)), 3))
        # colors[:,0] = colors[:,0]
        #cond.colors = o3d.utility.Vector3dVector(colors)
        t625_frame_list += t625_frame
        cond_list += cond

    # cond_file = cond_files[gt_file]
    # gt_file = gt_files[cond_file]
    gt_points = o3d.io.read_point_cloud(gt_file)
    rot_matrix_gt = read_pose_txt(os.path.join(dataset_dir, raw_data, sequence_name, 'poses/', pose_file))
    #gt_points.transform(utils.inverse_homogeneous_transform(rot_matrix))
    #gt_points = np.load(os.path.join(dataset_dir, trajs_dir, sequence_name, gt_file))


    o3d.visualization.draw_geometries([cond_list, gt_points, t625_frame_list])


