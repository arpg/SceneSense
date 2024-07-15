
import os.path
import numpy as np
from scipy.spatial.transform import Rotation as R
#from pypcd import pypcd
#import open3d as o3d
from dataclasses import dataclass
#import sensor_msgs.point_cloud2 as pc2
from natsort import natsorted
# from pypcd4 import PointCloud
import open3d as o3d

def pc_to_str(pc):
    good_idx = ~np.any(np.isnan(pc), axis=1)
    pc = pc[good_idx, :]

    return ''.join([' '.join([str(x) for x in xx]) + '\n' for xx in pc[:, :3]])

if __name__ == '__main__':
    base_path = '/home/brendan/realsense_data/'
    bag_name = 'doncey2'
    dataset_dir = '/home/brendan/realsense_data/dataset/raw/'
    data_dir = 'doncey_full/'
    text_file_name = 'doncey_full.log'
    poses = dataset_dir + data_dir + 'poses/'
    pcs = dataset_dir + data_dir + 'point_clouds/'
    
    pose_files = [s for s in os.listdir(poses)]
    pose_files= natsorted(pose_files)
    pc_files = [s for s in os.listdir(pcs)]
    pc_files = natsorted(pc_files)
    num_pcs = len(pc_files)
    print(num_pcs)
    with open(dataset_dir + data_dir + text_file_name, 'w') as file:

        for i, files in enumerate(zip(pc_files,pose_files)):
            pc_file, pose_file = files
            with open(os.path.join(poses, pose_file), 'r') as f:
                odom = f.readline()
                odom = np.array([float(x) for x in odom.split(' ')])

            pc_ = o3d.io.read_point_cloud(os.path.join(pcs, pc_file))
            pc = np.asarray(pc_.points)
            pc_depth = np.sqrt(np.linalg.norm(pc, axis=1))
            pc = pc[pc_depth < 5]
            file.write(f'NODE {" ".join([str(x) for x in odom])}\n')
            file.write(pc_to_str(pc))
            if i % 100 == 0:
                print("{:.2f}".format(i / num_pcs))


