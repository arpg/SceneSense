import numpy as np
import open3d as o3d
import utils.utils as utils
import spconv
from spconv.pytorch.utils import PointToVoxel
from spconv.pytorch.core import SparseConvTensor
import torch
import cv2
import os
from natsort import natsorted
from scipy.spatial.transform import Rotation

resolution = 0.1

#load the training folders
training_dirs = os.listdir("/home/arpg/Documents/habitat-lab/full_training_data/")

for house_name in training_dirs:
    #iterate throught the txt and save the pointclouds
    f = open("/home/arpg/Documents/habitat-lab/full_training_data/" + house_name + "/sample_octomap_running.txt", "r")
    node_count = 0
    for x in f:
        if x[0:4] == "NODE": 
            print(node_count)
            if node_count != 0:
                np.save("data/full_conditioning_pcs/" + house_name + "_pc_" + str(node_count - 1) + ".npy", pointcloud)
            node_count += 1
            pose = x.split()
            #make empty pointcloud to fill  
            pointcloud = np.zeros((3,65536), dtype=np.single)
            pc_count = 0
        else:
            coord = np.fromstring(x, dtype=np.single, sep=' ')
            pointcloud[:,pc_count] = coord
            pc_count += 1
    #generate the transform matrix