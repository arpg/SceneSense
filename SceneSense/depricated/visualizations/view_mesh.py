import open3d as o3d
from natsort import natsorted
import os
import numpy as np
from scipy import interpolate

path = "/home/arpg/Documents/habitat-lab/house_2/occupancy_gt.pcd"

mesh = o3d.io.read_point_cloud(path)
coor = o3d.geometry.TriangleMesh.create_coordinate_frame()

o3d.visualization.draw_geometries([mesh,coor])