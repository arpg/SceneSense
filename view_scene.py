import open3d as o3d
from natsort import natsorted
import os
import numpy as np
from scipy import interpolate

path = "/hdd/hm3d_glb/00013-sfbj7jspYWj/sfbj7jspYWj.glb"

mesh = o3d.io.read_triangle_mesh(path,enable_post_processing=True)
mesh.compute_vertex_normals()
coor = o3d.geometry.TriangleMesh.create_coordinate_frame()

#load the coordiantes:
#loop through all the directories in house 1
house_path = "/hdd/sceneDiff_data/house_1/"
#get all the directories
house_dirs = natsorted(os.listdir(house_path))
# print(house_dirs)nning octomap
coor_arr = np.empty((0,3), float)
for step in house_dirs:
    curr_coor = np.loadtxt( house_path + step + "/running_octomap/curr_pose.txt")
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
colors = np.zeros((len(coor_arr), 3))
colors[:,0] = colors[:,0]
colors[0,0] = 0
colors[0,1] = 1 
coor_pcd.colors = o3d.utility.Vector3dVector(colors)
o3d.visualization.draw_geometries([mesh, coor_pcd])