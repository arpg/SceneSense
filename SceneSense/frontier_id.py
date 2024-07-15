import open3d as o3d
import numpy as np
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN

def is_surrounded(point, close_points):
    x_pos = False
    x_neg = False
    y_pos = False
    y_neg = False
    z_pos = False
    z_neg = False
    difference_vals = close_points - point
    # print(difference_vals)
    positive_x = any(p[0] > point[0] for p in close_points)
    negative_x = any(p[0] < point[0] for p in close_points)

    positive_y = any(p[1] > point[1] for p in close_points)
    negative_y = any(p[1] < point[1] for p in close_points)

    positive_z = any(p[2] > point[2] for p in close_points)
    negative_z = any(p[2] < point[2] for p in close_points)

    # Return True if there are positive and negative values for x, y, and z
    return positive_x and negative_x and positive_y and negative_y and positive_z and negative_z


    # for query_point in close_point:
        
def categorize_points(points, categories):
    categories_dict = {}
    
    for i in range(len(points)):
        category = categories[i]
        point = points[i]
        
        if category not in categories_dict:
            categories_dict[category] = []
        
        categories_dict[category].append(point)
    
    return categories_dict
def calculate_median_points(categories_dict):
    medians = []

    for category, points_list in categories_dict.items():
        median_point = np.median(points_list, axis=0)
        medians.append(median_point)

    return np.array(medians)


occ_pcd_path = '/home/arpg/Documents/habitat-lab/running_octomap/running_occ.pcd'
unocc_pcd_path = '/home/arpg/Documents/habitat-lab/test_unoc.pcd'

occ_pcd = o3d.io.read_point_cloud(occ_pcd_path)
colors = np.zeros((len(np.asarray(occ_pcd.points)), 3))
# colors[:,0] = colors[:,0]
occ_pcd.colors = o3d.utility.Vector3dVector(colors)

unocc_pcd = o3d.io.read_point_cloud(unocc_pcd_path)
colors = np.zeros((len(np.asarray(unocc_pcd.points)), 3))
colors[:,0] = colors[:,0] + 1
unocc_pcd.colors = o3d.utility.Vector3dVector(colors)

#find all free points with open next to them
#convert pcds to numpy arrays
occ_points = np.asarray(occ_pcd.points)
unocc_points = np.asarray(unocc_pcd.points)
all_points = np.append(occ_points,(unocc_points), axis = 0)
print(occ_points.shape)
print(unocc_points.shape)
print(all_points.shape)
#for each unoccupied point, find the 6 cloest points
# https://stackoverflow.com/questions/54114728/finding-nearest-neighbor-for-python-numpy-ndarray-in-3d-space 
kdtree=KDTree(all_points)
#need to query 7 because it includes itself
dist,points = kdtree.query(unocc_points,7)
# print(points.shape)
# print(points[0])
#now check if there are points surrounding the voxel ()



front_point_arr = np.empty((0,3), float)
for unoc_point,near_point_idx in zip(unocc_points, points):
    # print(unoc_point)
    # print(near_point_idx)
    # print(all_points[near_point_idx])
    
    # front_point = has_close_points_in_both_directions(unoc_point,all_points[near_point_idx])
    front_point = is_surrounded(unoc_point,all_points[near_point_idx])
    
    if front_point == False:
        unoc_point = unoc_point[None,:]
        front_point_arr = np.append(front_point_arr, unoc_point, axis = 0)
print(front_point_arr.shape)
#need to crop top half of map out
front_point_arr = front_point_arr[front_point_arr[:,1] < 1.3]
#and crop out bottom
front_point_arr = front_point_arr[front_point_arr[:,1] > 0.3 ]


model = DBSCAN(eps=0.2, min_samples=5)
model.fit_predict(front_point_arr)
pred = model.fit_predict(front_point_arr)


print("number of cluster found: {}".format(len(set(model.labels_))))
print('cluster for each point: ', model.labels_)

#update the colors of the pointcloud
cluster_colors = np.zeros((len(front_point_arr), 3))
for idx, cluster in enumerate(model.labels_):
    # print(cluster + 1)
    np.random.seed(cluster + 1)
    cluster_colors[idx,0] = np.random.rand()
    cluster_colors[idx,1] = np.random.rand()
    cluster_colors[idx,2] = np.random.rand()

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(front_point_arr)
# colors = np.zeros((len(np.asarray(pcd.points)), 3))
# colors[:,2] = colors[:,2] + 1
# pcd.colors = o3d.utility.Vector3dVector(colors)
pcd.colors = o3d.utility.Vector3dVector(cluster_colors)

#compute cluster centroids, probably the median point in 3d
catagory_dict = categorize_points(front_point_arr,model.labels_ + 1)
#for each cluster find the centroid of the frontier
median_fronts = calculate_median_points(catagory_dict)
# print(median_fronts)
colors = np.zeros((len(median_fronts), 3))
colors[:,2] = colors[:,2] + 1
pcd_centroids = o3d.geometry.PointCloud()
pcd_centroids.points = o3d.utility.Vector3dVector(median_fronts)
pcd_centroids.colors = o3d.utility.Vector3dVector(colors)
o3d.visualization.draw_geometries([occ_pcd,pcd, pcd_centroids])
