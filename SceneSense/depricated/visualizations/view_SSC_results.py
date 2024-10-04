import numpy as np
import open3d as o3d

def create_pointcloud_from_occupancy_grid(grid):
    """
    Create a point cloud from an occupancy grid.
    
    Parameters:
    - grid: A 3D numpy array where 0 denotes unoccupied and 1 denotes occupied.
    
    Returns:
    - points: A list of tuples, where each tuple represents the (x, y, z) coordinates
              of an occupied point in the grid.
    """
    points = []
    # Iterate through the grid
    point_arr = np.empty((0,3), float)
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            for z in range(grid.shape[2]):
                if grid[x, y, z] == 1:
                    # Add the occupied point to the list
                    point_arr = np.append(point_arr,np.array([[x, y, z]]), axis = 0)
                    print("point added")
                    print(point_arr.shape)
                    
                    # points.append((x, y, z))
    return point_arr

# Example usage
# Create an example occupancy grid with the shape (60, 36, 60)
# For simplicity, this example will create a grid with random occupancy
# np.random.seed(0) # For reproducible results
# occupancy_grid = np.random.choice([0, 1], size=(60, 36, 60), p=[0.95, 0.05])

# Create the point cloud

input_pc_path = "/hdd/sceneDiff_data/house_1/step_100/running_octomap/sample_pc.npy"
point_path =np.load(input_pc_path)
print(point_path.shape)

# file_path = "/hdd/results_sketch/0001.npy"
# obj = np.load(file_path)
# print(obj.shape)
# print(np.unique(obj))


# point_cloud = create_pointcloud_from_occupancy_grid(obj)

# print(f"Generated point cloud with {len(point_cloud)} points.")
# print(point_cloud)

conditioning_pcd = o3d.geometry.PointCloud()
conditioning_pcd.points = o3d.utility.Vector3dVector(point_path)
o3d.visualization.draw_geometries([conditioning_pcd])