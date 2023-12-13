from pointnet2_scene_diffusion import get_model
import  torch
import numpy as np

# all we tell the model is how many classes we have, which we are going to remove since we 
# are getting rid of that part
test_pcd = np.loadtxt("test_pcd.txt", dtype = np.single)
test_pcd = test_pcd.T
test_pcd = np.expand_dims(test_pcd, axis = 0)
test_pcd = torch.from_numpy(test_pcd)
print("pointnet_input shape: ", test_pcd.shape)
model = get_model()
xyz = torch.rand(1, 3, 4096)
print(xyz.dtype)
print(test_pcd.dtype)
print(xyz.shape)
print(test_pcd.shape)
x = model(test_pcd)
print(x.shape)
