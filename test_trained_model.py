from dataclasses import dataclass
from diffusers import UNet2DConditionModel
import torch
from diffusers import DDPMScheduler
import torch.nn.functional as F
from pointnet2_scene_diffusion import get_model
import os
from natsort import natsorted
import numpy as np
import copy
import open3d as o3d
from tqdm.auto import tqdm
import wandb
import random
from huggingface_hub import login
from diffusers.optimization import get_cosine_schedule_with_warmup
import utils.utils as utils

model = UNet2DConditionModel.from_pretrained("alre5639/diff_unet")
conditioning_model = get_model()
conditioning_model.load_state_dict(torch.load("/home/arpg/Documents/SceneDiffusion/conditioning_model_weights/cond_model" + str(217)))


#make sure all the data moves through the network correctly
sample_noise_start = torch.randn(1,22,30, 30)
sample_noise_target = torch.randn(1,22,30, 30)
sample_pc_in = torch.randn(1, 3, 65536)
#input to pointnet needs to be shape: 1, 3, 65536
sample_conditioning = conditioning_model(sample_pc_in)
#need to swap axis 1 and 2 to get it in the right shape
sample_conditioning = sample_conditioning.swapaxes(1, 2)
#output from pointnet neeeds to be shape: 1,n, channels
print(sample_conditioning.shape)
print("Unet output shape:", model(sample_noise_start, timestep=1.0, encoder_hidden_states=sample_conditioning).sample.shape)


f = open("/home/arpg/Documents/habitat-lab/out_training_data/sample_octomap_running.txt", "r")

final_pointcloud = np.zeros((1,3,65536), dtype=np.single)

node_count = 0

for x in f:
    if x[0:4] == "NODE":
        if node_count == 1:
            final_pointcloud = copy.deepcopy(pointcloud)
        elif node_count == 0:
            pass
        #add a check that stops the conditoning building when it is the same size as the gt
        elif node_count == 15:
            break
        else:
            final_pointcloud = np.append(final_pointcloud, pointcloud, axis = 0)
        
        node_count += 1
        pc_count = 0
        #make empty pointcloud to fill  
        pointcloud = np.zeros((1,3,65536), dtype=np.single)
        print(final_pointcloud.shape)
    else:
        coord = np.fromstring(x, dtype=np.single, sep=' ')
        pointcloud[0,:,pc_count] = coord
        pc_count += 1

        
#   print(x)
final_pointcloud = torch.from_numpy(final_pointcloud)
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
#do the denoising:
post_model_conditioning_batch = conditioning_model(final_pointcloud)
post_model_conditioning_batch = post_model_conditioning_batch.swapaxes(1, 2)
print(post_model_conditioning_batch[13].shape)
test_conditioning = post_model_conditioning_batch[13]
test_conditioning = test_conditioning[None,:,:]
point_map = utils.denoise_guided_inference(model, noise_scheduler,test_conditioning, 30)

#this should go in utils 
arr = np.empty((0,3), float)
for idx_x,x in enumerate(point_map[0]):
    for idx_y, y in enumerate(x):
        for idx_z, z in enumerate(y):
            if z == 1:
                arr = np.append(arr, np.array([[idx_x,idx_y,idx_z]]), axis=0)



# pcd_local = o3d.geometry.PointCloud()
# pcd_local.points = o3d.utility.Vector3dVector(final_pointcloud[13].T)
# o3d.visualization.draw_geometries([pcd_local])

pcd_local = o3d.geometry.PointCloud()
pcd_local.points = o3d.utility.Vector3dVector(arr)
o3d.visualization.draw_geometries([pcd_local])

# point_map= np.load("training_pointmaps/pointmap_14.npy")
# print(point_map.shape)

# arr = np.empty((0,3), float)
# for idx_x,x in enumerate(point_map):
#     for idx_y, y in enumerate(x):
#         for idx_z, z in enumerate(y):
#             if z == 1:
#                 arr = np.append(arr, np.array([[idx_x,idx_y,idx_z]]), axis=0)


# pcd_local = o3d.geometry.PointCloud()
# pcd_local.points = o3d.utility.Vector3dVector(arr)
# o3d.visualization.draw_geometries([pcd_local])
