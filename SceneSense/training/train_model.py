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
# import open3d as o3d
from tqdm.auto import tqdm
import wandb
import random
from huggingface_hub import login
from diffusers.optimization import get_cosine_schedule_with_warmup
import re
import utils.utils as utils

login(token="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")
wandb.init(
    # set the wandb project where this run will be logged
    project="scene_diffusion",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.02,
    "architecture": "CNN",
    "dataset": "CIFAR-100",
    "epochs": 10,
    }
)

@dataclass
class TrainingConfig:
    image_size = 128  # the generated image resolution
    train_batch_size = 16
    eval_batch_size = 16  # how many images to sample during evaluation
    num_epochs = 250
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 10
    save_model_epochs = 20
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "baseline_road_diffusion"  # the model name locally and on the HF Hub

    push_to_hub = True  # whether to upload the saved model to the HF Hub
    hub_private_repo = False
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0

torch_device = "cuda"
config = TrainingConfig()

#import a Unet2Dmodel 
model = UNet2DConditionModel(
    sample_size=40,  # the target image resolution
    in_channels=13,  # the number of input channels, 3 for RGB images
    out_channels=13,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    cross_attention_dim=128, #the dim of the guidance data?
    block_out_channels=(128, 256, 512,512),  # the number of output channels for each UNet block
    down_block_types=(
        "CrossAttnDownBlock2D",  # a regular ResNet downsampling block
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",  # a regular ResNet upsampling block
        "CrossAttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
    ),

)

#set up conditioning model
conditioning_model = get_model()

#make sure all the data moves through the network correctly
sample_noise_start = torch.randn(1,13,40, 40)
sample_noise_target = torch.randn(1,13,40, 40)
sample_pc_in = torch.randn(1, 6, 800)
#input to pointnet needs to be shape: 1, 3, 65536
sample_conditioning = conditioning_model(sample_pc_in)
#need to swap axis 1 and 2 to get it in the right shape
sample_conditioning = sample_conditioning.swapaxes(1, 2)
#output from pointnet neeeds to be shape: 1,n, channels
print(sample_conditioning.shape)
print("Unet output shape:", model(sample_noise_start, timestep=1.0, encoder_hidden_states=sample_conditioning).sample.shape)


########################
#get gt data
#########################3
gt_dir = "data/full_gt_pointmaps/"
gt_files = natsorted(os.listdir(gt_dir))

# ##################################
# #get the conditioning data
# ##################################
cond_dir = "data/full_conditioning_rgbd/"
cond_files = natsorted(os.listdir(cond_dir))


#make sure the gt and conditioning data are aligned
aligned_gt_files = []
aligned_cond_files = []
for i in range(len(cond_files)):
    split_gt_files = re.split(r'[._]', gt_files[i])
    split_conditioning_files = re.split(r'[._]', cond_files[i])
    if split_gt_files[0] == split_conditioning_files[0] and int(split_gt_files[1]) ==  int(split_conditioning_files[2]):
        # print(split_gt_files," ", split_conditioning_files)
        aligned_gt_files.append(gt_files[i])
        aligned_cond_files.append(cond_files[i])
    else:
        print(split_gt_files," ", split_conditioning_files)
        gt_files.pop(i)
        # break
 
 #shuffle arrays:
np.random.seed(1)
np.random.shuffle(aligned_gt_files)
np.random.seed(1)
np.random.shuffle(aligned_cond_files)

for i in range(len(aligned_cond_files)):
    print(aligned_cond_files[i], " ", aligned_gt_files[i])
# ##########################################################
# #Code for viewing test input and gt pointcloud
# ########################################################
# test_pm = np.load(gt_dir + aligned_gt_files[0])
# gt_points = utils.pointmap_to_pc(test_pm, voxel_size  = 0.1, x_y_bounds = [-2, 2], z_bounds = [-1.4, 0.9])
# conditioning_data = np.load(cond_dir + aligned_cond_files[0])
# print(conditioning_data.shape)

# pcd_local = o3d.geometry.PointCloud()
# pcd_conditioning = o3d.geometry.PointCloud()
# pcd_local.points = o3d.utility.Vector3dVector(gt_points)
# pcd_conditioning.points = o3d.utility.Vector3dVector(conditioning_data[:,0:3])
# o3d.visualization.draw_geometries([pcd_local,pcd_conditioning])
# ##############################################
#create data loaders
#############################################
# print(gt_data.shape)
# print(len(cond_files))
gt_dataloader = torch.utils.data.DataLoader(aligned_gt_files, batch_size=config.train_batch_size, shuffle=False)
conditioning_dataloader = torch.utils.data.DataLoader(aligned_cond_files, batch_size=config.train_batch_size, shuffle=False)


# ################################################
# # setup training stuff
# ################################################
optimizer = torch.optim.AdamW(list(model.parameters()) + list(conditioning_model.parameters()), lr=config.learning_rate)
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
bs = sample_noise_start.shape[0]
timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=sample_noise_start.device).long()

global_step = 0
dropout = 0.1
# Now you train the model
zeroed_conditioning = np.zeros((sample_conditioning.shape[1],sample_conditioning.shape[2]), dtype = np.single)
zeroed_conditioning = np.expand_dims(zeroed_conditioning, 0)
zeroed_conditioning = torch.tensor(zeroed_conditioning, dtype=torch.float)

lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(gt_dataloader) * config.num_epochs),
)

#make sure everything is on the GPU
model = model.to(torch_device)
conditioning_model = conditioning_model.to(torch_device)

for epoch in range(config.num_epochs):
    progress_bar = tqdm(total=len(gt_dataloader))
    progress_bar.set_description(f"Epoch {epoch}")

    for (step, batch), conditioning_batch in zip(enumerate(gt_dataloader),conditioning_dataloader):
        #####################################################
        #load the gt data
        ####################################################
        #initalize gt data cube:
        gt_data = np.load(gt_dir + batch[0])
        #add batch dim
        #remove top data
        gt_data = gt_data[None,0:13,:,:]

        for gt_file in batch[1:]:
            single_gt_data = np.load(gt_dir + gt_file)
            #add batch dim
            single_gt_data = single_gt_data[None,0:13,:,:]
            #append to data cube
            gt_data = np.append(gt_data, single_gt_data, axis = 0)
        gt_data = gt_data.astype(np.single)
        #gt_data to tensor:
        gt_data = torch.tensor(gt_data).to(torch_device)
        #######################3
        # turn gt data into the images
        #here what it does in the old scripts
        clean_images = gt_data
        # Sample noise to add to the images
        noise = torch.randn(clean_images.shape).to(clean_images.device)
        bs = clean_images.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device
        ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
        noisy_images = noisy_images.to(torch_device)
        #load the conditioning files
        #initalize gt data cube:
        cond_data = np.load(cond_dir + conditioning_batch[0])
        #transpose data for pointnet
        cond_data = cond_data.T
        #add batch dim
        cond_data = cond_data[None,:,:]
        #pass through pointnet
        # print(cond_data.shape)
        cond_tensor = conditioning_model(torch.tensor(cond_data.astype(np.single)).to(torch_device))
        # print(conditioning_model(torch.tensor(cond_data.astype(np.single))).shape)


        # since all the data is differnt sized we cant do it this way
        # we will just save the files
        for cond_file in conditioning_batch[1:]:
            single_cond_data = np.load(cond_dir + cond_file)
            #transpose data for pointnet
            single_cond_data = single_cond_data.T
            #add batch dim
            single_cond_data = single_cond_data[None,:,:]
            single_cond_tensor = conditioning_model(torch.tensor(single_cond_data.astype(np.single)).to(torch_device))
            #append to data cube
            cond_tensor = torch.cat((cond_tensor, single_cond_tensor), 0)
        # print(cond_tensor.shape)

        post_model_conditioning_batch = cond_tensor.swapaxes(1, 2)
        # Predict the noise residual
        #compute if one of the conditioning batches should be set to zeros
        n = random.sample(range(0,100), len(post_model_conditioning_batch))
        dropout_idx = [i for i,v in enumerate(n) if v < dropout*100]

        for i in dropout_idx:
            post_model_conditioning_batch[i] = zeroed_conditioning           
        post_model_conditioning_batch = post_model_conditioning_batch.to(torch_device)
        # print(noisy_images.dtype)
        # print(post_model_conditioning_batch.dtype)
        # print(noisy_images.shape)
        # print(post_model_conditioning_batch.shape)
        noise_pred = model(noisy_images, timesteps, encoder_hidden_states=post_model_conditioning_batch, return_dict=False)[0]
        loss = F.mse_loss(noise_pred, noise)
        loss.backward()
        # torch.nn.utils.clip_grad_norm(list(model.parameters()) + list(conditioning_model.parameters()),options['clip_gradient_norm'])

        #NEED TO ADD THIS 
        # accelerator.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        progress_bar.update(1)
        wandb.log({"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step, "epoch": epoch})

        global_step += 1
    
    print("\nPushing to Hub\n")
    model.push_to_hub("full_rgbd_unet_512_more_pointnet_short")
    torch.save(conditioning_model.state_dict(), "full_sim_pointnet_weights_more_pointnet_short/" + str(epoch))
    
#     # repo.push_to_hub(commit_message=f"Epoch {epoch}", blocking=True)
#     # conditioning_model.push_to_hub("diff_pointnet")
#     # repo.push_to_hub(commit_message=f"Epoch {epoch}", blocking=True)





