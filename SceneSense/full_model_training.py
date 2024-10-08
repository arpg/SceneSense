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
from tqdm.auto import tqdm
import wandb
import random
from huggingface_hub import login
from diffusers.optimization import get_cosine_schedule_with_warmup

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
    lr_warmup_steps = 100
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
    sample_size=30,  # the target image resolution
    in_channels=22,  # the number of input channels, 3 for RGB images
    out_channels=22,  # the number of output channels
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


########################
#get gt data
#########################3
gt_dir = "/home/arpg/Documents/open3d_from_habitat/training_pointmaps/"
gt_files = natsorted(os.listdir(gt_dir))

#initalize gt data cube:
gt_data = np.load(gt_dir + gt_files[0])
#get into correct shape 
gt_data = np.transpose(gt_data, (2,0,1))
#add batch dim
gt_data = gt_data[None,:,:,:]

for gt_file in gt_files:
    single_gt_data = np.load(gt_dir + gt_file)
    #get into correct shape
    single_gt_data = np.transpose(single_gt_data, (2,0,1))
    #add batch dim
    single_gt_data = single_gt_data[None,:,:,:]
    #append to data cube
    gt_data = np.append(gt_data, single_gt_data, axis = 0)
print(gt_data.shape)
gt_data = gt_data.astype(np.single)
##################################
#get the conditioning data
##################################
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
        elif node_count == gt_data.shape[0] + 1:
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
print(final_pointcloud.shape)

#shuffle arrays:
np.random.seed(1)
np.random.shuffle(gt_data)
np.random.seed(1)
np.random.shuffle(final_pointcloud)


# ##########################################################
# #Code for viewing test input and gt pointcloud
# ########################################################
# test_pc = final_pointcloud[0].T
# print(test_pc.shape)
# pcd_local = o3d.geometry.PointCloud()
# pcd_local.points = o3d.utility.Vector3dVector(test_pc)
# o3d.visualization.draw_geometries([pcd_local])

# #view gt data
# point_map = gt_data[0]
# arr = np.empty((0,3), float)
# for idx_x,x in enumerate(point_map):
#     for idx_y, y in enumerate(x):
#         for idx_z, z in enumerate(y):
#             if z == 1:
#                 arr = np.append(arr, np.array([[idx_x,idx_y,idx_z]]), axis=0)


# pcd_local = o3d.geometry.PointCloud()
# pcd_local.points = o3d.utility.Vector3dVector(arr)
# o3d.visualization.draw_geometries([pcd_local])
# ##############################################
#create data loaders
#############################################
gt_dataloader = torch.utils.data.DataLoader(gt_data, batch_size=config.train_batch_size, shuffle=False)
conditioning_dataloader = torch.utils.data.DataLoader(final_pointcloud, batch_size=config.train_batch_size, shuffle=False)


################################################
# setup training stuff
################################################
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

# model = model.to(torch_device)
# conditioning_model = conditioning_model.to(torch_device)

for epoch in range(config.num_epochs):
    progress_bar = tqdm(total=len(gt_dataloader))
    progress_bar.set_description(f"Epoch {epoch}")

    for (step, batch), conditioning_batch in zip(enumerate(gt_dataloader),conditioning_dataloader):
        
        
        #here what it does in the old scripts
        clean_images = batch
        # clean_images = clean_images.to(torch_device)
        # Sample noise to add to the images
        noise = torch.randn(clean_images.shape).to(clean_images.device)
        # noise = noise.to(torch_device)
        bs = clean_images.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device
        ).long()

        # timesteps = timesteps.to(torch_device)

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

        #get the conditioning data
        # print(conditioning_batch.shape)
        post_model_conditioning_batch = conditioning_model(conditioning_batch)
        post_model_conditioning_batch = post_model_conditioning_batch.swapaxes(1, 2)
        # Predict the noise residual
        #compute if one of the conditioning batches should be set to zeros
        n = random.sample(range(0,100), len(post_model_conditioning_batch))
        dropout_idx = [i for i,v in enumerate(n) if v < dropout*100]

        for i in dropout_idx:
            post_model_conditioning_batch[i] = zeroed_conditioning           

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
    model.push_to_hub("diff_unet_512_arpg")
    torch.save(conditioning_model.state_dict(), "/home/arpg/Documents/SceneDiffusion/full_conditioning_weights/full_cond_model" + str(epoch))
    
    # repo.push_to_hub(commit_message=f"Epoch {epoch}", blocking=True)
    # conditioning_model.push_to_hub("diff_pointnet")
    # repo.push_to_hub(commit_message=f"Epoch {epoch}", blocking=True)






# #simple training loop test
# for i in range(10):
#     sample_conditioning = conditioning_model(input)
#     sample_conditioning = sample_conditioning[None,:,:]
#     print(sample_conditioning)
#     noise_pred = model(sample_noise_start, timesteps, encoder_hidden_states=sample_conditioning, return_dict=False)[0]
#     loss = F.mse_loss(noise_pred, sample_noise_target)
#     print(loss.item())
#     loss.backward()
#     #backprop
#     optimizer.step()



