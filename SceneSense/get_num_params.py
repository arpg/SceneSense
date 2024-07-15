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
from scipy.spatial.transform import Rotation

#load the models
model = UNet2DConditionModel.from_pretrained("alre5639/diff_unet_512_arpg")
conditioning_model = get_model()
conditioning_model.load_state_dict(torch.load("/home/arpg/Documents/SceneDiffusion/rgbd_cond_model_weights/cond_model" + str(249)))

num_params: int = sum(p.numel() for p in model.parameters() if p.requires_grad)
num_params_pn: int = sum(p.numel() for p in conditioning_model.parameters() if p.requires_grad)
print("number of trainable unet params", print(num_params + num_params_pn))

IoU = np.loadtxt("data/rgbd_512_out/IoU_arr.txt")
print(np.mean(IoU))