import copy
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
from scipy.spatial.transform import Rotation
from tqdm.auto import tqdm

from SceneSense.utils.utils import homogeneous_transform, inverse_homogeneous_transform, pc_to_pointmap


def inpainting_pointmaps_w_freespace(
    model,
    noise_scheduler,
    conditioning,
    width,
    inpainting_target,
    inpainting_unocc,
    torch_device="cpu",
    denoising_steps=30,
    guidance_scale=3,
    sample_batch_size=1,
    mcmc_steps: int = 0,
    lambda_: float = 0.5,
):
    # initialize noise scheudler and noise to be operated on
    noise_scheduler.set_timesteps(denoising_steps)

    # create noise input
    noise = torch.randn(inpainting_target.shape)

    # inpainting target for occupied voxels as torch tensor
    inpainting_target_torch = torch.tensor(inpainting_target, dtype=torch.float)

    # get the coordinates of the unoccupied voxels
    inpainting_target_torch_unocc = torch.tensor(inpainting_unocc, dtype=torch.float)

    # generate noise to be diffused
    generator = torch.manual_seed(0)  # Seed generator to create the inital latent noise

    # set sample conditioning, concating uncondtioned noise so we only need to do one forward pass
    uncond_embeddings = torch.zeros(conditioning.shape[1], conditioning.shape[2])
    uncond_embeddings = uncond_embeddings[None, :, :]
    sample_conditioning = torch.cat(
        [
            uncond_embeddings,
            torch.tensor(conditioning, dtype=torch.float).to(torch_device),
        ]
    )

    # generate model latents
    generator = torch.manual_seed(0)
    latents = torch.randn(
        (sample_batch_size, model.config.in_channels, width, width), generator=generator
    )
    latents = latents * noise_scheduler.init_noise_sigma

    # perform diffusion
    for t in tqdm(noise_scheduler.timesteps):

        # add noise to inpainting data to match the current timestep
        noisy_images = noise_scheduler.add_noise(
            inpainting_target_torch, noise, timesteps=torch.tensor([t.item()])
        )
        noisy_unoc_images = noise_scheduler.add_noise(
            1 - inpainting_target_torch_unocc, noise, timesteps=torch.tensor([t.item()])
        )


        # do inpainting
        latents[0][inpainting_target_torch > 0.9] = noisy_images[
            inpainting_target_torch > 0.9
        ]
        latents[0][inpainting_target_torch_unocc > 0.9] = noisy_unoc_images[
            inpainting_target_torch_unocc > 0.9
        ]

        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = noise_scheduler.scale_model_input(
            latent_model_input, timestep=t
        )
        latent_model_input = latent_model_input.to(torch_device)
        # predict the noise residual
        with torch.no_grad():
            noise_pred = model(
                latent_model_input, t, encoder_hidden_states=sample_conditioning
            ).sample
        # perform guidance
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_cond - noise_pred_uncond
        )
        # compute the previous noisy sample x_t -> x_t-1
        latents = noise_scheduler.step(noise_pred, t, latents).prev_sample # \mu (x_t)

        std = (2 * lambda_) ** 0.5
        for _ in range(mcmc_steps):
            # Calculate the gradient of log-probability (score) from the model's output
            with torch.no_grad():
                mcmc_latents  = torch.cat([latents] * 2)
                mcmc_latents = noise_scheduler.scale_model_input(
                    mcmc_latents, timestep=t-1
                )
                new_pred = model(
                    mcmc_latents, t-1, encoder_hidden_states=sample_conditioning
                ).sample # x_t-1
            # perform guidance
            mcmc_noise_pred_uncond, mcmc_noise_pred_cond = new_pred.chunk(2)
            score_func = mcmc_noise_pred_uncond + guidance_scale * (
                mcmc_noise_pred_cond - mcmc_noise_pred_uncond
            )
            score_func = (1 / np.sqrt(1 - noise_scheduler.alphas_cumprod[t-1])) * score_func # s_theta (x_t-1)


            noise_MCMC = torch.randn_like(score_func) * std  # (B, 3, H, W)
            latents = (latents + score_func * lambda_) + noise_MCMC

    # inpaint again
    latents[0][inpainting_target_torch > 0.9] = noisy_images[
        inpainting_target_torch > 0.9
    ]
    latents[0][inpainting_target_torch_unocc > 0.9] = noisy_unoc_images[
        inpainting_target_torch_unocc > 0.9
    ]

    return latents
