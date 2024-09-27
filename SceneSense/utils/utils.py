import copy
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
from scipy.spatial.transform import Rotation
from tqdm.auto import tqdm


def inverse_homogeneous_transform(matrix):
    if matrix.shape != (4, 4):
        raise ValueError("Input matrix must be a 4x4 numpy array")

    # Extract the rotation and translation components
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]

    # Compute the inverse of the rotation matrix
    inverse_rotation = np.transpose(rotation)

    # Compute the inverse translation
    inverse_translation = -np.dot(inverse_rotation, translation)

    # Construct the inverse homogeneous transform matrix
    inverse_matrix = np.eye(4)
    inverse_matrix[:3, :3] = inverse_rotation
    inverse_matrix[:3, 3] = inverse_translation

    return inverse_matrix


def homogeneous_transform(translation, rotation):
    """
    Generate a homogeneous transformation matrix from a translation vector
    and a quaternion rotation.

    Parameters:
    - translation: 1D NumPy array or list of length 3 representing translation along x, y, and z axes.
    - rotation: 1D NumPy array or list of length 4 representing a quaternion rotation.

    Returns:
    - 4x4 homogeneous transformation matrix.
    """

    # Ensure that the input vectors have the correct dimensions
    translation = np.array(translation, dtype=float)
    rotation = np.array(rotation, dtype=float)

    if translation.shape != (3,) or rotation.shape != (4,):
        raise ValueError(
            "Translation vector must be of length 3, and rotation quaternion must be of length 4."
        )

    # Normalize the quaternion to ensure it is a unit quaternion
    rotation /= np.linalg.norm(rotation)

    # Create a rotation matrix from the quaternion using scipy's Rotation class
    rotation_matrix = Rotation.from_quat(rotation).as_matrix()

    # Create a 4x4 homogeneous transformation matrix
    homogeneous_matrix = np.eye(4)
    homogeneous_matrix[:3, :3] = rotation_matrix
    homogeneous_matrix[:3, 3] = translation

    return homogeneous_matrix


def points_within_distance(x, y, points, distance):
    """
    Find all 3D points within a specified distance from a given (x, y) location.

    Parameters:
    - x, y: The x and y coordinates of the location.
    - points: NumPy array of shape (num_points, 3) representing 3D points.
    - distance: The maximum distance for points to be considered within.

    Returns:
    - NumPy array of points within the specified distance.
    """

    # Extract x, y coordinates from the 3D points
    # this actually needs to be xz
    xy_coordinates = points[:, [0, 2]]

    # Calculate the Euclidean distance from the given location to all points
    distances = np.linalg.norm(xy_coordinates - np.array([x, y]), axis=1)

    # Find indices of points within the specified distance
    within_distance_indices = np.where(distances <= distance)[0]

    # Extract points within the distance
    points_within_distance = points[within_distance_indices]

    return points_within_distance


def points_within_distance2(x, y, points, distance):
    """
    Find all 3D points within a specified distance from a given (x, y) location.

    Parameters:
    - x, y: The x and y coordinates of the location.
    - points: NumPy array of shape (num_points, 3) representing 3D points.
    - distance: The maximum distance for points to be considered within.

    Returns:
    - NumPy array of points within the specified distance.
    """

    # Extract x, y coordinates from the 3D points
    # this actually needs to be xz
    xy_coordinates = points[:, [0, 1]]

    # Calculate the Euclidean distance from the given location to all points
    distances = np.linalg.norm(xy_coordinates - np.array([x, y]), axis=1)

    # Find indices of points within the specified distance
    within_distance_indices = np.where(distances <= distance)[0]

    # Extract points within the distance
    points_within_distance = points[within_distance_indices]

    return points_within_distance


def get_pose_and_pc_from_running_txt(path, pose_num):
    f = open(path, "r")
    node_count = 0
    for x in f:
        if x[0:4] == "NODE":
            if node_count == pose_num + 1:
                break
            # if node_count != 0:
            # print(x)
            node_count += 1
            pose = x.split()
            # make empty pointcloud to fill
            pointcloud = np.zeros((3, 65536), dtype=np.single)
            pc_count = 0
        else:
            coord = np.fromstring(x, dtype=np.single, sep=" ")
            pointcloud[:, pc_count] = coord
            pc_count += 1
    # generate the transform matrix
    rot_vector = [float(pose[4]), float(pose[5][:-5]), 0.0]
    rot = Rotation.from_rotvec(rot_vector)
    # print(rot_vector)
    # print(rot.as_rotvec())
    trans = np.array([float(pose[1]), float(pose[2]), float(pose[3])])
    homo_trans_matrix = homogeneous_transform(trans, rot.as_quat())
    # print(trans)

    f.close()
    return homo_trans_matrix, pointcloud.T


def pc_to_pointmap(
    pointcloud, voxel_size=0.1, x_y_bounds=[-1.5, 1.5], z_bounds=[-1.4, 0.9]
):
    # takes a pointcloud and return a pointmap
    x_y_width = round((x_y_bounds[1] - x_y_bounds[0]) / voxel_size)
    z_width = round((z_bounds[1] - z_bounds[0]) / voxel_size)
    # change pointcloud start at zero (no negative number since pointmap indices are positive)
    pointcloud[:, 0] = pointcloud[:, 0] + abs(x_y_bounds[0])
    pointcloud[:, 1] = pointcloud[:, 1] + abs(z_bounds[0])
    pointcloud[:, 2] = pointcloud[:, 2] + abs(x_y_bounds[0])

    # create empty pointmap
    point_map = np.zeros((z_width, x_y_width, x_y_width), dtype=float)
    # compute which voxels to fill in the pointmap
    prec_vox_X = pointcloud[:, 0] / (x_y_bounds[1] - x_y_bounds[0])
    prec_vox_Y = pointcloud[:, 1] / (z_bounds[1] - z_bounds[0])
    prec_vox_Z = pointcloud[:, 2] / (x_y_bounds[1] - x_y_bounds[0])

    # for each point fill a point in the pointmap
    for idx, val in enumerate(prec_vox_X):
        point_map[
            math.floor(prec_vox_Y[idx] * z_width),
            math.floor(prec_vox_X[idx] * x_y_width),
            math.floor(prec_vox_Z[idx] * x_y_width),
        ] = 1.0

    return point_map


#####THere is something here with the threshold for prediction that was making our predictions worse!
def pointmap_to_pc(
    pointmap,
    voxel_size=0.1,
    x_y_bounds=[-1.5, 1.5],
    z_bounds=[-1.4, 0.9],
    prediction_thresh=0.8,
):
    # setup empty pointcloud
    arr = np.empty((0, 3), np.single())
    # reads in a pointmap and returns a pointcloud
    for y_idx, y_val in enumerate(pointmap):
        for x_idx, x_val in enumerate(y_val):
            for z_idx, z_val in enumerate(x_val):
                if z_val > prediction_thresh:
                    arr = np.append(
                        arr,
                        np.array(
                            [
                                [
                                    x_idx * voxel_size + voxel_size / 2,
                                    y_idx * voxel_size + voxel_size / 2,
                                    z_idx * voxel_size + voxel_size / 2,
                                ]
                            ]
                        ),
                        axis=0,
                    )

    arr[:, 0] = arr[:, 0] - abs(x_y_bounds[0])
    arr[:, 1] = arr[:, 1] - abs(z_bounds[0])
    arr[:, 2] = arr[:, 2] - abs(x_y_bounds[0])

    return arr


def denoise_guided_inference(
    model,
    noise_scheduler,
    conditioning,
    width,
    torch_device="cuda:1",
    denoising_steps=50,
    guidance_scale=15,
    sample_batch_size=1,
):
    # set the timesteps for denoiseing
    noise_scheduler.set_timesteps(denoising_steps)

    # generate noise to be diffused
    generator = torch.manual_seed(0)  # Seed generator to create the inital latent noise
    latents = torch.randn(
        (sample_batch_size, model.config.in_channels, width, width), generator=generator
    )
    # latents = latents.to(torch_device)
    latents = latents * noise_scheduler.init_noise_sigma

    # set sample conditioning, concating uncondtioned noise so we only need to do one forward pass
    uncond_embeddings = torch.zeros(conditioning.shape[1], conditioning.shape[2])
    uncond_embeddings = uncond_embeddings[None, :, :]
    # uncond_embeddings = uncond_embeddings.to(torch_device)
    print(conditioning.shape)
    print(uncond_embeddings.shape)
    sample_conditioning = torch.cat(
        [uncond_embeddings, torch.tensor(conditioning, dtype=torch.float)]
    )

    # generate progress bar
    progress_bar = tqdm(total=len(noise_scheduler.timesteps))
    progress_bar.set_description(f"diffusion step: ")
    # perform diffusion
    for t in tqdm(noise_scheduler.timesteps):
        #     # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = noise_scheduler.scale_model_input(
            latent_model_input, timestep=t
        )
        # predict the noise residual
        with torch.no_grad():
            noise_pred = model(
                latent_model_input, t, encoder_hidden_states=sample_conditioning
            ).sample
        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )
        # compute the previous noisy sample x_t -> x_t-1
        latents = noise_scheduler.step(noise_pred, t, latents).prev_sample
        progress_bar.update(1)
    return latents


def inpainting_pointmaps(
    model,
    noise_scheduler,
    conditioning,
    width,
    inpainting_target,
    torch_device="cpu",
    denoising_steps=50,
    guidance_scale=3,
    sample_batch_size=1,
):
    # set up initialize noise scheudler and noise to be operated on
    noise_scheduler.set_timesteps(denoising_steps)
    noise = torch.randn(inpainting_target.shape)
    # load the inpainting target
    voxel_grid = torch.tensor(inpainting_target, dtype=torch.float)
    # get the coordinates of the occupied voxels
    input_coordinate = np.where(voxel_grid > 0.9)
    print(input_coordinate)
    # print("shape: ", input_coordinate.shape)

    # generate noise to be diffused
    generator = torch.manual_seed(0)  # Seed generator to create the inital latent noise

    # set sample conditioning, concating uncondtioned noise so we only need to do one forward pass
    uncond_embeddings = torch.zeros(conditioning.shape[1], conditioning.shape[2])
    uncond_embeddings = uncond_embeddings[None, :, :]
    # uncond_embeddings = uncond_embeddings.to(torch_device)
    sample_conditioning = torch.cat(
        [
            uncond_embeddings,
            torch.tensor(conditioning, dtype=torch.float).to(torch_device),
        ]
    )

    # generate progress bar
    # perform diffusion
    # print(noise_scheduler.timesteps)
    generator = torch.manual_seed(0)
    latents = torch.randn(
        (sample_batch_size, model.config.in_channels, width, width), generator=generator
    )
    # latents = latents.to(torch_device)
    latents = latents * noise_scheduler.init_noise_sigma
    for t in tqdm(noise_scheduler.timesteps):
        # get the noisy scan points
        # this adds noise to voxel grid which is our conditioning targets
        noisy_images = noise_scheduler.add_noise(
            voxel_grid, noise, timesteps=torch.tensor([t.item()])
        )

        # print(latents.shape)
        # add in the noise image wehre the input scans are
        # replace the data with the overwrited noisified current octomap image

        print(latents.shape)

        # now we just iterate through all the coordinates,eveywhere we have a cordinate we put in the noisy new oocumancy
        for idx, z_val in enumerate(input_coordinate[0]):
            # we are iterating though the tuple using the first coord
            x_val = input_coordinate[1][idx]
            y_val = input_coordinate[2][idx]
            # print(latents.shape)
            # print(noisy_images.shape)
            # print(z_val, x_val, y_val)
            # replace the latent value with the new noisified input value
            latents[0][z_val, x_val, y_val] = noisy_images[z_val, x_val, y_val]

        #     break
        # break
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
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )
        # compute the previous noisy sample x_t -> x_t-1
        latents = noise_scheduler.step(noise_pred, t, latents).prev_sample
    # one more inpainting step
    # for idx, z_val in enumerate(input_coordinate[0]):
    #     #we are iterating though the tuple using the first coord
    #     x_val = input_coordinate[1][idx]
    #     y_val = input_coordinate[2][idx]
    #     # print(latents.shape)
    #     # print(noisy_images.shape)
    #     # print(z_val, x_val, y_val)
    #     #replace the latent value with the new noisified input value
    #     latents[0][z_val, x_val, y_val] = noisy_images[z_val, x_val, y_val]
    return latents


def get_IoU(gt, prediction):
    prediction = np.rint(prediction)
    intsection = np.logical_and(gt, prediction)
    overlap = np.logical_or(gt, prediction)
    return np.count_nonzero(intsection) / np.count_nonzero(overlap)


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
    # set up initialize noise scheudler and noise to be operated on
    noise_scheduler.set_timesteps(denoising_steps)
    noise = torch.randn(inpainting_target.shape)
    # load the inpainting target
    voxel_grid = torch.tensor(inpainting_target, dtype=torch.float)
    # get the coordinates of the occupied voxels
    input_coordinate = np.where(voxel_grid > 0.9)
    print(input_coordinate)
    # get the coordinates of the unoccupied voxels
    unnoc_grid = torch.tensor(inpainting_unocc, dtype=torch.float)
    unnoc_coordinate = np.where(unnoc_grid > 0.9)
    # print("shape: ", input_coordinate.shape)

    # generate noise to be diffused
    generator = torch.manual_seed(0)  # Seed generator to create the inital latent noise

    # set sample conditioning, concating uncondtioned noise so we only need to do one forward pass
    uncond_embeddings = torch.zeros(conditioning.shape[1], conditioning.shape[2])
    uncond_embeddings = uncond_embeddings[None, :, :]
    # uncond_embeddings = uncond_embeddings.to(torch_device)
    sample_conditioning = torch.cat(
        [
            uncond_embeddings,
            torch.tensor(conditioning, dtype=torch.float).to(torch_device),
        ]
    )

    # generate progress bar
    # perform diffusion
    # print(noise_scheduler.timesteps)
    generator = torch.manual_seed(0)
    latents = torch.randn(
        (sample_batch_size, model.config.in_channels, width, width), generator=generator
    )
    # latents = latents.to(torch_device)
    latents = latents * noise_scheduler.init_noise_sigma
    for t in tqdm(noise_scheduler.timesteps):
        # get the noisy scan points
        # this adds noise to voxel grid which is our conditioning targets
        noisy_images = noise_scheduler.add_noise(
            voxel_grid, noise, timesteps=torch.tensor([t.item()])
        )
        # WE ACTUALLY NEED TO TURN THIS TO Zeros so we do 1 -
        # I think this 1- operation might mess it up
        noisy_unoc_images = noise_scheduler.add_noise(
            1 - unnoc_grid, noise, timesteps=torch.tensor([t.item()])
        )
        # print(latents.shape)
        # add in the noise image wehre the input scans are
        # replace the data with the overwrited noisified current octomap image

        print(latents.shape)

        # now we just iterate through all the coordinates,eveywhere we have a cordinate we put in the noisy new oocumancy
        for idx, z_val in enumerate(input_coordinate[0]):
            # we are iterating though the tuple using the first coord
            x_val = input_coordinate[1][idx]
            y_val = input_coordinate[2][idx]
            # print(latents.shape)
            # print(noisy_images.shape)
            # print(z_val, x_val, y_val)
            # replace the latent value with the new noisified input value
            latents[0][z_val, x_val, y_val] = noisy_images[z_val, x_val, y_val]
        # also iterate through all the coordinates and input our freespace
        for idx, z_val in enumerate(unnoc_coordinate[0]):
            # we are iterating though the tuple using the first coord
            x_val = unnoc_coordinate[1][idx]
            y_val = unnoc_coordinate[2][idx]
            # print(latents.shape)
            # print(noisy_images.shape)
            # print(z_val, x_val, y_val)
            # replace the latent value with the new noisified input value
            latents[0][z_val, x_val, y_val] = noisy_unoc_images[z_val, x_val, y_val]
        # EXPECT THESE
        # break
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
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )
        # compute the previous noisy sample x_t -> x_t-1
        latents = noise_scheduler.step(noise_pred, t, latents).prev_sample

        # Perform MCMC sampling (ULA for simplicity)
        std = (2 * lambda_) ** 0.5
        for _ in range(mcmc_steps):
            # Calculate the gradient of log-probability (score) from the model's output
            with torch.no_grad():
                new_pred = model(latents, t).sample
            noise_MCMC = torch.randn_like(new_pred) * std  # (B, 3, H, W)
            latents = latents + new_pred * lambda_ * noise_MCMC
            # print(latents.shape)
            # gradient = noise_scheduler.scale_model_input(new__pred, t)

            # # Langevin step: gradient ascent with noise
            # latents = latents + 0.5 * step_size * gradient + torch.sqrt(step_size) * torch.randn_like(latents)

        ############################################
        # just for viewing the outputs
        ###############################################
        # inpained_points = pointmap_to_pc(latents[0],
        #                                  voxel_size = 0.1,
        #                                  x_y_bounds = [-2, 2],
        #                                   z_bounds = [-1.4, 0.9])
        # pcd_inpaint = o3d.geometry.PointCloud()

        # # print("inpainted shape: ", inpained_points.shape)
        # pcd_inpaint.points = o3d.utility.Vector3dVector(inpained_points)
        # colors = np.zeros((len(np.asarray(pcd_inpaint.points)), 3))
        # R = pcd_inpaint.get_rotation_matrix_from_xyz((np.pi/2, 0, 0))
        # pcd_inpaint.rotate(R, center=(0, 0, 0))
        # # colors[:,0] = 1
        # # colors[:,1] = 0
        # # colors[:,2] = 0
        # # pcd_inpaint.colors = o3d.utility.Vector3dVector(colors)
        # # pcd_inpaint.transform(hm_tx_mat)
        # o3d.visualization.draw_geometries([pcd_inpaint])

    # one more inpainting step
    for idx, z_val in enumerate(input_coordinate[0]):
        # we are iterating though the tuple using the first coord
        x_val = input_coordinate[1][idx]
        y_val = input_coordinate[2][idx]
        # print(latents.shape)
        # print(noisy_images.shape)
        # print(z_val, x_val, y_val)
        # replace the latent value with the new noisified input value
        latents[0][z_val, x_val, y_val] = voxel_grid[z_val, x_val, y_val]
    # also iterate through all the coordinates and input our freespace
    for idx, z_val in enumerate(unnoc_coordinate[0]):
        # we are iterating though the tuple using the first coord
        x_val = unnoc_coordinate[1][idx]
        y_val = unnoc_coordinate[2][idx]
        # print(latents.shape)
        # print(noisy_images.shape)
        # print(z_val, x_val, y_val)
        # replace the latent value with the new noisified input value
        latents[0][z_val, x_val, y_val] = 1 - unnoc_grid[z_val, x_val, y_val]
    return latents
