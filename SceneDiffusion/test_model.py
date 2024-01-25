from dataclasses import dataclass
from diffusers import UNet2DConditionModel
import torch
from diffusers import DDPMScheduler
import torch.nn.functional as F

@dataclass
class TrainingConfig:
    image_size = 128  # the generated image resolution
    train_batch_size = 16
    eval_batch_size = 16  # how many images to sample during evaluation
    num_epochs = 500
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
    sample_size=96,  # the target image resolution
    in_channels=1,  # the number of input channels, 3 for RGB images
    out_channels=1,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    cross_attention_dim=3, #the dim of the guidance data?
    block_out_channels=(128, 256, 256),  # the number of output channels for each UNet block
    down_block_types=(
        "CrossAttnDownBlock2D",  # a regular ResNet downsampling block
        "CrossAttnDownBlock2D",
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",  # a regular ResNet upsampling block
        "CrossAttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "CrossAttnUpBlock2D",
    ),

)

#test conditioning Model
conditioning_model =  torch.nn.Linear(20, 3)
#test inputs
input = torch.randn(5, 20)
test_labels = torch.randn(20, 5)
#passing data through network
test_out = conditioning_model(input)
print(test_out.shape)



# #define optimizer
# optimizer = torch.optim.SGD(conditioning_model.parameters(), lr=0.001, momentum=0.9)
# # define loss function
# loss_fn = torch.nn.CrossEntropyLoss()

#here is a training loop that works for the linear layer
# for i in range(10):
#     output = conditioning_model(input)
#     loss = loss_fn(output, test_labels)
#     print(loss.item())
#     #compute gradients
#     loss.backward()
#     #backprop
#     optimizer.step()

#Now we make a simple test for the unet
sample_noise_start = torch.randn(1,1,96, 96)
sample_noise_target = torch.randn(1,1,96, 96)
sample_conditioning = torch.randn(1,5,3)
print("Unet output shape:", model(sample_noise_start, timestep=1.0, encoder_hidden_states=sample_conditioning).sample.shape)

#set up necessary stuff for unet
#Wait this works?
#this for sure works
optimizer = torch.optim.SGD(list(model.parameters()) + list(conditioning_model.parameters()), lr=0.001, momentum=0.9)
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
bs = sample_noise_start.shape[0]
timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=sample_noise_start.device).long()

#simple training loop test
for i in range(10):
    sample_conditioning = conditioning_model(input)
    sample_conditioning = sample_conditioning[None,:,:]
    print(sample_conditioning)
    noise_pred = model(sample_noise_start, timesteps, encoder_hidden_states=sample_conditioning, return_dict=False)[0]
    loss = F.mse_loss(noise_pred, sample_noise_target)
    print(loss.item())
    loss.backward()
    #backprop
    optimizer.step()



