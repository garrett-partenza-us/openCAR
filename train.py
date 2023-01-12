from utils.dataloaders import BSDS300
from models.car import ResamplerNet, Downsampler
from models.espcnn import ESPCNN
from utils.loss import MSGE

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor, optim
from tqdm import tqdm

from itertools import chain
import random


if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("running on the GPU")
else:
    device = torch.device("cpu")
    print("running on the CPU")
    
    
SCALE = 2
KSIZE = 3
EPOCHS = 10
BATCH = 1
KH, KW, KC = 96, 96, 3
DH, DW, DC = 96, 96, 3
# UNFOLD_SHAPE = [BATCH, 480//96, 288//96, 1, KC, KH, KW]

kernel_generation_net = ResamplerNet(rgb_range=1.0)
downsampler_net = Downsampler(KSIZE, SCALE, BATCH)
upsampler_net = ESPCNN()

optimizer = optim.Adam(nn.ModuleList([kernel_generation_net, upsampler_net]).parameters(), lr=0.001)
mseloss = MSGE(batch=BATCH)
dataset = BSDS300(directory="/home/partenza.g/sr/benchmarks/BSDS300", transform=[lambda x: x/255.0], downscale=2, patch_size=96)


for epoch in (pbar := tqdm(range(EPOCHS))):
    
    images = dataset[random.choices(dataset.train_ids, k=BATCH)]
    images = Tensor(images['hr']).permute(0,3,1,2)
    images.requires_grad = False
    patches = images.unfold(1, KC, DC).unfold(2, KH, DH).unfold(3, KW, DW)
    patches.requires_grad = True
    unfold_shape = patches.size()
    patches = patches.contiguous().view(-1, KC, KH, DW)
    target = patches.clone()
    
    print("generating kernels...")
    kernels, offsets_x, offsets_y = kernel_generation_net(patches)
    print("downscaling...")
    downscaled_imgs = downsampler_net(patches, kernels, offsets_x, offsets_y)
    print("upscaling...")
    upscaled_imgs = upsampler_net(downscaled_imgs.permute(0,3,1,2))
    
#     patches_orig = upscaled_imgs.view(unfold_shape)
#     output_c = unfold_shape[1] * unfold_shape[4]
#     output_h = unfold_shape[2] * unfold_shape[5]
#     output_w = unfold_shape[3] * unfold_shape[6]
#     patches_orig = patches_orig.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
#     patches_orig = patches_orig.view(1, output_c, output_h, output_w)
    
    print("calculating loss...")
    loss, px_mse = mseloss(upscaled_imgs, target)
    
    print("backward pass...")
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(kernel_generation_net.parameters(), 1.0)
    torch.nn.utils.clip_grad_norm_(upsampler_net.parameters(), 1.0)
    print("stepping optimizer...")
    optimizer.step()
    
    px_mse = np.round(np.sqrt(px_mse)*255, 2)
    pbar.set_description("Loss: {}".format(px_mse))
    
#     if epoch%1000==0 and epoch!=0:
#         torch.save(model.state_dict(), "./models/model-{}.pt".format(epoch))

    del images, patches, unfold_shape, target, kernels, offsets_x, offsets_y, downscaled_imgs, upscaled_imgs, loss, px_mse