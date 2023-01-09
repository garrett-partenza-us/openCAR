from utils.dataloaders import BSDS300
from models.car import ResamplerNet
from models.adaptive_gridsampler.gridsampler import Downsampler
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
UNFOLD_SHAPE = [BATCH, 480//16, 320//16, 1, KH*2, KW*2, KC]

kernel_generation_net = ResamplerNet(rgb_range=1.0)
downsampler_net = Downsampler(SCALE, KSIZE)
upsampler_net = ESPCNN()

optimizer = optim.Adam(nn.ModuleList([kernel_generation_net, upsampler_net]).parameters(), lr=0.001)
mseloss = MSGE(batch=BATCH)
dataset = BSDS300(directory="/home/partenza.g/sr/benchmarks/BSDS300", transform=[lambda x: x/255.0], downscale=2, patch_size=16)


for epoch in (pbar := tqdm(range(EPOCHS))):
    
    images = dataset[random.choices(dataset.train_ids, k=BATCH)]
    images = Tensor(images['hr']).permute(0,3,1,2)
    images.requires_grad = True
    
#     images = images.unfold(1, KH, DH).unfold(2, KW, DW).unfold(3, KC, DC)
#     images = images.contiguous().view(-1, KH, KW, KC).permute(0,3,1,2)
    
    kernels, offsets_h, offsets_v = kernel_generation_net(images)
    downscaled_imgs = downsampler_net(images, kernels, offsets_h, offsets_v, SCALE)
    upscaled_imgs = upsampler_net(downscaled_imgs)
    
#     upscaled_imgs = upscaled_imgs.view(UNFOLD_SHAPE)
#     output_c = UNFOLD_SHAPE[1] * UNFOLD_SHAPE[4]
#     output_h = UNFOLD_SHAPE[2] * UNFOLD_SHAPE[5]
#     output_w = UNFOLD_SHAPE[3] * UNFOLD_SHAPE[6]
#     upscaled_imgs = upscaled_imgs.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
#     upscaled_imgs = upscaled_imgs.view(BATCH, output_c, output_h, output_w)
    
    loss, px_mse = mseloss(upscaled_imgs, images)
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(kernel_generation_net.parameters(), 1.0)
    torch.nn.utils.clip_grad_norm_(upsampler_net.parameters(), 1.0)
    optimizer.step()
    
    px_mse = np.round(np.sqrt(px_mse)*255, 2)
    pbar.set_description("Loss: {}".format(px_mse))
    
#     if epoch%1000==0:
#         torch.save(model.state_dict(), "./models/model-{}.pt".format(epoch))

    del images, kernels, offsets_h, offsets_v, downscaled_imgs, upscaled_imgs, loss, px_mse
    

    

# DEAD CODE 
    
# kh, kw, kc = 16, 16, 3
# dh, dw, dc = 16, 16, 3
# unfold_shape = [batch, 480//16, 320//16, 1, kw*2, kh*2, kc]
#     x = x.unfold(1, kh, dh).unfold(2, kw, dw).unfold(3,kc, dc)
#     x = x.contiguous().view(-1, kh, kw, kc)
#     y = Tensor(images['hr']).to(device)
#     out = out.view(unfold_shape)
#     output_c = unfold_shape[1] * unfold_shape[4]
#     output_h = unfold_shape[2] * unfold_shape[5]
#     output_w = unfold_shape[3] * unfold_shape[6]
#     out = out.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
#     out = out.view(batch, output_c, output_h, output_w)
#     
