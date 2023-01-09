print("importing libraries...")
from models.car import ResamplerNet
from models.espcnn import ESPCNN
from models.adaptive_gridsampler.gridsampler import Downsampler
import torch
from math import floor
from tqdm import tqdm
from models.espcnn import ESPCNN
from utils.loss import MSGE
from torch import optim
import torch.nn as nn

SCALE = 2
KSIZE = 3 
batch = 1
images = torch.rand((batch,3,96,96))
images.requires_grad=True

dtype = torch.FloatTensor
dtype_long = torch.LongTensor

mseloss = MSGE(batch=batch)
upsampler_net = ESPCNN()
kernel_generation_net = ResamplerNet(rgb_range=1.0)
optimizer = optim.Adam(nn.ModuleList([kernel_generation_net, upsampler_net]).parameters(), lr=0.001)

# https://gist.github.com/peteflorence/a1da2c759ca1ac2b74af9a83f69ce20e
def bilinear_interpolate_torch(im, x, y):
    
    # Channel last
    im = im.permute(1,2,0)
    x0 = torch.floor(x)
    x1 = x0 + 1
    
    y0 = torch.floor(y)
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, im.shape[1]-1)
    x1 = torch.clamp(x1, 0, im.shape[1]-1)
    y0 = torch.clamp(y0, 0, im.shape[0]-1)
    y1 = torch.clamp(y1, 0, im.shape[0]-1)
    
    Ia = im[ y0, x0 ][0]
    Ib = im[ y1, x0 ][0]
    Ic = im[ y0, x1 ][0]
    Id = im[ y1, x1 ][0]
    
    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    return torch.t((torch.t(Ia)*wa)) + torch.t(torch.t(Ib)*wb) + torch.t(torch.t(Ic)*wc) + torch.t(torch.t(Id)*wd)

kernels, offsets_h, offsets_v = kernel_generation_net(images)

# first batch and permute kernel last
kernels, offsets_h, offsets_v = kernels[0], offsets_h[0], offsets_v[0]
kernels, offsets_h, offsets_v = kernels.permute(1,2,0), offsets_h.permute(1,2,0), offsets_v.permute(1,2,0)
kernels, offsets_h, offsets_v = kernels.reshape(*kernels.size()[:-1], 3, 3), offsets_h.reshape(*offsets_h.size()[:-1], 3, 3), offsets_v.reshape(*offsets_v.size()[:-1], 3, 3)

# original hr image
out = []
for batch in images.shape[0]:
    hr = hr[batch]
    lr_pixels = []
    for x in tqdm(range(kernels.shape[0])):
        for y in tqdm(range(kernels.shape[1]), desc=" inner loop", position=1, leave=False):
            u, v = x+0.5*SCALE-0.5, y+0.5*SCALE-0.5
            for c in range(3):
                res = torch.zeros(1)
                for i in range(KSIZE):
                    for j in range(KSIZE):
                        kernel_val = kernels[x][y][i][j]
                        k, q = u+i-(KSIZE/2)+offsets_h[x][y][i][j], v+j-(KSIZE/2)+offsets_v[x][y][i][j]
                        hr_pix = bilinear_interpolate_torch(torch.unsqueeze(hr[c], 0), k, q)
                        res+=kernel_val*hr_pix
                lr_pixels.append(res)
                    
    lr = torch.stack(lr_pixels, 0).reshape(1,3,48,48)
    out.append(lr)
out = torch.stack(out, 0)

out = upsampler_net(lr)
out = torch.clamp(out, 0, 1)
out = torch.round(out * 255)

loss, px_mse = mseloss(out, images)

loss.backward()

optimizer.step()