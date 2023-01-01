print("importing libraries...")
from models.car import ResamplerNet
from models.espcnn import ESPCNN
from models.adaptive_gridsampler.gridsampler import Downsampler
import torch
from math import floor

SCALE = 2
KSIZE = 3 
kernel_generation_net = ResamplerNet(rgb_range=1.0)

x = torch.rand((1,3,96,96))
kernels, offsets_h, offsets_v = kernel_generation_net(x)

# first batch and permute kernel last
kernels, offsets_h, offsets_v = kernels[0], offsets_h[0], offsets_v[0]
kernels, offsets_h, offsets_v = kernels.permute(1,2,0), offsets_h.permute(1,2,0), offsets_v.permute(1,2,0)
kernels, offsets_h, offsets_v = kernels.reshape(*kernels.size()[:-1], 3, 3), offsets_h.reshape(*offsets_h.size()[:-1], 3, 3), offsets_v.reshape(*offsets_v.size()[:-1], 3, 3)

# original hr image
hr = x[0]

for x in range(kernels.shape[0]):
    for y in range(kernels.shape[1]):
        lr_pix = torch.zeros(3)
        u, v = x+0.5*SCALE-0.5, y+0.5*SCALE-0.5
        for c in range(3):
            for i in range(KSIZE):
                for j in range(KSIZE):
                    kernel_val = kernels[x][y][i][j]
                    k, q = u+i-(KSIZE/2)+offsets_h[x][y][i][j], v+j-(KSIZE/2)+offsets_v[x][y][i][j]
                    
                    # replace integer conversion with bilinear interpolation
                    k, q = int(k), int(q)
                    
                    hr_pix = hr[c][q][k]
                    lr_pix[c]+=kernel_val*hr_pix



