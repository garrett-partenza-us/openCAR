print("importing libraries...")
from models.car import ResamplerNet
from models.espcnn import ESPCNN
from models.adaptive_gridsampler.gridsampler import Downsampler
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SCALE = 2
KSIZE = 3
kernel_generation_net = ResamplerNet(rgb_range=1.0)
downsampler_net = Downsampler(SCALE, KSIZE)
upsampler_net = ESPCNN()


kernels, offsets_h, offsets_v = kernel_generation_net(x)
downscaled_img = downsampler_net(x, kernels, offsets_h, offsets_v, SCALE)
downscaled_img = torch.clamp(downscaled_img, 0, 1)
downscaled_img = torch.round(downscaled_img * 255)
out = upsampler_net(downscaled_img)
