from utils.dataloaders import BSDS300
from models.car import ResamplerNet
from models.espcnn import ESPCNN
from models.adaptive_gridsampler.gridsampler import Downsampler
from torch import optim
from torch.nn import MSELoss
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm
import random
import torch
import numpy as np
from utils.loss import MSGE
import random
from itertools import chain

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("running on the GPU")
else:
    device = torch.device("cpu")
    print("running on the CPU")
    
SCALE = 2
KSIZE = 3

kernel_generation_net = ResamplerNet(rgb_range=1.0)
downsampler_net = Downsampler(SCALE, KSIZE)
upsampler_net = ESPCNN()

optimizer = optim.Adam(nn.ModuleList([kernel_generation_net, downsampler_net, upsampler_net]).parameters(), lr=0.001)

dataset = BSDS300(directory="/home/partenza.g/sr/benchmarks/BSDS300", transform=[lambda x: x/255.0], downscale=2, patch_size=16)

epochs = 1
batch = 1
mseloss = MSGE(batch=batch).cpu()
# kh, kw, kc = 16, 16, 3
# dh, dw, dc = 16, 16, 3
# unfold_shape = [batch, 480//16, 320//16, 1, kw*2, kh*2, kc]

for epoch in (pbar := tqdm(range(epochs))):
    
    images = dataset[random.choices(dataset.train_ids, k=batch)]
    
    x = Tensor(images['hr'])
    x.requires_grad = True
    x = x.permute(0,3,1,2)
    
#     x = x.unfold(1, kh, dh).unfold(2, kw, dw).unfold(3,kc, dc)
#     x = x.contiguous().view(-1, kh, kw, kc)
#     y = Tensor(images['hr']).to(device)

    kernels, offsets_h, offsets_v = kernel_generation_net(x)
    downscaled_img = downsampler_net(x, kernels, offsets_h, offsets_v, SCALE)
    downscaled_img = torch.clamp(downscaled_img, 0, 1)
    downscaled_img = torch.round(downscaled_img * 255)
    out = upsampler_net(downscaled_img).cpu()
#     out = out.view(unfold_shape)
#     output_c = unfold_shape[1] * unfold_shape[4]
#     output_h = unfold_shape[2] * unfold_shape[5]
#     output_w = unfold_shape[3] * unfold_shape[6]
#     out = out.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
#     out = out.view(batch, output_c, output_h, output_w)
    
    loss, px_mse = mseloss(out, x)
    
    optimizer.zero_grad()
    loss.backward()
#     torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    px_mse = np.round(np.sqrt(px_mse)*255, 2)
    pbar.set_description("Loss: {}".format(px_mse))
    
    if epoch%1000==0:
        torch.save(model.state_dict(), "./models/model-{}.pt".format(epoch))

    del x, y, loss
    
    
    
