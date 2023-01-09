from models.car import ResamplerNet
from models.downsampler import Downsampler
import torch

torch.pi = torch.acos(torch.zeros(1)).item() * 2

def softround(x, alpha=1.0):
        return x - alpha * (torch.sin( 2 * torch.pi * x ) / (2 * torch.pi))
        
def batch_bli(im, x, y, channel_first=False, dtype=torch.FloatTensor, dtype_long=torch.LongTensor):
    # ensure channel last
    if channel_first:
        im = im.permute(0,2,3,1)
    batch = im.shape[0]
    num_points = x.shape[1]
    assert x.shape==y.shape
    # Get four corner indicies
    x0 = torch.floor(x).type(dtype_long)
    x1 = x0 + 1
    y0 = torch.floor(y).type(dtype_long)
    y1 = y0 + 1
    # Clamp within h, w boundries
    x0 = torch.clamp(x0, 0, im.shape[2]-1)
    x1 = torch.clamp(x1, 0, im.shape[2]-1)
    y0 = torch.clamp(y0, 0, im.shape[1]-1)
    y1 = torch.clamp(y1, 0, im.shape[1]-1)
    # Get four corner pixel values
    Ia = torch.cat([im[b, x, y, :] for b in range(batch) for x, y in zip(x0[b], y0[b])])
    Ib = torch.cat([im[b, x, y, :] for b in range(batch) for x, y in zip(x0[b], y0[b])])
    Ic = torch.cat([im[b, x, y, :] for b in range(batch) for x, y in zip(x0[b], y0[b])])
    Id = torch.cat([im[b, x, y, :] for b in range(batch) for x, y in zip(x0[b], y0[b])])
    # Define matricies
    scale = (1 / ( (x1-x0) * (y1-y0) ) ).flatten()
    m1 = torch.cat([ torch.sub(x1, x), torch.sub(x, x0)], dim=1)
    m2 = torch.stack([Ib, Ia, Id, Ic], dim=1).reshape(batch*num_points,2,2,3)
    m3 = torch.cat([ torch.sub(y1, y), torch.sub(y, y0) ], dim=1)
    # Reshape for batch matmul
    m1 = m1.reshape(batch*num_points,1,1,2).repeat(1,2,1,1)
    m3 = m3.reshape(batch*num_points,1,2,1)
    return scale[:,None] * torch.matmul( torch.matmul(m1, m2).permute(0,3,2,1), m3 ).flatten(start_dim=1)


scale = 2
kernel_size = 3
kernel_generation_net = ResamplerNet(rgb_range=1.0)
downsampler_net = Downsampler()
x = torch.rand((1,3,48,48))
x.requires_grad=True

kernels, offsets_h, offsets_v = kernel_generation_net(x)

batch = x.shape[0]
h, w = x.shape[2]//scale, x.shape[3]//scale
kernels = kernels.permute(0,2,3,1)
offsets_h = offsets_h.permute(0,2,3,1)
offsets_v = offsets_v.permute(0,2,3,1)
u, v = torch.arange(h)+0.5*scale-0.5, torch.arange(w)+0.5*scale-0.5
coords_x = torch.add(offsets_h, kernel_size/2)
coords_x = torch.add(coords_x, torch.arange(3).reshape(3,1).repeat(1,3).flatten())
coords_x = torch.add(coords_x, u.reshape(h,1).repeat(1,kernel_size**2))
coords_y = torch.add(offsets_v, kernel_size/2)
coords_y = torch.add(coords_y, torch.arange(3).repeat(3))
coords_y = torch.add(coords_y, u.reshape(w,1).repeat(1,kernel_size**2))
pix_hr = batch_bli(x.permute(0,2,3,1), coords_x.flatten(start_dim=1), coords_y.flatten(start_dim=1))
pix_hr = pix_hr.reshape(batch,h,w,kernel_size**2,3)
pix_lr = torch.mul(kernels.unsqueeze(-1).repeat(1,1,1,1,3), pix_hr)
out = torch.sum(pix_lr, axis=-2)
out = softround(out*255.0)

loss = out.sum()

loss.backward()
