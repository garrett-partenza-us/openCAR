import torch
import torch.nn as nn
import math

LEAKY_FACTOR = 0.2
RES_FACTOR = 1.0
torch.pi = torch.acos(torch.zeros(1)).item() * 2
        
class Downsampler(nn.Module):
    def __init__(self, ksize, scale, batch):
        super(Downsampler, self).__init__()
        self.ksize = ksize
        self.scale = scale
        self.batch = batch
        
    def softround(self, x, alpha=1.0):
        return x - alpha * (torch.sin( 2 * torch.pi * x ) / (2 * torch.pi))
    
    def batch_bli(self, im, x, y, channel_first=False, dtype=torch.FloatTensor, dtype_long=torch.LongTensor):
        num_points = x.shape[1]
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
        Ia = torch.cat([im[b, x, y, :] for b in range(self.batch) for x, y in zip(x0[b], y0[b])])
        Ib = torch.cat([im[b, x, y, :] for b in range(self.batch) for x, y in zip(x0[b], y1[b])])
        Ic = torch.cat([im[b, x, y, :] for b in range(self.batch) for x, y in zip(x1[b], y0[b])])
        Id = torch.cat([im[b, x, y, :] for b in range(self.batch) for x, y in zip(x1[b], y1[b])])
        # Define matricies
        scale = (1 / ( (x1-x0) * (y1-y0) ) ).flatten()
        m1 = torch.cat([ torch.sub(x1, x), torch.sub(x, x0)], dim=1).float()
        m2 = torch.stack([Ib, Ia, Id, Ic], dim=1).reshape(self.batch*num_points,2,2,3).float()
        m3 = torch.cat([ torch.sub(y1, y), torch.sub(y, y0) ], dim=1).float()
        # Reshape for batch matmul
        m1 = m1.reshape(self.batch*num_points,1,1,2).repeat(1,2,1,1)
        m3 = m3.reshape(self.batch*num_points,1,2,1)
        return scale[:,None] * torch.matmul( torch.matmul(m1, m2).permute(0,3,2,1), m3 ).flatten(start_dim=1)
    
    def forward(self, images, kernels, offsets_x, offsets_y, channel_first=False):
        # ensure channel last
        if channel_first:
            images = images.permute(0,2,3,1)
        self.batch = images.shape[0]
        h, w = images.shape[2]//self.scale, images.shape[3]//self.scale
        kernels = kernels.permute(0,2,3,1)
        offsets_x = offsets_x.permute(0,2,3,1)
        offsets_y = offsets_y.permute(0,2,3,1)
        u, v = torch.arange(h)+0.5*self.scale-0.5, torch.arange(w)+0.5*self.scale-0.5
        coords_x = torch.add(offsets_x, self.ksize/2)
        coords_x = torch.add(coords_x, torch.arange(3).reshape(3,1).repeat(1,3).flatten())
        coords_x = torch.add(coords_x, u.reshape(h,1).repeat(1,self.ksize**2))
        coords_y = torch.add(offsets_y, self.ksize/2)
        coords_y = torch.add(coords_y, torch.arange(3).repeat(3))
        coords_y = torch.add(coords_y, u.reshape(w,1).repeat(1,self.ksize**2))
        pix_hr = self.batch_bli(images.permute(0,2,3,1), coords_x.flatten(start_dim=1), coords_y.flatten(start_dim=1), self.batch)
        pix_hr = pix_hr.reshape(self.batch, h, w, self.ksize**2,3)
        pix_lr = torch.mul(kernels.unsqueeze(-1).repeat(1,1,1,1,3), pix_hr)
        out = torch.sum(pix_lr, axis=-2)
        return self.softround(out*255.0)
        


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean=(0.5, 0.5, 0.5), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(in_channels=3, out_channels=3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False
            
            
class PixelUnshuffle(nn.Module):
    def __init__(self, down_scale):
        super(PixelUnshuffle, self).__init__()
        if not isinstance(down_scale, int):
            raise ValueError('Down scale factor must be a integer number')
        self.down_scale = down_scale

    def forward(self, input):
        b, c, h, w = input.size()
        assert h % self.down_scale == 0
        assert w % self.down_scale == 0
        oc = c * self.down_scale ** 2
        oh = int(h / self.down_scale)
        ow = int(w / self.down_scale)
        output_reshaped = input.reshape(b, c, oh, self.down_scale, ow, self.down_scale)
        output = output_reshaped.permute(0, 1, 3, 5, 2, 4).reshape(b, oc, oh, ow)
        return output


class DownsampleBlock(nn.Module):
    def __init__(self, scale, in_channels, out_channels):
        super(DownsampleBlock, self).__init__()
        self.unshuffle = PixelUnshuffle(scale)
        self.conv = nn.Conv2d(in_channels*scale**2, out_channels, kernel_size=1, stride=1)
        
    def forward(self, x):
        x = self.unshuffle(x)
        x = self.conv(x)
        return x
    
    
class UpsampleBlock(nn.Module):
    def __init__(self, scale, in_channels, out_channels):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels*scale**2, kernel_size=1, stride=1)
        self.shuffle = nn.PixelShuffle(scale)

    def forward(self, x):
        x = self.conv(x)
        x = self.shuffle(x)
        return x

    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ResidualBlock, self).__init__()
        self.transform = nn.Sequential(
            nn.ReflectionPad2d(kernel_size//2),
            nn.Conv2d(in_channels, out_channels, kernel_size),
            nn.LeakyReLU(LEAKY_FACTOR),
            nn.ReflectionPad2d(kernel_size//2),
            nn.Conv2d(in_channels, out_channels, kernel_size)
        )
        
    def forward(self, x):
        return x + self.transform(x) * RES_FACTOR
        

class TrunkBlock(nn.Module):
    def __init__(self, upscale, in_channels, out_channels):
        super(TrunkBlock, self).__init__()
        self.transform = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1),
            nn.ReLU(),
            UpsampleBlock(scale=(8//upscale), in_channels=256, out_channels=256),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
    def forward(self, x):
        x = self.transform(x)
        return x


class ResamplerNet(nn.Module):
    def __init__(self, rgb_range, res_blocks=5, kernel_size=(3,3)):
        super(ResamplerNet, self).__init__()
        
        self.meanshift = MeanShift(rgb_range)
        
        self.ds_1 = nn.Sequential(
            nn.ReflectionPad2d(2),
            nn.Conv2d(3, 64, kernel_size=(5,5), stride=1),
            nn.LeakyReLU(LEAKY_FACTOR)
        )
        
        self.ds_2 = DownsampleBlock(2, 64, 128)
        self.ds_4 = DownsampleBlock(2, 128, 128)
        res_4 = list()
        for idx in range(res_blocks):
            res_4 += [ResidualBlock(128, 128, 3)]
        self.res_4 = nn.Sequential(*res_4)
        self.ds_8 = DownsampleBlock(2, 128, 256)
        
        self.kernel_trunk = TrunkBlock(2, 256, 256)
        self.offset_trunk = TrunkBlock(2, 256, 256)
        
        self.kernel_prediction = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, math.prod(kernel_size), 3)
        )
        
        self.offset_h_prediction = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, math.prod(kernel_size), 3),
            nn.Tanh()
        )
        
        self.offset_v_prediction = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, math.prod(kernel_size), 3),
            nn.Tanh()
        )
        
    def forward(self, x):
        x = self.meanshift(x)
        x = self.ds_1(x)
        x = self.ds_2(x)
        x = self.ds_4(x)
        x = self.res_4(x)
        x = self.ds_8(x)
        
        kernels = self.kernel_trunk(x)
        kernels = torch.clamp(self.kernel_prediction(kernels), min=1e-6, max=1.0)
        kernels = kernels / torch.sum(kernels, dim=1, keepdim=True).clamp(min=1e-6)
        
        offsets = self.offset_trunk(x)
        offsets_h, offsets_v = self.offset_h_prediction(offsets), self.offset_v_prediction(offsets)
    
        return kernels, offsets_h, offsets_v
    
    