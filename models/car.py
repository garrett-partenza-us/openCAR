import torch
import torch.nn as nn
import math

LEAKY_FACTOR = 0.2
RES_FACTOR = 1.0


"""
Subtract RGB means for standardization.
Why is the standard deviation 1.0?
https://github.com/sunwj/CAR/blob/05b22776b9f690dac94ced8baeb455bb722c0997/modules.py#L105"""
class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean=(0.5, 0.5, 0.5), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(in_channels=3, out_channels=3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False
            
            
class PixelUnshuffle(nn.Module):
    """
    Inverse process of pytorch pixel shuffle module
    """
    def __init__(self, down_scale):
        """
        :param down_scale: int, down scale factor
        """
        super(PixelUnshuffle, self).__init__()

        if not isinstance(down_scale, int):
            raise ValueError('Down scale factor must be a integer number')
        self.down_scale = down_scale

    def forward(self, input):
        """
        :param input: tensor of shape (batch size, channels, height, width)
        :return: tensor of shape(batch size, channels * down_scale * down_scale, height / down_scale, width / down_scale)
        """
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
        self.pad = nn.ReflectionPad2d(kernel_size//2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.act = nn.LeakyReLU(LEAKY_FACTOR)
        
    def forward(self, x):
        x = self.pad(x)
        x = self.conv1(x)
        x = self.act(x)
        x = self.pad(x)
        x = self.conv2(x)
        return x
        

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
        self.res = [ResidualBlock(128, 128, 3) for _ in range(res_blocks)]
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
        for block in self.res:
            x = x + block(x) * RES_FACTOR
        x = self.ds_8(x)
        
        kernels = self.kernel_trunk(x)
        kernels = torch.clamp(self.kernel_prediction(kernels), min=1e-6, max=1.0)
        kernels = kernels / torch.sum(kernels, dim=1, keepdim=True).clamp(min=1e-6)
        
        offsets = self.offset_trunk(x)
        offsets_h, offsets_v = self.offset_h_prediction(offsets), self.offset_v_prediction(offsets)
        
        return kernels, offsets_h, offsets_v