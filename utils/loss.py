import torch.nn as nn
from torch.nn import MSELoss
import torch
import torch.nn.functional as F
import numpy as np

class MSGE(nn.Module):
    """Weighted mean gradient squared eror
    expects images to be channel first"""
    
    def __init__(self, batch, grad_weight=0.1, eps=1e-5):
        super(MSGE, self).__init__()
        self.name = "MSGE"
        self.mse = MSELoss()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch = batch
        self.eps = torch.tensor(eps).to(self.device)
        self.grad_weight = torch.tensor(grad_weight).to(self.device)
        self.h = torch.Tensor([[1, 0, -1],
            [2, 0, -2],
            [1, 0, -1]]
        ).view((1,1,3,3)).repeat(batch,3,1,1).to(self.device)
        self.v = torch.Tensor([[1, 2, 1],
            [0, 0, 0],
            [-1, -2, -1]]
        ).view((1,1,3,3)).repeat(batch,3,1,1).to(self.device)

    # This implementation increases vanilla MSE training time by ~120%
    # Faster convergence in early epochs
    def forward(self, inputs, targets):
        Gx_input = F.conv2d(inputs, self.h)
        Gy_input = F.conv2d(inputs, self.v)
        Gx_target = F.conv2d(inputs, self.h)
        Gy_target = F.conv2d(inputs, self.v)
        Gx_mse = self.mse(Gx_input, Gx_target)
        Gy_mse = self.mse(Gy_input, Gy_target)
        Px_mse = self.mse(inputs, targets)
        loss = Px_mse + (self.grad_weight * Gx_mse) + (self.grad_weight * Gy_mse) + self.eps
        return loss, Px_mse.detach().cpu().item()
    

class PSNR:
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255]"""

    def __init__(self):
        self.name = "PSNR"

    @staticmethod
    def __call__(img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        return 20 * torch.log10(255.0 / torch.sqrt(mse))
        
        
        
        
        
        