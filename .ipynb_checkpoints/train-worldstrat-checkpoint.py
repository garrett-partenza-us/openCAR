from loader import WorldStrat
from model import SupNet
from torch import optim
from torch.nn import MSELoss
from torch import Tensor
from tqdm import tqdm
import random
import torch
import numpy as np
from loss import MSGE
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = SupNet()
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

dataset = WorldStrat()

epochs = 10000
batch = 16
mseloss = MSGE(batch=batch)
kh, kw, kc = 16, 16, 3
dh, dw, dc = 16, 16, 3
num_patches = 527**2 // 16**2
unfold_shape = [batch, int(np.sqrt(num_patches)), int(np.sqrt(num_patches)), 1, kw*2, kh*2, kc]
losses = []

for epoch in (pbar := tqdm(range(epochs))):
    
    # randomly sample worldstrat
    images = dataset[np.random.randint(1, len(dataset), batch)]
    y = torch.cat([image['hr'] for image in images], dim=0).to(device)
    
    # hr to 1024x1024 and lr to 512x512
    y = F.interpolate(y, size=(1024, 1024), mode='bicubic', align_corners=False)
    x = F.interpolate(y, size=(1024//2, 1024//2), mode='bicubic', align_corners=False)
    
    # break lr into patches
    x = x.unfold(1, kc, dc).unfold(2, kh, dh).unfold(3,kw, dw)
    x = x.contiguous().view(-1, kc, kh, kw)

    # forward pass
    out = model(x)
    
    # reconstruct image from patches
    out = out.view(unfold_shape)
    output_c = unfold_shape[1] * unfold_shape[4]
    output_h = unfold_shape[2] * unfold_shape[5]
    output_w = unfold_shape[3] * unfold_shape[6]
    out = out.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
    out = out.view(batch, output_c, output_h, output_w)

    # compute loss (channel first)
    loss, px_mse = mseloss(out.permute(0,3,1,2), y)
    
    # backward pass
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    # log denormalized pixel loss
    px_mse = np.round(np.sqrt(px_mse)*255, 2)
    pbar.set_description("Loss: {}".format(px_mse))
    losses.append(px_mse)
    
    # save loss plot every 50 epochs
    if epoch%50==0:
        plt.figure(figsize=(12,6))
        plt.title("MSE Pixel Loss")
        plt.plot(losses, label="training loss", c='g')
        plt.legend()
        plt.savefig("loss.png")
        plt.clf()
        
    # save model every 1000 batches
    if epoch%1000==0:
        torch.save(model.state_dict(), "./models/model-{}.pt".format(epoch))

    del images, x, y, out, px_mse, loss
    
    
    
