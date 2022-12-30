from dataloaders import BSDS300
from model import SupNet
from torch import optim
from torch.nn import MSELoss
from torch import Tensor
from tqdm import tqdm
import random
import torch
import numpy as np
from loss import MSGE
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = SupNet()
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

dataset = BSDS300(directory="/home/partenza.g/sr/benchmarks/BSDS300", transform=[lambda x: x/255.0], downscale=2, patch_size=16)

epochs = 1
batch = 32
mseloss = MSGE(batch=batch)
kh, kw, kc = 16, 16, 3
dh, dw, dc = 16, 16, 3
unfold_shape = [batch, 480//16, 320//16, 1, kw*2, kh*2, kc]

for epoch in (pbar := tqdm(range(epochs))):
    
    images = dataset[random.choices(dataset.train_ids, k=batch)]
    
    x = Tensor(images['lr']).to(device)
    x = x.unfold(1, kh, dh).unfold(2, kw, dw).unfold(3,kc, dc)
    x = x.contiguous().view(-1, kh, kw, kc)
    
    y = Tensor(images['hr']).to(device)
        
    out = model(x)
    out = out.view(unfold_shape)

    output_c = unfold_shape[1] * unfold_shape[4]
    output_h = unfold_shape[2] * unfold_shape[5]
    output_w = unfold_shape[3] * unfold_shape[6]
    out = out.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
    out = out.view(batch, output_c, output_h, output_w)
    
    loss, px_mse = mseloss(out, y)
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    px_mse = np.round(np.sqrt(px_mse)*255, 2)
    pbar.set_description("Loss: {}".format(px_mse))
    
    if epoch%1000==0:
        torch.save(model.state_dict(), "./models/model-{}.pt".format(epoch))

    del x, y, loss
    
    
    
