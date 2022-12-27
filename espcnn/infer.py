from model import SupNet
import cv2
from torch import Tensor
import torch
from loader import WorldStrat
import numpy as np
import torch.nn.functional as F

dataset = WorldStrat()



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
permute = [2, 1, 0]
model = SupNet()
model.load_state_dict(torch.load("./models/model-1000.pt"))
model.eval()
model.to(device)

images = dataset[[1234]]
y = torch.cat([image['hr'] for image in images], dim=0).to(device)

# hr to 1024x1024 and lr to 512x512
x = F.interpolate(y, size=(1024//2, 1024//2), mode='bicubic', align_corners=False)
cv2.imwrite("512m.jpg", x[0].permute(1,2,0).cpu().numpy()[...,::-1]*255)

with torch.no_grad():
    pred = model(x)

res = cv2.imwrite("1024m.jpg", pred[0].detach().cpu().numpy()[...,::-1]*255)

print(res)

with torch.no_grad():
    pred = model(pred.permute(0,3,1,2))

res = cv2.imwrite("2048m.jpg", pred[0].detach().cpu().numpy()[...,::-1]*255)

print(res)


_1 = F.interpolate(y, size=(512, 512), mode='bicubic', align_corners=False)
_2 = F.interpolate(_1, size=(1024, 1024), mode='bicubic', align_corners=False)
_3 = F.interpolate(_1, size=(2048, 2048), mode='bicubic', align_corners=False)
res = cv2.imwrite("512i.jpg", _1[0].permute(1,2,0).detach().cpu().numpy()[...,::-1]*255)
res = cv2.imwrite("1024i.jpg", _2[0].permute(1,2,0).detach().cpu().numpy()[...,::-1]*255)
res = cv2.imwrite("2048i.jpg", _3[0].permute(1,2,0).detach().cpu().numpy()[...,::-1]*255)
