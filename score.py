import torch
import argparse 
import os
from models.espcnn import ESPCNN
from utils.dataloaders import BSDS300
from utils.loss import PSNR
import numpy as np 
import os
from torch import optim
from torch import Tensor
from tqdm import tqdm
import cv2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_dir = "weights/"

parser = argparse.ArgumentParser(
    prog = 'Infer.py',
    description = 'Calculate the PSNR of a model on the BSDS300 test set',
)
parser.add_argument('-m', '--model')
parser.add_argument('-s', '--save')
args = parser.parse_args()

psnr = PSNR()
model = ESPCNN()
model.load_state_dict(torch.load(os.path.join(model_dir, args.model+".pt")))
model.eval()
model.to(device)

dataset = BSDS300(directory="/home/partenza.g/sr/benchmarks/BSDS300", transform=[lambda x: x/255.0], downscale=2, patch_size=16)
dataset.test()

errors = []
save = int(args.save)
for idx in tqdm(dataset.test_ids):
    img = dataset[int(idx)]
    x = Tensor(img['lr']).to(device).permute(0,3,1,2)
    y = Tensor(img['hr']).to(device).permute(0,3,1,2) 
    out = model(x)
    error = psnr(out*255.0, y*255.0)
    errors.append(error.cpu().item())
    if save:
        img = (out[0].permute(1,2,0).detach().cpu().numpy()*255.0).astype(np.uint8)
        cv2.imwrite("images/{}_{}.jpg".format(args.model, idx), img)
        save-=1
    
print("PSNR: {}".format(np.mean(errors)))