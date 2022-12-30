from torch.utils.data import Dataset
import cv2
import numpy as np
import pandas as pd
import os

class KaggleDataset(Dataset):

    def __init__(self, lr_dir, hr_dir, filename, img_count, transform=None):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.filename = filename
        self.transform = transform

    def __len__(self):
        return self.img_count

    def __getitem__(self, idx):
        lr, hr = [], []
        for _idx in idx:
            lr.append(cv2.imread(self.lr_dir+self.filename.format(_idx)))
            hr.append(cv2.imread(self.hr_dir+self.filename.format(_idx)))
        return {"lr": np.array(lr), "hr": np.array(hr)}


class BSDS300(Dataset):

    def __init__(self, directory, patch_size, transform=None, downscale=2):
        self.directory = directory
        self.transform = transform
        self.patch_size = patch_size
        self.downscale = downscale
        self.folder = "images/train"
        with open(os.path.join(self.directory, "iids_train.txt")) as file:
            self.train_ids = file.read().splitlines()
        with open(os.path.join(self.directory, "iids_test.txt")) as file:
            self.test_ids = file.read().splitlines()

    def __len__(self):
        return len(self.train_ids)
    
    def test(self):
        self.folder = "images/test"
        
    def train(self):
        self.folder = "images/train"

    def __getitem__(self, idx, test=False):
        if isinstance(idx, int):
            idx = np.array([idx])
        lr, hr = [], []
        for _idx in idx:
            img = cv2.imread(os.path.join(self.directory, self.folder, "{}.jpg".format(_idx)))
            for t in self.transform:
                img = t(img)
            h, w, _ = img.shape
            if h==321 and w==481:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                h, w, _ = img.shape
            hp, wp = h-h%self.patch_size, w-w%self.patch_size
            img = img[:hp, :wp, :]
            hr.append(img)
            img = cv2.GaussianBlur(img, ksize=(3,3), sigmaX=1.0)
            img = cv2.resize(img, (wp//self.downscale, hp//self.downscale))
            lr.append(img)
        return {"lr": np.array(lr), "hr": np.array(hr)}
    
