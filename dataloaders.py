from torch.utils.data import Dataset
import cv2
import numpy as np

from worldstrat.datasets import SatelliteDataset, DictDataset, make_transforms_JIF
from worldstrat.datasources import S2_ALL_12BANDS
from multiprocessing import Manager
from worldstrat.datasources import *
import pandas as pd

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
        with open(os.path.join(self.directory, "iids_train.txt")) as file:
            self.train_ids = file.read().splitlines()
        with open(os.path.join(self.directory, "iids_test.txt")) as file:
            self.test_ids = file.read().splitlines()

    def __len__(self):
        return len(self.train_ids)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            idx = np.array([idx])
        lr, hr = [], []
        for _idx in idx:
            img = cv2.imread(os.path.join(self.directory, "images/train", "{}.jpg".format(_idx)))
            for t in self.transform:
                img = t(img)
            h, w, _ = img.shape
            if h==321 and w==481:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                h, w, _ = img.shape
            hp, wp = h-h%self.patch_size, w-w%self.patch_size
            img = img[:hp, :wp, :]
            hr.append(img)
            lr.append(cv2.resize(cv2.GaussianBlur(img, ksize=(3,3), sigmaX=1.0), (hp//self.downscale, wp//self.downscale)))
        return {"lr": np.array(lr), "hr": np.array(hr)}
    
    
class WorldStrat(Dataset):
    
    def __init__(self, dataset_root="/scratch/partenza.g/", hr_dataset_folder="hr_dataset_folder/", metadata_file="metadata.csv"):
        self.transforms = make_transforms_JIF(lr_bands_to_use='true_color', radiometry_depth=12)
        self.multiprocessing_manager = Manager()
        self.hr_rgb = SatelliteDataset(
            root=os.path.join(dataset_root, hr_dataset_folder, "*"),
            file_postfix="_rgbn.tiff",
            transform=self.transforms["hr"],
            bands_to_read=SPOT_RGB_BANDS,
            number_of_revisits=1,
            multiprocessing_manager=self.multiprocessing_manager
        )
        self.dataset = DictDataset(
            **{
                "hr": self.hr_rgb,
            }
        )
        self.img_count = len(self.dataset)
        self.metadata = pd.read_csv(os.path.join(dataset_root, metadata_file))
        print(f"Loaded dataset with {self.img_count} images.")
    
    def __len__(self):
        return self.img_count

    def __getitem__(self, idx):
        return list(self.dataset[_idx] for _idx in idx)
    
    
