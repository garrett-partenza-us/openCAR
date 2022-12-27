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
    
    