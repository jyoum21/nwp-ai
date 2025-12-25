import torch
import h5py
import numpy as np
from torch.utils.data import Dataset

class HurricaneH5Dataset(Dataset):
    def __init__(self, h5_path, transform=None):
        self.h5_path = h5_path
        self.transform = transform
        self.dataset = None 
        
        # min and max temperatures allowed in data
        self.MIN_TEMP = 150.0
        self.MAX_TEMP = 340.0

        with h5py.File(h5_path, 'r') as f:
            self.length = f.attrs['total_samples']

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.dataset is None:
            self.dataset = h5py.File(self.h5_path, 'r')

        # load raw data
        raw_image = self.dataset['images/data'][idx]
        
        # linear normalization from 150-340K to 0-1
        norm_image = (raw_image - self.MIN_TEMP) / (self.MAX_TEMP - self.MIN_TEMP)
        norm_image = np.clip(norm_image, 0.0, 1.0)

        # add channel dimension
        norm_image = np.expand_dims(norm_image, axis=0)

        # convert to tensor
        image_tensor = torch.from_numpy(norm_image).float()
        
        # load label (actual wind speeds)
        label = torch.tensor(self.dataset['metadata/wind_speeds'][idx]).float()

        # apply augmentations
        if self.transform:
            image_tensor = self.transform(image_tensor)

        return image_tensor, label

    def close(self):
        if self.dataset:
            self.dataset.close()
