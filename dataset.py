import os
from torch.utils.data import Dataset
import torch
import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split


class ResNetDataset(Dataset):
    def __init__(self, data_dir, split):

        self.split = split
        
        if split == "train":
            images = glob(os.path.join(data_dir, "*.pt"))
            labels = [torch.tensor(0) if "ncr" in item else torch.tensor(1) for item in images]
            # labels = torch.randint(low=0, high=2, size=(len(images),))
            self.images, _, self.labels, _ = train_test_split(images, labels, train_size=0.8, random_state=42)
        
        elif split == "val":
            images = glob(os.path.join(data_dir, "*.pt"))
            labels = [torch.tensor(0) if "ncr" in item else torch.tensor(1) for item in images]
            # labels = torch.randint(low=0, high=2, size=(len(images),))
            _, self.images, _, self.labels = train_test_split(images, labels, train_size=0.8, random_state=42)

        elif split == "test":
            self.images = glob(os.path.join(data_dir, "*.pt"))
            self.labels = [torch.tensor(0) if "ncr" in item else torch.tensor(1) for item in self.images]

        else:
            raise ValueError("Check split.")

    def __len__(self):            
        return len(self.images)
        

    def __getitem__(self, idx):

        image = torch.load(self.images[idx], weights_only=True)
        # image = torch.ones((1, 32, 32, 32))
        label = self.labels[idx].long()

        return image, label