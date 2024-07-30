import os
import torch

from glob import glob
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class ResNetDataset(Dataset):
    def __init__(self, data_dir, split):

        self.split = split
        
        if split == "train":
            images = glob(os.path.join(data_dir, "*.pt"))
            labels = [torch.tensor(0) if "ncr" in item else torch.tensor(1) for item in images]
            self.images, _, self.labels, _ = train_test_split(images, labels, train_size=0.8, random_state=42)
        
        elif split == "val":
            images = glob(os.path.join(data_dir, "*.pt"))
            labels = [torch.tensor(0) if "ncr" in item else torch.tensor(1) for item in images]
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
        label = self.labels[idx].long()

        # for testing purposes only
        # image = torch.ones((1, 32, 32, 32))
        # label = torch.tensor(0).long()

        return image, label