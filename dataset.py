import os
import torch
import numpy as np

from glob import glob
from pathlib import Path
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight


class ResNetDataset(Dataset):
    def __init__(self, config, split):

        self.config = config
        self.split = split
        
        match config.dataset:
            case "sarcoma":
                self.data_dir = "/home/johannes/Code/ResponsePrediction/data/sarcoma/preprocessed"
            case "glioblastoma":
                self.data_dir = "/home/johannes/Code/ResponsePrediction/data/glioblastoma/preprocessed"
        
        self.train_dir = os.path.join(self.data_dir, f"train_{self.config.sequence}_{self.config.examination}")
        self.test_dir = os.path.join(self.data_dir, f"test_{self.config.sequence}_{self.config.examination}")

        # for regression only
        self.train_labels = [(int(os.path.basename(file).split("_")[-1].replace(".pt", ""))) for file in glob(os.path.join(self.train_dir, "*.pt"))]
        self.test_labels = [(int(os.path.basename(file).split("_")[-1].replace(".pt", ""))) for file in glob(os.path.join(self.test_dir, "*.pt"))]
        # self.scaler = StandardScaler()
        self.scaler = MinMaxScaler()
        self.train_labels = list(self.scaler.fit_transform(np.array(self.train_labels).reshape(-1, 1)).flatten())
        self.test_labels = list(self.scaler.transform(np.array(self.test_labels).reshape(-1, 1)).flatten())        
        
        if split == "train":
            images = glob(os.path.join(self.train_dir, "*.pt"))
            match self.config.task:
                case "classification":
                    labels = [(int(os.path.basename(file).split("_")[-1].replace(".pt", ""))) for file in glob(os.path.join(self.train_dir, "*.pt"))]
                    labels = [torch.tensor(0) if int(label) < 95 else torch.tensor(1) for label in labels]
                    
                    # labels = [torch.tensor(0) if (int(os.path.basename(file).split("_")[-1].replace(".pt", "")) < 95) in file else torch.tensor(1) for file in images]
                case "regression":
                    labels = self.train_labels
            self.images, _, self.labels, _ = train_test_split(images, labels, train_size=0.8, random_state=42, stratify=labels)

            labels_arr = np.array([tensor.numpy() for tensor in self.labels])
            self.class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels_arr), y=labels_arr)
            self.class_weights = torch.tensor(self.class_weights).to(torch.float32).to(self.config.device)
            self.sample_weights = self.class_weights[torch.tensor(labels_arr)]
        
        elif split == "val":
            images = glob(os.path.join(self.train_dir, "*.pt"))
            match self.config.task:
                case "classification":
                    labels = [(int(os.path.basename(file).split("_")[-1].replace(".pt", ""))) for file in glob(os.path.join(self.train_dir, "*.pt"))]
                    labels = [torch.tensor(0) if int(label) < 95 else torch.tensor(1) for label in labels]
                    # labels = [torch.tensor(0) if (int(os.path.basename(file).split("_")[-1].replace(".pt", "")) < 95) in file else torch.tensor(1) for file in images]
                case "regression":
                    labels = self.train_labels
            _, self.images, _, self.labels = train_test_split(images, labels, train_size=0.8, random_state=42, stratify=labels)

        elif split == "test":
            self.images = glob(os.path.join(self.test_dir, "*.pt"))
            match self.config.task:
                case "classification":
                    self.labels = [(int(os.path.basename(file).split("_")[-1].replace(".pt", ""))) for file in glob(os.path.join(self.test_dir, "*.pt"))]
                    self.labels = [torch.tensor(0) if int(label) < 95 else torch.tensor(1) for label in self.labels]
                    # self.labels = [torch.tensor(0) if (int(os.path.basename(file).split("_")[-1].replace(".pt", "")) < 95) in file else torch.tensor(1) for file in self.images]
                case "regression":
                    self.labels = self.test_labels

        else:
            raise ValueError("Check split.")

    def __len__(self):            
        return len(self.images)
        

    def __getitem__(self, idx):

        image_path = self.images[idx]
        match self.config.channels:
            case 1:
                image = torch.load(image_path, weights_only=True)  
                # image = torch.ones((1, 32, 32, 32))
            case _:
                parent_dir = Path(image_path).parent
                if self.split == "test":
                    patient_id = os.path.basename(image_path)[:4]
                else:
                    patient_id = os.path.basename(image_path)[:6]
                all_patient_files = sorted(glob(os.path.join(parent_dir, f"{patient_id}*.pt")))
                image = [torch.load(path, weights_only=True) for path in all_patient_files]
                image = torch.concatenate(image, dim=0)

        match self.config.task:
            case "classification":
                label = self.labels[idx].to(torch.long)
            case "regression":
                label = torch.tensor(self.labels[idx]).to(torch.float32)

        # for testing purposes only
        # image = torch.ones((1, 32, 32, 32))
        # label = torch.tensor(0).long()

        return image, label