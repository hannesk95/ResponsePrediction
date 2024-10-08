import os
import monai.transforms
import torch
import scipy
import numpy as np
import torchio as tio
import pandas as pd
import nibabel as ni
import SimpleITK as sitk
import monai

from glob import glob
from pathlib import Path
from torch.utils.data import Dataset
from monai.data import CacheDataset
from scipy.ndimage import map_coordinates, gaussian_filter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import OneHotEncoder

from batchgenerators.transforms.spatial_transforms import SpatialTransform_2, MirrorTransform
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, GammaTransform, ContrastAugmentationTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
# from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.dataloading.data_loader import SlimDataLoaderBase
from monai.transforms import Compose

class ResNetDataset(Dataset):
    def __init__(self, config, split):

        self.config = config
        self.split = split
        self.tio_transform = tio.transforms.RandomAffine(scales=0,
                                                         degrees=0,
                                                         translation=5)
        
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

        if self.split == "train":
            if self.config.augmentation:
                # image = self.augment_image(image, apply_elastic=False)
                image = self.tio_transform(image)

        # for testing purposes only
        # image = torch.ones((1, 32, 32, 32))
        # label = torch.tensor(0).long()

        return image, label
    

    def random_flip(self, img):
        if torch.rand(1).item() > 0.5:
            img = torch.flip(img, dims=[0])  # Flip along the x-axis
        if torch.rand(1).item() > 0.5:
            img = torch.flip(img, dims=[1])  # Flip along the y-axis
        if torch.rand(1).item() > 0.5:
            img = torch.flip(img, dims=[2])  # Flip along the z-axis
        return img
    

    def random_rotation(self, img):
        angles = torch.rand(3) * 10  # Random angles between 0 and 10 degrees
        img = scipy.ndimage.rotate(img, angles[0], axes=(1, 2), reshape=False)
        img = scipy.ndimage.rotate(img, angles[1], axes=(0, 2), reshape=False)
        img = scipy.ndimage.rotate(img, angles[2], axes=(0, 1), reshape=False)
        return torch.tensor(img, dtype=torch.float32)
    

    def random_scaling(self, img, scale_range=(0.95, 1.05)):
        scale = torch.rand(1).item() * (scale_range[1] - scale_range[0]) + scale_range[0]
        img = scipy.ndimage.zoom(img, scale, order=1)
        return torch.tensor(img, dtype=torch.float32)
    
    def elastic_deformation(self, img, alpha, sigma):
        random_state = np.random.RandomState(None)
        
        shape = img.shape
        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dz = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        
        x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
        indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1)), np.reshape(z + dz, (-1, 1))
        
        distored_image = map_coordinates(img, indices, order=1, mode='reflect')
        distored_image = distored_image.reshape(img.shape)
        
        return torch.tensor(distored_image, dtype=torch.float32)
    
    def augment_image(self, img, apply_elastic=False):
        # img = self.random_flip(img)
        img = self.random_rotation(img)
        img = self.random_scaling(img)
        if apply_elastic:
            img = self.elastic_deformation(img, alpha=30, sigma=3)
        return img
    

class CombinedDataset(Dataset):
    def __init__(self, config, split):

        self.config = config
        self.split = split
        # self.tio_transform = tio.transforms.RandomAffine(scales=0,
        #                                                  degrees=0,
        #                                                  translation=5)

        self.affine = tio.transforms.RandomAffine(scales=[0.95, 1.05],
                                                  degrees=5,
                                                  translation=5)        
        self.blur = tio.transforms.RandomBlur()
        self.noise = tio.transforms.RandomNoise()
        self.gamma = tio.transforms.RandomGamma()
        self.tio_transform = tio.transforms.Compose([self.affine, self.blur, self.noise, self.gamma])
        
        match config.dataset:
            case "sarcoma":
                self.data_dir = "./data/sarcoma/jan/Combined"
                assert os.path.exists(self.data_dir)
            case "glioblastoma":
                pass

        if (self.config.sequence == "T1") & (self.config.examination == "pre"):
            self.data = glob(os.path.join(self.data_dir, "*T1*.pt"))
            self.data = [file for file in self.data if not "post" in file]            

        elif (self.config.sequence == "T2") & (self.config.examination == "pre"):
            self.data = glob(os.path.join(self.data_dir, "*T2*.pt"))
            self.data = [file for file in self.data if not "post" in file]            

        elif (self.config.sequence == "T1T2") & (self.config.examination == "pre"):
            self.data = glob(os.path.join(self.data_dir, "*.pt"))
            self.data = [file for file in self.data if not "post" in file]

        elif (self.config.sequence == "T1") & (self.config.examination == "post"):
            self.data = glob(os.path.join(self.data_dir, "*T1*.pt"))
            self.data = [file for file in self.data if "post" in file]
        
        elif (self.config.sequence == "T2") & (self.config.examination == "post"):
            self.data = glob(os.path.join(self.data_dir, "*T2*.pt"))
            self.data = [file for file in self.data if "post" in file]

        elif (self.config.sequence == "T1T2") & (self.config.examination == "post"):
            self.data = glob(os.path.join(self.data_dir, "*.pt"))
            self.data = [file for file in self.data if "post" in file]

        elif (self.config.sequence == "T1") & (self.config.examination == "prepost"):
            self.data = glob(os.path.join(self.data_dir, "*.pt"))
            self.data = [file for file in self.data if not "T2" in file]

        elif (self.config.sequence == "T2") & (self.config.examination == "prepost"):
            self.data = glob(os.path.join(self.data_dir, "*.pt"))
            self.data = [file for file in self.data if not "T1" in file]

        elif (self.config.sequence == "T1T2") & (self.config.examination == "prepost"):
            self.data = glob(os.path.join(self.data_dir, "*.pt"))

        
        self.patient_ids = [os.path.basename(filepath)[:6] if os.path.basename(filepath).startswith("Sar") else os.path.basename(filepath)[:4] 
                            for filepath in sorted(glob(os.path.join(self.data_dir, "*.pt")))]
        self.patient_ids = sorted(list(set(self.patient_ids)))

        if config.task == "classification":

            # binary 
            # self.labels = [int(os.path.basename(filepath).split("_")[-1].replace(".pt", ""))
            #             for filepath in sorted(glob(os.path.join(self.data_dir, "*.pt")))][::4]
            
            # self.ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(np.array(self.labels).reshape(-1, 1))

            # multiclass
            self.pcr_values = torch.load("/home/johannes/Code/ResponsePrediction/data/sarcoma/jan/pcr_values.pt", weights_only=True)
            self.labels = [self.pcr_values[patient_id] for patient_id in self.patient_ids]
            self.labels = [0 if label < 50 else 1 if (label>=50)&(label<95) else 2 for label in self.labels ]
            self.ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(np.array(self.labels).reshape(-1, 1))
        
        elif config.task == "regression":
            self.pcr_values = torch.load("/home/johannes/Code/ResponsePrediction/data/sarcoma/jan/pcr_values.pt", weights_only=True)
            self.labels = [self.pcr_values[patient_id]/100 for patient_id in self.patient_ids] 
                
        if self.split == "train":

            if config.task == "classification":
                self.patient_ids, _, self.labels, _ = train_test_split(self.patient_ids, self.labels, train_size=0.5, random_state=42, stratify=self.labels)

                labels_arr = np.array(self.labels)
                self.class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels_arr), y=labels_arr)
                self.class_weights = torch.tensor(self.class_weights).to(torch.float32).to(self.config.device)
                self.sample_weights = self.class_weights[torch.tensor(labels_arr)]
            elif config.task == "regression":
                self.patient_ids, _, self.labels, _ = train_test_split(self.patient_ids, self.labels, train_size=0.5, random_state=42)

        
        elif self.split == "val":
            if config.task == "classification":
                _, self.patient_ids, _, self.labels = train_test_split(self.patient_ids, self.labels, test_size=0.5, random_state=42, stratify=self.labels)
                self.patient_ids, _, self.labels, _ = train_test_split(self.patient_ids, self.labels, test_size=0.5, random_state=42, stratify=self.labels)
            elif config.task == "regression":
                _, self.patient_ids, _, self.labels = train_test_split(self.patient_ids, self.labels, test_size=0.5, random_state=42)
                self.patient_ids, _, self.labels, _ = train_test_split(self.patient_ids, self.labels, test_size=0.5, random_state=42)


        else:
            if config.task == "classification":
                _, self.patient_ids, _, self.labels = train_test_split(self.patient_ids, self.labels, test_size=0.5, random_state=42, stratify=self.labels)
                _, self.patient_ids, _, self.labels = train_test_split(self.patient_ids, self.labels, test_size=0.5, random_state=42, stratify=self.labels)
            elif config.task == "regression":
                _, self.patient_ids, _, self.labels = train_test_split(self.patient_ids, self.labels, test_size=0.5, random_state=42)
                _, self.patient_ids, _, self.labels = train_test_split(self.patient_ids, self.labels, test_size=0.5, random_state=42)



    def __len__(self):            
        return len(self.patient_ids)
        

    def __getitem__(self, idx):

        patient_id = self.patient_ids[idx]        
        
        all_patient_files = sorted([file for file in self.data if os.path.basename(file).startswith(patient_id)])             

        images = [torch.load(path, weights_only=True) for path in all_patient_files]
        image = torch.concatenate(images, dim=0)

        if self.config.task == "classification":
            label = torch.tensor(int(os.path.basename(all_patient_files[0]).split("_")[-1].replace(".pt", ""))).to(torch.long)   

            temp = self.pcr_values[patient_id]

            if temp < 50:
                label = torch.tensor(0).to(torch.long)
            if (temp > 50) & (temp < 95):
                label = torch.tensor(1).to(torch.long)
            if temp >= 95:
                label = torch.tensor(2).to(torch.long)
            

        elif self.config.task == "regression":
            label = torch.tensor(self.pcr_values[patient_id]/100)
            # label = torch.tensor(self.pcr_values[patient_id])


        if self.split == "train":
            if self.config.augmentation:
                image = self.tio_transform(image)     

        return image, label, patient_id
    

class CENDataset(Dataset):
    def __init__(self, config, split):
        self.config = config
        self.split = split
        self.data_dir = "/home/johannes/Code/ResponsePrediction/data/sarcoma/jan/Combined"
        
        
        self.affine = tio.transforms.RandomAffine(scales=[0.95, 1.05],
                                                  degrees=15,
                                                  translation=15)        
        self.blur = tio.transforms.RandomBlur()
        self.noise = tio.transforms.RandomNoise()
        self.gamma = tio.transforms.RandomGamma()
        self.tio_transform = tio.transforms.Compose([self.affine, self.blur, self.noise, self.gamma])
        
        if (self.config.sequence == "T1T2") & (self.config.examination == "pre"):
            self.data = glob(os.path.join(self.data_dir, "*.pt"))
            self.data = [file for file in self.data if not "post" in file]            

        elif (self.config.sequence == "T1T2") & (self.config.examination == "post"):
            self.data = glob(os.path.join(self.data_dir, "*.pt"))
            self.data = [file for file in self.data if not "pre" in file]            

        elif (self.config.sequence == "T1") & (self.config.examination == "prepost"):
            self.data = glob(os.path.join(self.data_dir, "*T1*.pt"))
            # self.data = [file for file in self.data if not "post" in file]

        elif (self.config.sequence == "T2") & (self.config.examination == "prepost"):
            self.data = glob(os.path.join(self.data_dir, "*T2*.pt"))
            # self.data = [file for file in self.data if "post" in file]
        
        elif (self.config.sequence == "T1T2") & (self.config.examination == "prepost"):
            self.data = glob(os.path.join(self.data_dir, "*.pt"))

        self.patient_ids = [os.path.basename(filepath)[:6] if os.path.basename(filepath).startswith("Sar") else os.path.basename(filepath)[:4] 
                            for filepath in sorted(glob(os.path.join(self.data_dir, "*.pt")))]
        self.patient_ids = sorted(list(set(self.patient_ids)))
        self.labels = [int(os.path.basename(filepath).split("_")[-1].replace(".pt", ""))
                       for filepath in sorted(glob(os.path.join(self.data_dir, "*.pt")))][::4]        
        
        self.ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(np.array(self.labels).reshape(-1, 1))
        
        
        if self.split == "train":
            self.patient_ids, _, self.labels, _ = train_test_split(self.patient_ids, self.labels, train_size=0.5, random_state=42, stratify=self.labels)

            labels_arr = np.array(self.labels)
            self.class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels_arr), y=labels_arr)
            self.class_weights = torch.tensor(self.class_weights).to(torch.float32).to(self.config.device)
            self.sample_weights = self.class_weights[torch.tensor(labels_arr)]
        
        elif self.split == "val":
            _, self.patient_ids, _, self.labels = train_test_split(self.patient_ids, self.labels, test_size=0.5, random_state=42, stratify=self.labels)
            self.patient_ids, _, self.labels, _ = train_test_split(self.patient_ids, self.labels, test_size=0.5, random_state=42, stratify=self.labels)

        else:
            _, self.patient_ids, _, self.labels = train_test_split(self.patient_ids, self.labels, test_size=0.5, random_state=42, stratify=self.labels)
            _, self.patient_ids, _, self.labels = train_test_split(self.patient_ids, self.labels, test_size=0.5, random_state=42, stratify=self.labels) 

    def __len__(self):            
        return len(self.patient_ids)
        

    def __getitem__(self, idx):

        # example1 = torch.ones(size=(1, 64, 64, 64))
        # example2 = torch.zeros(size=(1, 64, 64, 64))
        # return [example1, example2, 0]

        patient_id = self.patient_ids[idx]  

        if (self.config.sequence == "T1T2") & (self.config.examination == "pre"):
            self.data = glob(os.path.join(self.data_dir, "*.pt"))
            self.data = [file for file in self.data if not "post" in file]

            t1_pre = [file for file in self.data if not "T2" in file]
            t2_pre = [file for file in self.data if not "T1" in file]   

            t1_pre_all_patient_files = sorted([file for file in t1_pre if os.path.basename(file).startswith(patient_id)])     
            t2_pre_all_patient_files = sorted([file for file in t2_pre if os.path.basename(file).startswith(patient_id)])    

            t1_pre_images = [torch.load(path, weights_only=True) for path in t1_pre_all_patient_files][0] 
            t2_pre_images = [torch.load(path, weights_only=True) for path in t2_pre_all_patient_files][0] 

            label = int(os.path.basename(t1_pre_all_patient_files[0]).split("_")[-1].replace(".pt", ""))
            label = torch.tensor(label).to(torch.long)

            if self.split == "train":
                if self.config.augmentation:
                    t1_pre_images = self.tio_transform(t1_pre_images)
                    t2_pre_images = self.tio_transform(t2_pre_images)

            return t1_pre_images, t2_pre_images, label

        elif (self.config.sequence == "T1T2") & (self.config.examination == "post"):
            self.data = glob(os.path.join(self.data_dir, "*.pt"))
            self.data = [file for file in self.data if not "pre" in file]  

            t1_post = [file for file in self.data if not "T2" in file]
            t2_post = [file for file in self.data if not "T1" in file] 

            t1_post_all_patient_files = sorted([file for file in t1_post if os.path.basename(file).startswith(patient_id)])     
            t2_post_all_patient_files = sorted([file for file in t2_post if os.path.basename(file).startswith(patient_id)])    

            t1_post_images = [torch.load(path, weights_only=True) for path in t1_post_all_patient_files][0] 
            t2_post_images = [torch.load(path, weights_only=True) for path in t2_post_all_patient_files][0] 

            label = int(os.path.basename(t1_post_all_patient_files[0]).split("_")[-1].replace(".pt", ""))
            label = torch.tensor(label).to(torch.long)

            if self.split == "train":
                if self.config.augmentation:
                    t1_post_images = self.tio_transform(t1_post_images)
                    t2_post_images = self.tio_transform(t2_post_images)

            return t1_post_images, t2_post_images, label

        elif (self.config.sequence == "T1") & (self.config.examination == "prepost"):
            self.data = glob(os.path.join(self.data_dir, "*T1*.pt"))
            # self.data = [file for file in self.data if not "post" in file]

            t1_pre = [file for file in self.data if not "post" in file]
            t1_post = [file for file in self.data if "post" in file]

            t1_pre_all_patient_files = sorted([file for file in t1_pre if os.path.basename(file).startswith(patient_id)])     
            t1_post_all_patient_files = sorted([file for file in t1_post if os.path.basename(file).startswith(patient_id)])    

            t1_pre_images = [torch.load(path, weights_only=True) for path in t1_pre_all_patient_files][0] 
            t1_post_images = [torch.load(path, weights_only=True) for path in t1_post_all_patient_files][0] 

            label = int(os.path.basename(t1_pre_all_patient_files[0]).split("_")[-1].replace(".pt", ""))
            label = torch.tensor(label).to(torch.long)

            if self.split == "train":
                if self.config.augmentation:
                    t1_pre_images = self.tio_transform(t1_pre_images)
                    t1_post_images = self.tio_transform(t1_post_images)

            return t1_pre_images, t1_post_images, label

        elif (self.config.sequence == "T2") & (self.config.examination == "prepost"):
            self.data = glob(os.path.join(self.data_dir, "*T2*.pt"))
            # self.data = [file for file in self.data if "post" in file]

            t2_pre = [file for file in self.data if not "post" in file]
            t2_post = [file for file in self.data if "post" in file]

            t2_pre_all_patient_files = sorted([file for file in t2_pre if os.path.basename(file).startswith(patient_id)])     
            t2_post_all_patient_files = sorted([file for file in t2_post if os.path.basename(file).startswith(patient_id)])    

            t2_pre_images = [torch.load(path, weights_only=True) for path in t2_pre_all_patient_files][0] 
            t2_post_images = [torch.load(path, weights_only=True) for path in t2_post_all_patient_files][0] 

            label = int(os.path.basename(t2_pre_all_patient_files[0]).split("_")[-1].replace(".pt", ""))
            label = torch.tensor(label).to(torch.long)

            if self.split == "train":
                if self.config.augmentation:
                    t2_pre_images = self.tio_transform(t2_pre_images)
                    t2_post_images = self.tio_transform(t2_post_images)

            return t2_pre_images, t2_post_images, label
        
        elif (self.config.sequence == "T1T2") & (self.config.examination == "prepost"):
            self.data = glob(os.path.join(self.data_dir, "*.pt"))

            t1_pre = [file for file in self.data if "T1" in file]
            t1_pre = [file for file in t1_pre if not "post" in file]

            t2_pre = [file for file in self.data if "T2" in file]
            t2_pre = [file for file in t2_pre if not "post" in file]

            pre = t1_pre + t2_pre

            t1_post = [file for file in self.data if "T1" in file]
            t1_post = [file for file in t1_post if "post" in file]
            
            t2_post = [file for file in self.data if "T2" in file]
            t2_post = [file for file in t2_post if "post" in file]

            post = t1_post + t2_post

            pre_all_patient_files = sorted([file for file in pre if os.path.basename(file).startswith(patient_id)])     
            post_all_patient_files = sorted([file for file in post if os.path.basename(file).startswith(patient_id)])    

            pre_images = [torch.load(path, weights_only=True) for path in pre_all_patient_files] 
            post_images = [torch.load(path, weights_only=True) for path in post_all_patient_files] 

            pre_image = torch.concatenate(pre_images, dim=0)
            post_image = torch.concatenate(post_images, dim=0)

            label = int(os.path.basename(pre_all_patient_files[0]).split("_")[-1].replace(".pt", ""))
            label = torch.tensor(label).to(torch.long)

            if self.split == "train":
                if self.config.augmentation:
                    pre_image = self.tio_transform(pre_image)
                    post_image = self.tio_transform(post_image)

            return pre_image, post_image, label
 
class BurdenkoDataset(Dataset):
    def __init__(self, config, split):
        self.config = config
        self.split = split

        # self.affine = tio.transforms.RandomAffine(scales=[0.95, 1.05],
        #                                           degrees=10,
        #                                           translation=10)      
        self.flip = tio.transforms.RandomFlip(axes=(0, 1))  
        # self.blur = tio.transforms.RandomBlur(std=(0.5, 1.5))
        self.blur = tio.transforms.RandomBlur()
        # self.noise = tio.transforms.RandomNoise(std=(0, 0.05))
        self.noise = tio.transforms.RandomNoise()
        # self.gamma = tio.transforms.RandomGamma(log_gamma=(0.5, 2))
        self.gamma = tio.transforms.RandomGamma()
        self.tio_transform = tio.transforms.Compose([self.flip, self.blur, self.noise, self.gamma])

        if self.split == "train":
            data_dict = torch.load("/home/johannes/Code/ResponsePrediction/data/glioblastoma/Burdenko/train_patient_ids_burdenko.pt", weights_only=True)

            files = []
            labels = []
            for key in data_dict.keys():
                files.extend(list(data_dict[key].keys()))
                labels.extend(list(data_dict[key].values()))

            self.files = files
            self.labels = [0 if label=="progression" else 1 for label in labels]

            labels_arr = np.array(self.labels)
            self.class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels_arr), y=labels_arr)
            self.class_weights = torch.tensor(self.class_weights).to(torch.float32).to(self.config.device)
            self.sample_weights = self.class_weights[torch.tensor(labels_arr)]
        
        elif self.split == "val":
            data_dict = torch.load("/home/johannes/Code/ResponsePrediction/data/glioblastoma/Burdenko/val_patient_ids_burdenko.pt", weights_only=True)

            files = []
            labels = []
            for key in data_dict.keys():
                files.extend(list(data_dict[key].keys()))
                labels.extend(list(data_dict[key].values()))

            self.files = files
            self.labels = [0 if label=="progression" else 1 for label in labels]
        
        elif self.split == "test":
            data_dict = torch.load("/home/johannes/Code/ResponsePrediction/data/glioblastoma/Burdenko/test_patient_ids_burdenko.pt", weights_only=True)

            files = []
            labels = []
            for key in data_dict.keys():
                files.extend(list(data_dict[key].keys()))
                labels.extend(list(data_dict[key].values()))
            
            self.files = files
            self.labels = [0 if label=="progression" else 1 for label in labels]        

        self.ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(np.array([0, 1]).reshape(-1, 1))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):   

        file = self.files[idx]
        label = self.labels[idx]

        patient_id = file.split("_")[0]
        pre_idx = file.split("_")[1]
        post_idx = file.split("_")[2]

        if self.config.sequence == "T1":
            pre_image_path = glob(os.path.join("/home/johannes/Code/ResponsePrediction/data/glioblastoma/Burdenko/final_dataset", patient_id, f"{pre_idx}*", "*mrcet1*crop*.pt"))
            post_image_path = glob(os.path.join("/home/johannes/Code/ResponsePrediction/data/glioblastoma/Burdenko/final_dataset", patient_id, f"{post_idx}*", "*mrcet1*crop*.pt"))

            images = pre_image_path + post_image_path

        elif self.config.sequence == "T2":
            pre_image_path = glob(os.path.join("/home/johannes/Code/ResponsePrediction/data/glioblastoma/Burdenko/final_dataset", patient_id, f"{pre_idx}*", "*flair*crop*.pt"))
            post_image_path = glob(os.path.join("/home/johannes/Code/ResponsePrediction/data/glioblastoma/Burdenko/final_dataset", patient_id, f"{post_idx}*", "*flair*crop*.pt"))

            images = pre_image_path + post_image_path

        elif self.config.sequence == "T1T2":

            pre_image_path = glob(os.path.join("/home/johannes/Code/ResponsePrediction/data/glioblastoma/Burdenko/final_dataset", patient_id, f"{pre_idx}*", "*mrcet1*crop*.pt"))
            post_image_path = glob(os.path.join("/home/johannes/Code/ResponsePrediction/data/glioblastoma/Burdenko/final_dataset", patient_id, f"{post_idx}*", "*mrcet1*crop*.pt"))

            images = pre_image_path + post_image_path

            pre_image_path = glob(os.path.join("/home/johannes/Code/ResponsePrediction/data/glioblastoma/Burdenko/final_dataset", patient_id, f"{pre_idx}*", "*flair*crop*.pt"))
            post_image_path = glob(os.path.join("/home/johannes/Code/ResponsePrediction/data/glioblastoma/Burdenko/final_dataset", patient_id, f"{post_idx}*", "*flair*crop*.pt"))

            images = images + pre_image_path + post_image_path

        # image = [torch.tensor(ni.load(file).get_fdata()).unsqueeze(dim=0) for file in images]
        image = [torch.load(file, weights_only=True).unsqueeze(dim=0) for file in images]
        image = torch.concatenate(image, dim=0)

        if self.split == "train":
            if self.config.augmentation:
                image = self.tio_transform(image) 

        return image, label
    

class BurdenkoDatasetDKFZ(SlimDataLoaderBase):
    def __init__(self, config, split):
        self.config = config
        self.split = split

        # self.affine = tio.transforms.RandomAffine(scales=[0.95, 1.05],
        #                                           degrees=10,
        #                                           translation=10)      
        self.flip = tio.transforms.RandomFlip(axes=(0, 1))  
        # self.blur = tio.transforms.RandomBlur(std=(0.5, 1.5))
        self.blur = tio.transforms.RandomBlur()
        # self.noise = tio.transforms.RandomNoise(std=(0, 0.05))
        self.noise = tio.transforms.RandomNoise()
        # self.gamma = tio.transforms.RandomGamma(log_gamma=(0.5, 2))
        self.gamma = tio.transforms.RandomGamma()
        self.tio_transform = tio.transforms.Compose([self.flip, self.blur, self.noise, self.gamma])

        if self.split == "train":
            data_dict = torch.load("/media/johannes/WD Elements/Burdenko-GBM-Progression/manifest-1679410600140/train_patient_ids.pt", weights_only=True)

            files = []
            labels = []
            for key in data_dict.keys():
                files.extend(list(data_dict[key].keys()))
                labels.extend(list(data_dict[key].values()))

            self.files = files
            self.labels = [0 if label=="response" else 1 if label=="stable" else 2 for label in labels]

            labels_arr = np.array(self.labels)
            self.class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels_arr), y=labels_arr)
            self.class_weights = torch.tensor(self.class_weights).to(torch.float32).to(self.config.device)
            self.sample_weights = self.class_weights[torch.tensor(labels_arr)]
        
        elif self.split == "val":
            data_dict = torch.load("/media/johannes/WD Elements/Burdenko-GBM-Progression/manifest-1679410600140/val_patient_ids.pt", weights_only=True)

            files = []
            labels = []
            for key in data_dict.keys():
                files.extend(list(data_dict[key].keys()))
                labels.extend(list(data_dict[key].values()))

            self.files = files
            self.labels = [0 if label=="response" else 1 if label=="stable" else 2 for label in labels]
        
        elif self.split == "test":
            data_dict = torch.load("/media/johannes/WD Elements/Burdenko-GBM-Progression/manifest-1679410600140/test_patient_ids.pt", weights_only=True)

            files = []
            labels = []
            for key in data_dict.keys():
                files.extend(list(data_dict[key].keys()))
                labels.extend(list(data_dict[key].values()))
            
            self.files = files
            self.labels = [0 if label=="response" else 1 if label=="stable" else 2 for label in labels]        

        self.ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(np.array([0, 1, 2]).reshape(-1, 1))

    # def __len__(self):
    #     return len(self.files)

    

    def generate_train_batch(self):   

        idx = np.random.randint(len(self.files))

        file = self.files[idx]
        label = self.labels[idx]

        patient_id = file.split("_")[0]
        pre_idx = file.split("_")[1]
        post_idx = file.split("_")[2]

        if self.config.sequence == "T1":
            pre_image_path = glob(os.path.join("/home/johannes/Code/ResponsePrediction/data/glioblastoma/Burdenko-Dataset/", patient_id, f"{pre_idx}*", "*mrcet1*10.pt"))
            post_image_path = glob(os.path.join("/home/johannes/Code/ResponsePrediction/data/glioblastoma/Burdenko-Dataset/", patient_id, f"{post_idx}*", "*mrcet1*10.pt"))

            images = pre_image_path + post_image_path

        elif self.config.sequence == "T2":
            pre_image_path = glob(os.path.join("/home/johannes/Code/ResponsePrediction/data/glioblastoma/Burdenko-Dataset/", patient_id, f"{pre_idx}*", "*flair*10.pt"))
            post_image_path = glob(os.path.join("/home/johannes/Code/ResponsePrediction/data/glioblastoma/Burdenko-Dataset/", patient_id, f"{post_idx}*", "*flair*10.pt"))

            images = pre_image_path + post_image_path

        elif self.config.sequence == "T1T2":

            pre_image_path = glob(os.path.join("/home/johannes/Code/ResponsePrediction/data/glioblastoma/Burdenko-Dataset/", patient_id, f"{pre_idx}*", "*mrcet1*10.pt"))
            post_image_path = glob(os.path.join("/home/johannes/Code/ResponsePrediction/data/glioblastoma/Burdenko-Dataset/", patient_id, f"{post_idx}*", "*mrcet1*10.pt"))

            images = pre_image_path + post_image_path

            pre_image_path = glob(os.path.join("/home/johannes/Code/ResponsePrediction/data/glioblastoma/Burdenko-Dataset/", patient_id, f"{pre_idx}*", "*flair*10.pt"))
            post_image_path = glob(os.path.join("/home/johannes/Code/ResponsePrediction/data/glioblastoma/Burdenko-Dataset/", patient_id, f"{post_idx}*", "*flair*10.pt"))

            images = images + pre_image_path + post_image_path

        # image = [torch.tensor(ni.load(file).get_fdata()).unsqueeze(dim=0) for file in images]
        image = [torch.load(file, weights_only=True).unsqueeze(dim=0) for file in images]
        image = torch.concatenate(image, dim=0).unsqueeze(dim=0)

        # if self.split == "train":
        #     if self.config.augmentation:
        #         image = self.tio_transform(image) 

        
        return {'data': image, 'label': torch.tensor(label)}

    def get_train_transform():
        # we now create a list of transforms. These are not necessarily the best transforms to use for BraTS, this is just
        # to showcase some things
        tr_transforms = []

        # now we mirror along all axes
        tr_transforms.append(MirrorTransform(axes=(0, 1)))

        # gamma transform. This is a nonlinear transformation of intensity values
        # (https://en.wikipedia.org/wiki/Gamma_correction)
        tr_transforms.append(GammaTransform(gamma_range=(0.5, 2), invert_image=False, per_channel=True, p_per_sample=0.15))

        # we can also invert the image, apply the transform and then invert back
        tr_transforms.append(GammaTransform(gamma_range=(0.5, 2), invert_image=True, per_channel=True, p_per_sample=0.15))

        # Gaussian Noise
        tr_transforms.append(GaussianNoiseTransform(noise_variance=(0, 0.05), p_per_sample=0.15))

        # blurring. Some BraTS cases have very blurry modalities. This can simulate more patients with this problem and
        # thus make the model more robust to it
        tr_transforms.append(GaussianBlurTransform(blur_sigma=(0.5, 1.5), different_sigma_per_channel=True, p_per_channel=0.5, p_per_sample=0.15))
        
        tr_transforms.append(ContrastAugmentationTransform((1.0, 1.75), per_channel=True, p_per_sample=0.15))

        # brightness transform for 15% of samples
        tr_transforms.append(BrightnessMultiplicativeTransform((0.7, 1.5), per_channel=True, p_per_sample=0.15))     

        # now we compose these transforms together
        tr_transforms = Compose(tr_transforms)

        return tr_transforms


class LumiereDataset(Dataset):
    def __init__(self, config, split):
        self.config = config
        self.split = split

        # self.affine = tio.transforms.RandomAffine(scales=[0.95, 1.05],
        #                                           degrees=10,
        #                                           translation=10)      
        self.flip = tio.transforms.RandomFlip(axes=(0, 1))  
        # self.blur = tio.transforms.RandomBlur(std=(0.5, 1.5))
        self.blur = tio.transforms.RandomBlur()
        # self.noise = tio.transforms.RandomNoise(std=(0, 0.05))
        self.noise = tio.transforms.RandomNoise()
        # self.gamma = tio.transforms.RandomGamma(log_gamma=(0.5, 2))
        self.gamma = tio.transforms.RandomGamma()
        self.tio_transform = tio.transforms.Compose([self.flip, self.blur, self.noise, self.gamma])

        if self.split == "train":
            data_dict = torch.load("/home/johannes/Code/ResponsePrediction/data/glioblastoma/Lumiere/train_patient_ids_lumiere.pt", weights_only=True)

            files = []
            labels = []
            for key in data_dict.keys():
                files.extend(list(data_dict[key].keys()))
                labels.extend(list(data_dict[key].values()))

            self.files = files
            self.labels = [0 if label=="progression" else 1 for label in labels]

            labels_arr = np.array(self.labels)
            self.class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels_arr), y=labels_arr)
            self.class_weights = torch.tensor(self.class_weights).to(torch.float32).to(self.config.device)
            self.sample_weights = self.class_weights[torch.tensor(labels_arr)]
        
        elif self.split == "val":
            data_dict = torch.load("/home/johannes/Code/ResponsePrediction/data/glioblastoma/Lumiere/val_patient_ids_lumiere.pt", weights_only=True)

            files = []
            labels = []
            for key in data_dict.keys():
                files.extend(list(data_dict[key].keys()))
                labels.extend(list(data_dict[key].values()))

            self.files = files
            self.labels = [0 if label=="progression" else 1 for label in labels]
        
        elif self.split == "test":
            data_dict = torch.load("/home/johannes/Code/ResponsePrediction/data/glioblastoma/Lumiere/test_patient_ids_lumiere.pt", weights_only=True)

            files = []
            labels = []
            for key in data_dict.keys():
                files.extend(list(data_dict[key].keys()))
                labels.extend(list(data_dict[key].values()))
            
            self.files = files
            self.labels = [0 if label=="progression" else 1 for label in labels]        

        self.ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(np.array([0, 1]).reshape(-1, 1))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):   

        file = self.files[idx]
        label = self.labels[idx]

        patient_id = file.split("_")[0]
        pre_idx = file.split("_")[1]
        post_idx = file.split("_")[2]

        if self.config.sequence == "T1":
            pre_image_path = glob(os.path.join("/home/johannes/Code/ResponsePrediction/data/glioblastoma/Lumiere/dataset", patient_id, f"{pre_idx}_*", "*mrcet1*10.pt"))
            post_image_path = glob(os.path.join("/home/johannes/Code/ResponsePrediction/data/glioblastoma/Lumiere/dataset", patient_id, f"{post_idx}_*", "*mrcet1*10.pt"))

            images = pre_image_path + post_image_path

        elif self.config.sequence == "T2":
            pre_image_path = glob(os.path.join("/home/johannes/Code/ResponsePrediction/data/glioblastoma/Lumiere/dataset", patient_id, f"{pre_idx}_*", "*flair*10.pt"))
            post_image_path = glob(os.path.join("/home/johannes/Code/ResponsePrediction/data/glioblastoma/Lumiere/dataset", patient_id, f"{post_idx}_*", "*flair*10.pt"))

            images = pre_image_path + post_image_path

        elif self.config.sequence == "T1T2":

            pre_image_path = glob(os.path.join("/home/johannes/Code/ResponsePrediction/data/glioblastoma/Lumiere/dataset", patient_id, f"{pre_idx}_*", "*mrcet1*10.pt"))
            post_image_path = glob(os.path.join("/home/johannes/Code/ResponsePrediction/data/glioblastoma/Lumiere/dataset", patient_id, f"{post_idx}_*", "*mrcet1*10.pt"))

            images = pre_image_path + post_image_path

            pre_image_path = glob(os.path.join("/home/johannes/Code/ResponsePrediction/data/glioblastoma/Lumiere/dataset", patient_id, f"{pre_idx}_*", "*flair*10.pt"))
            post_image_path = glob(os.path.join("/home/johannes/Code/ResponsePrediction/data/glioblastoma/Lumiere/dataset", patient_id, f"{post_idx}_*", "*flair*10.pt"))

            images = images + pre_image_path + post_image_path

        image = [torch.load(file, weights_only=True).unsqueeze(dim=0) for file in images]
        image = torch.concatenate(image, dim=0)

        if self.split == "train":
            if self.config.augmentation:
                image = self.tio_transform(image) 

        return image, label
    

class BurdenkoLumiereDataset(Dataset):
    def __init__(self, config, split):
        self.config = config
        self.split = split

        # TorchIO Augmentations
        # affine = tio.transforms.RandomAffine(scales=[0.95, 1.05], degrees=10, translation=10)  
        # blur = tio.transforms.RandomBlur()
        # noise = tio.transforms.RandomNoise()
        # gamma = tio.transforms.RandomGamma()
        # self.transforms = tio.transforms.Compose([affine, blur, noise, gamma])

        # MONAI Augmentations
        rotate = monai.transforms.RandRotate(prob=0.2, range_x=10, range_y=10, range_z=10)
        scale = monai.transforms.RandZoom(prob=0.2, min_zoom=0.7, max_zoom=1.4)
        gaussian_noise = monai.transforms.RandGaussianNoise()
        gaussian_blur = monai.transforms.RandGaussianSmooth(prob=0.2, sigma_x=(0.5, 1.0), sigma_y=(0.5, 10.0), sigma_z=(0.5, 1.0))
        contrast = monai.transforms.RandAdjustContrast()
        intensity = monai.transforms.RandScaleIntensity(factors=(2, 10))
        histogram_shift = monai.transforms.RandHistogramShift()
        self.transforms = Compose([rotate, scale, gaussian_noise, gaussian_blur, contrast, intensity, histogram_shift])


        if self.split == "train":
            data_dict_lumiere = torch.load("/home/johannes/Code/ResponsePrediction/data/glioblastoma/Lumiere/train_patient_ids_lumiere.pt", weights_only=True)
            data_dict_burdenko = torch.load("/home/johannes/Code/ResponsePrediction/data/glioblastoma/Burdenko/train_patient_ids_burdenko.pt", weights_only=True)

            files = []
            labels = []
            for key in data_dict_lumiere.keys():
                files.extend(list(data_dict_lumiere[key].keys()))
                labels.extend(list(data_dict_lumiere[key].values()))

            for key in data_dict_burdenko.keys():
                files.extend(list(data_dict_burdenko[key].keys()))
                labels.extend(list(data_dict_burdenko[key].values()))
            

            self.files = files
            self.labels = [0 if label=="progression" else 1 for label in labels]

            labels_arr = np.array(self.labels)
            self.class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels_arr), y=labels_arr)
            self.class_weights = torch.tensor(self.class_weights).to(torch.float32).to(self.config.device)
            self.sample_weights = self.class_weights[torch.tensor(labels_arr)]
        
        elif self.split == "val":
            data_dict_lumiere = torch.load("/home/johannes/Code/ResponsePrediction/data/glioblastoma/Lumiere/val_patient_ids_lumiere.pt", weights_only=True)
            data_dict_burdenko = torch.load("/home/johannes/Code/ResponsePrediction/data/glioblastoma/Burdenko/val_patient_ids_burdenko.pt", weights_only=True)

            files = []
            labels = []
            for key in data_dict_lumiere.keys():
                files.extend(list(data_dict_lumiere[key].keys()))
                labels.extend(list(data_dict_lumiere[key].values()))

            for key in data_dict_burdenko.keys():
                files.extend(list(data_dict_burdenko[key].keys()))
                labels.extend(list(data_dict_burdenko[key].values()))

            self.files = files
            self.labels = [0 if label=="progression" else 1 for label in labels]
        
        elif self.split == "test":
            data_dict_lumiere = torch.load("/home/johannes/Code/ResponsePrediction/data/glioblastoma/Lumiere/test_patient_ids_lumiere.pt", weights_only=True)
            data_dict_burdenko = torch.load("/home/johannes/Code/ResponsePrediction/data/glioblastoma/Burdenko/test_patient_ids_burdenko.pt", weights_only=True)

            files = []
            labels = []
            for key in data_dict_lumiere.keys():
                files.extend(list(data_dict_lumiere[key].keys()))
                labels.extend(list(data_dict_lumiere[key].values()))

            for key in data_dict_burdenko.keys():
                files.extend(list(data_dict_burdenko[key].keys()))
                labels.extend(list(data_dict_burdenko[key].values()))
            
            self.files = files
            self.labels = [0 if label=="progression" else 1 for label in labels]        

        self.ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(np.array([0, 1]).reshape(-1, 1))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):   

        file = self.files[idx]
        label = self.labels[idx]

        patient_id = file.split("_")[0]
        pre_idx = file.split("_")[1]
        post_idx = file.split("_")[2]

        if self.config.sequence == "T1":
            if "Burdenko" in patient_id:
                pre_image_path = glob(os.path.join("/home/johannes/Code/ResponsePrediction/data/glioblastoma/Burdenko/final_dataset", patient_id, f"{pre_idx}*", "*mrcet1*crop*.pt"))
                post_image_path = glob(os.path.join("/home/johannes/Code/ResponsePrediction/data/glioblastoma/Burdenko/final_dataset", patient_id, f"{post_idx}*", "*mrcet1*crop*.pt"))
            else:
                pre_image_path = glob(os.path.join("/home/johannes/Code/ResponsePrediction/data/glioblastoma/*/dataset", patient_id, f"{pre_idx}_*", "*mrcet1*10.pt"))
                post_image_path = glob(os.path.join("/home/johannes/Code/ResponsePrediction/data/glioblastoma/*/dataset", patient_id, f"{post_idx}_*", "*mrcet1*10.pt"))

            images = pre_image_path + post_image_path

        elif self.config.sequence == "T2":
            if "Burdenko" in patient_id:
                pre_image_path = glob(os.path.join("/home/johannes/Code/ResponsePrediction/data/glioblastoma/Burdenko/final_dataset", patient_id, f"{pre_idx}*", "*flair*crop*.pt"))
                post_image_path = glob(os.path.join("/home/johannes/Code/ResponsePrediction/data/glioblastoma/Burdenko/final_dataset", patient_id, f"{post_idx}*", "*flair*crop*.pt"))

            else:
                pre_image_path = glob(os.path.join("/home/johannes/Code/ResponsePrediction/data/glioblastoma/*/dataset", patient_id, f"{pre_idx}_*", "*flair*10.pt"))
                post_image_path = glob(os.path.join("/home/johannes/Code/ResponsePrediction/data/glioblastoma/*/dataset", patient_id, f"{post_idx}_*", "*flair*10.pt"))

            images = pre_image_path + post_image_path

        elif self.config.sequence == "T1T2":

            if "Burdenko" in patient_id:
                pre_image_path = glob(os.path.join("/home/johannes/Code/ResponsePrediction/data/glioblastoma/Burdenko/final_dataset", patient_id, f"{pre_idx}*", "*mrcet1*crop*.pt"))
                post_image_path = glob(os.path.join("/home/johannes/Code/ResponsePrediction/data/glioblastoma/Burdenko/final_dataset", patient_id, f"{post_idx}*", "*mrcet1*crop*.pt"))

                images = pre_image_path + post_image_path

                pre_image_path = glob(os.path.join("/home/johannes/Code/ResponsePrediction/data/glioblastoma/Burdenko/final_dataset", patient_id, f"{pre_idx}*", "*flair*crop*.pt"))
                post_image_path = glob(os.path.join("/home/johannes/Code/ResponsePrediction/data/glioblastoma/Burdenko/final_dataset", patient_id, f"{post_idx}*", "*flair*crop*.pt"))

                images = images + pre_image_path + post_image_path
            
            else:
                pre_image_path = glob(os.path.join("/home/johannes/Code/ResponsePrediction/data/glioblastoma/*/dataset", patient_id, f"{pre_idx}_*", "*mrcet1*10.pt"))
                post_image_path = glob(os.path.join("/home/johannes/Code/ResponsePrediction/data/glioblastoma/*/dataset", patient_id, f"{post_idx}_*", "*mrcet1*10.pt"))

                images = pre_image_path + post_image_path

                pre_image_path = glob(os.path.join("/home/johannes/Code/ResponsePrediction/data/glioblastoma/*/dataset", patient_id, f"{pre_idx}_*", "*flair*10.pt"))
                post_image_path = glob(os.path.join("/home/johannes/Code/ResponsePrediction/data/glioblastoma/*/dataset", patient_id, f"{post_idx}_*", "*flair*10.pt"))

                images = images + pre_image_path + post_image_path

        image = [torch.load(file, weights_only=True).unsqueeze(dim=0) for file in images]
        image = torch.concatenate(image, dim=0)

        if self.split == "train":
            if self.config.augmentation:
                image = self.transforms(image) 

        # image = tio.CropOrPad(target_shape=(64, 64, 64))(image)######################################

        if self.config.model_name == "ViT":
            image  = tio.CropOrPad(target_shape=(128, 128, 128))(image)

        return image, label
    

def BurdenkoLumiereCacheDataset(config, split):

    match config.sequence:
        case "T1":
            image_keys = ["pre_t1", "post_t1"]
        case "T2":
            image_keys = ["pre_t2", "post_t2"]
        case "T1T2":
            image_keys = ["pre_t1", "post_t1", "pre_t2", "post_t2"]

    load = monai.transforms.LoadImaged(keys=image_keys+["target"], ensure_channel_first=True, image_only=True)
    # load = monai.transforms.LoadImaged(keys=image_keys, ensure_channel_first=True, image_only=True)
    rotate = monai.transforms.RandRotated(prob=0.2, range_x=10, range_y=10, range_z=10, keys=image_keys)
    scale = monai.transforms.RandZoomd(prob=0.2, min_zoom=0.7, max_zoom=1.4, keys=image_keys)
    gaussian_noise = monai.transforms.RandGaussianNoised(keys=image_keys)
    gaussian_blur = monai.transforms.RandGaussianSmoothd(prob=0.2, sigma_x=(0.5, 1.0), sigma_y=(0.5, 10.0), sigma_z=(0.5, 1.0), keys=image_keys)
    contrast = monai.transforms.RandAdjustContrastd(keys=image_keys)
    intensity = monai.transforms.RandScaleIntensityd(factors=(2, 10), keys=image_keys)
    histogram_shift = monai.transforms.RandHistogramShiftd(keys=image_keys)
    to_tensor = monai.transforms.ToTensord(keys=image_keys+["target"])
    
    train_transforms = Compose([load, rotate, scale, gaussian_noise, gaussian_blur, contrast, intensity, histogram_shift, to_tensor])
    val_transforms = Compose([load, to_tensor])
    test_transforms = Compose([load, to_tensor])

    sample_weights = None

    match split:
        case "train":            
            data_dicts = torch.load("/home/johannes/Code/ResponsePrediction/data/glioblastoma/train_data_burenko_lumiere_cache_dataset_data_dict.pt", weights_only=False)
            dataset = CacheDataset(data=data_dicts, transform=train_transforms, cache_rate=1.0, cache_num=1000, num_workers=4, copy_cache=False)        

            # labels = [int(np.load(item["label"])) for item in data_dicts]
            labels = [ni.load(item["target"]).get_fdata() for item in data_dicts]
            labels = [int(label[0,0,0]) for label in labels]
            labels_arr = np.array(labels)
            class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels_arr), y=labels_arr)
            class_weights = torch.tensor(class_weights).to(torch.float32).to(config.device)
            sample_weights = class_weights[torch.tensor(labels_arr)]    

        case "val":
            data_dicts = torch.load("/home/johannes/Code/ResponsePrediction/data/glioblastoma/val_data_burenko_lumiere_cache_dataset_data_dict.pt", weights_only=False)
            dataset = CacheDataset(data=data_dicts, transform=val_transforms, cache_rate=1.0, cache_num=1000, num_workers=4, copy_cache=False)

        case "test":
            data_dicts = torch.load("/home/johannes/Code/ResponsePrediction/data/glioblastoma/test_data_burenko_lumiere_cache_dataset_data_dict.pt", weights_only=False)
            dataset = CacheDataset(data=data_dicts, transform=test_transforms, cache_rate=1.0, cache_num=1000, num_workers=4, copy_cache=False)
    
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(np.array([0, 1]).reshape(-1, 1))
    
    return dataset, sample_weights, ohe


