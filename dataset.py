import os
import torch
import scipy
import numpy as np
import torchio as tio

from glob import glob
from pathlib import Path
from torch.utils.data import Dataset
from scipy.ndimage import map_coordinates, gaussian_filter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight


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
        self.tio_transform = tio.transforms.RandomAffine(scales=0,
                                                         degrees=0,
                                                         translation=5)
        
        match config.dataset:
            case "sarcoma":
                self.data_dir = "/home/johannes/Code/ResponsePrediction/data/sarcoma/jan/Combined"
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
        self.labels = [int(os.path.basename(filepath).split("_")[-1].replace(".pt", ""))
                       for filepath in sorted(glob(os.path.join(self.data_dir, "*.pt")))][::4]
        
        
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

        patient_id = self.patient_ids[idx]        
        
        all_patient_files = sorted([file for file in self.data if os.path.basename(file).startswith(patient_id)])             

        images = [torch.load(path, weights_only=True) for path in all_patient_files]
        image = torch.concatenate(images, dim=0)
        label = torch.tensor(int(os.path.basename(all_patient_files[0]).split("_")[-1].replace(".pt", ""))).to(torch.long)     

        if self.split == "train":
            if self.config.augmentation:
                image = self.tio_transform(image)     

        return image, label, patient_id
    

class CENDataset(Dataset):
    def __init__(self, config, split):
        self.config = config
        self.split = split
        self.data_dir = "/home/johannes/Code/ResponsePrediction/data/sarcoma/jan/Combined"
        self.tio_transform = tio.transforms.RandomAffine(scales=0,
                                                         degrees=0,
                                                         translation=5)
        
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

        # example1 = torch.ones(size=(1, 64, 64, 64)).cuda()
        # example2 = torch.zeros(size=(1, 64, 64, 64)).cuda()
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

            return t1_pre_images.cuda(), t2_pre_images.cuda(), label.cuda()

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

            return t1_post_images.cuda(), t2_post_images.cuda(), label.cuda()

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

            return t1_pre_images.cuda(), t1_post_images.cuda(), label.cuda()

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

            return t2_pre_images.cuda(), t2_post_images.cuda(), label.cuda()
        
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

            return pre_image.cuda(), post_image.cuda(), label.cuda()


        
        
