import SimpleITK as sitk
import torchio as tio
import os
from glob import glob
from tqdm import tqdm
import numpy as np
from pathlib import Path
import nibabel as nib
import torch

class Preprocessor:
    def __init__(self, config) -> None:
        self.config = config

        self.target_spacing = (None, None, None)
        self.target_size = (None, None, None)
        self.train_files = []
        self.test_files = []
        self.train_dir = None
        self.test_dir = None

    def get_raw_data(self) -> list:
        """TODO: Docstring"""
        
        match self.config.dataset:
            case "sarcoma":
                data_dir = "/home/johannes/Code/ResponsePrediction/data/sarcoma"                
                self.train_dir = os.path.join(data_dir, "preprocessed", f"train_{self.config.sequence}_{self.config.examination}")
                self.test_dir = os.path.join(data_dir, "preprocessed", f"test_{self.config.sequence}_{self.config.examination}")
                
                if (os.path.exists(self.train_dir)) & (os.path.exists(self.test_dir)):
                    return True
                else:
                    os.mkdir(self.train_dir)
                    os.mkdir(self.test_dir)
                
                if self.config.sequence == "T1T2":
                    sequences = ["T1", "T2"]
                else:
                    sequences = [self.config.sequence]

                if self.config.examination == "prepost":
                    examinations = ["pre", "post"]
                else:
                    examinations = [self.config.examination]

                train_files = []
                for sequence in sequences:
                    for examination in examinations:
                        train_files.extend(glob(os.path.join(data_dir, "raw", f"{sequence}_UWS_{examination}", "*")))
                self.train_files = [file for file in train_files if not "label" in file]

                test_files = []
                for sequence in sequences:
                    for examination in examinations:
                        test_files.extend(glob(os.path.join(data_dir, "raw", f"{sequence}_TUM_{examination}", "*")))
                self.test_files = [file for file in test_files if not "label" in file]

            case "glioblastoma":
                data_dir = "/home/johannes/Code/ResponsePrediction/data/glioblastoma"
                # TODO: Implement Code for Glioblastoma           

    def get_spacing(self, metric="median"):
        """TODO: Docstring"""

        spacings_x = []
        spacings_y = []
        spacings_z = []

        files = self.train_files

        for image_path in tqdm(files):
            img = sitk.ReadImage(image_path)
            spacing = img.GetSpacing()
            x = spacing[0]
            y = spacing[1]
            z = spacing[2]

            spacings_x.append(x)
            spacings_y.append(y)
            spacings_z.append(z)
    
        match metric:
            case "median":
                self.target_spacing = (np.ceil(np.median(spacings_x)), np.ceil(np.median(spacings_y)), np.ceil(np.median(spacings_z)))   
            case "mean":
                self.target_spacing = (np.ceil(np.mean(spacings_x)), np.ceil(np.mean(spacings_y)), np.ceil(np.mean(spacings_z)))   

        print(f"Target Spacing: {self.target_spacing} ({metric})")

    def resample_data(self):
        """TODO: Docstring"""

        resample = tio.transforms.Resample(target=self.target_spacing)

        # Resample training files
        files = self.train_files
        for img_path in tqdm(files):
            parent_dir = Path(img_path).parent
            img_filename = os.path.basename(img_path)
            patient_id = img_filename.split("_")[0]

            if "post" in img_path:
                therapy = img_filename.split("_")[1]
                patient_pcr_value = img_filename.split("_")[2]
                seg_filename = patient_id + "-label_" + therapy + "_" + patient_pcr_value
                seg_path = os.path.join(parent_dir, seg_filename)
            else:
                patient_pcr_value = img_filename.split("_")[1]
                seg_filename = patient_id + "-label_" + patient_pcr_value
                seg_path = os.path.join(parent_dir, seg_filename)

            img = tio.ScalarImage(img_path)
            seg = tio.LabelMap(seg_path)

            seg = tio.transforms.Resample(target=img)(seg)
            img = resample(img)
            seg = resample(seg)

            new_img_path = os.path.join(self.train_dir, img_filename).replace("nrrd", "nii.gz")
            new_seg_path = os.path.join(self.train_dir, seg_filename).replace("nrrd", "nii.gz")

            img.save(new_img_path)
            seg.save(new_seg_path)
        self.train_files = glob(os.path.join(self.train_dir, "*"))
        self.train_files = [file for file in self.train_files if not "label" in file]
        
        # Resample testing files
        files = self.test_files
        for img_path in tqdm(files):
            parent_dir = Path(img_path).parent
            img_filename = os.path.basename(img_path)
            patient_id = img_filename.split("_")[0]

            if "post" in img_path:
                therapy = img_filename.split("_")[1]
                patient_pcr_value = img_filename.split("_")[2]
                seg_filename = patient_id + "-label_" + therapy + "_" + patient_pcr_value
                seg_path = os.path.join(parent_dir, seg_filename)
            else:
                patient_pcr_value = img_filename.split("_")[1]
                seg_filename = patient_id + "-label_" + patient_pcr_value
                seg_path = os.path.join(parent_dir, seg_filename)

            img = tio.ScalarImage(img_path)
            seg = tio.LabelMap(seg_path)

            seg = tio.transforms.Resample(target=img)(seg)
            img = resample(img)
            seg = resample(seg)

            new_img_path = os.path.join(self.test_dir, img_filename).replace("nrrd", "nii.gz")
            new_seg_path = os.path.join(self.test_dir, seg_filename).replace("nrrd", "nii.gz")

            img.save(new_img_path)
            seg.save(new_seg_path)
        self.test_files = glob(os.path.join(self.test_dir, "*"))
        self.test_files = [file for file in self.test_files if not "label" in file]

    def register_data(self):
        pass

    def get_size(self, metric="median"):
        """TODO: Docstring"""
        
        size_x = []
        size_y = []
        size_z = []

        files = self.train_files

        for img_path in tqdm(files):
            img = sitk.ReadImage(img_path)
            size = img.GetSize()
            x = size[0]
            y = size[1]
            z = size[2]

            size_x.append(x)
            size_y.append(y)
            size_z.append(z)
    
        match metric:
            case "median":
                self.target_size = (int(np.ceil(np.median(size_x))), int(np.ceil(np.median(size_y))), int(np.ceil(np.median(size_z))))   
            case "mean":
                self.target_size = (int(np.ceil(np.mean(size_x))), int(np.ceil(np.mean(size_y))), int(np.ceil(np.mean(size_z)))) 
            case "min":
                self.target_size = (int(np.ceil(np.min(size_x))), int(np.ceil(np.min(size_y))), int(np.ceil(np.min(size_z)))) 
            case "max":
                self.target_size = (int(np.ceil(np.max(size_x))), int(np.ceil(np.max(size_y))), int(np.ceil(np.max(size_z)))) 

        print(f"Target Size: {self.target_size} ({metric})")

    def crop_pad(self):
        """TODO: Docstring"""        

        # Crop, Pad, Normalize Training Files
        files = self.train_files
        for img_path in tqdm(files):
            parent_dir = Path(img_path).parent
            img_filename = os.path.basename(img_path)
            patient_id = img_filename.split("_")[0]

            if "post" in img_filename:
                therapy = img_filename.split("_")[1]
                patient_pcr_value = img_filename.split("_")[2]
                seg_filename = patient_id + "-label_" + therapy + "_" + patient_pcr_value
                seg_path = os.path.join(parent_dir, seg_filename)
            else:
                patient_pcr_value = img_filename.split("_")[1]
                seg_filename = patient_id + "-label_" + patient_pcr_value
                seg_path = os.path.join(parent_dir, seg_filename)

            subject = tio.Subject(
                image=tio.ScalarImage(img_path),
                segmentation=tio.LabelMap(seg_path)
            )

            transform = tio.CropOrPad(target_shape=self.target_size, mask_name='segmentation')
            transformed = transform(subject)
            # img = tio.transforms.ZNormalization()(transformed.image)
            img = transformed.image
            seg = transformed.segmentation

            # img = tio.ScalarImage(img_path)
            # seg = tio.LabelMap(seg_path)

            # crop = tio.transforms.Crop(cropping=self.target_size)
            # normalize = tio.transforms.ZNormalization()
            # pad = tio.transforms.Pad(padding=self.target_size)

            # img = crop(img)
            # seg = crop(seg)

            # img = normalize(img)

            # img = pad(img)
            # seg = pad(seg)

            img.save(img_path.replace(".nii.gz", "_crop.nii.gz"))
            seg.save(seg_path.replace(".nii.gz", "_crop.nii.gz"))
        
        # Crop, Pad, Normalize Testing Files
        files = self.test_files
        for img_path in tqdm(files):
            parent_dir = Path(img_path).parent
            img_filename = os.path.basename(img_path)
            patient_id = img_filename.split("_")[0]

            if "post" in img_filename:
                therapy = img_filename.split("_")[1]
                patient_pcr_value = img_filename.split("_")[2]
                seg_filename = patient_id + "-label_" + therapy + "_" + patient_pcr_value
                seg_path = os.path.join(parent_dir, seg_filename)
            else:
                patient_pcr_value = img_filename.split("_")[1]
                seg_filename = patient_id + "-label_" + patient_pcr_value
                seg_path = os.path.join(parent_dir, seg_filename)

            subject = tio.Subject(
                image=tio.ScalarImage(img_path),
                segmentation=tio.LabelMap(seg_path)
            )

            transform = tio.CropOrPad(target_shape=self.target_size, mask_name='segmentation')
            transformed = transform(subject)
            # img = tio.transforms.ZNormalization()(transformed.image)
            img = transformed.image
            seg = transformed.segmentation
            
            # img = tio.ScalarImage(img_path)
            # seg = tio.LabelMap(seg_path)

            # crop = tio.transforms.Crop(cropping=self.target_size)
            # normalize = tio.transforms.ZNormalization()
            # pad = tio.transforms.Pad(padding=self.target_size)

            # img = crop(img)
            # seg = crop(seg)

            # img = normalize(img)

            # img = pad(img)
            # seg = pad(seg)

            img.save(img_path.replace(".nii.gz", "_crop.nii.gz"))
            seg.save(seg_path.replace(".nii.gz", "_crop.nii.gz"))

    def normalize(self):

        files = self.train_files
        for img_path in tqdm(files):
            img_path = img_path.replace(".nii.gz", "_crop.nii.gz")

            subject = tio.Subject(
                image=tio.ScalarImage(img_path)
            )
            img = tio.transforms.ZNormalization()(subject.image)
            img.save(img_path.replace("crop", "crop_norm"))

        
        files = self.test_files
        for img_path in tqdm(files):
            img_path = img_path.replace(".nii.gz", "_crop.nii.gz")

            subject = tio.Subject(
                image=tio.ScalarImage(img_path)
            )
            img = tio.transforms.ZNormalization()(subject.image)
            img.save(img_path.replace("crop", "crop_norm"))

    
    def save(self):
        """TODO: Docstring"""

        # Save training files as torch .pt file
        files = self.train_files
        for image_path in tqdm(files):
            image_path = image_path.replace(".nii.gz", "_crop_norm.nii.gz")
            nifti_img = nib.load(image_path)

            # Get the data as a numpy array
            nifti_data = nifti_img.get_fdata()

            # Convert the numpy array to a PyTorch tensor
            tensor_data = torch.tensor(nifti_data, dtype=torch.float32)

            # Optionally, add a channel dimension if needed
            tensor_data = tensor_data.unsqueeze(0)  # Shape: (1, H, W, D)

            torch_filename = image_path.replace("_crop_norm.nii.gz", ".pt")
            torch.save(tensor_data, torch_filename)
        
        # Save testing files as torch .pt file
        files = self.test_files
        for image_path in tqdm(files):
            image_path = image_path.replace(".nii.gz", "_crop_norm.nii.gz")
            nifti_img = nib.load(image_path)

            # Get the data as a numpy array
            nifti_data = nifti_img.get_fdata()

            # Convert the numpy array to a PyTorch tensor
            tensor_data = torch.tensor(nifti_data, dtype=torch.float32)

            # Optionally, add a channel dimension if needed
            tensor_data = tensor_data.unsqueeze(0)  # Shape: (1, H, W, D)

            torch_filename = image_path.replace("_crop_norm.nii.gz", ".pt")
            torch.save(tensor_data, torch_filename)
    
    def clean_up(self):
        """TODO: Docstring"""

        for file in glob(os.path.join(self.train_dir, "*.nii.gz")):
            os.remove(file)
        
        for file in glob(os.path.join(self.test_dir, "*.nii.gz")):
            os.remove(file)


    def __call__(self):
        if self.get_raw_data():
            return 
        else:
            self.get_spacing()
            self.resample_data()
            # self.register_data()
            self.get_size()
            self.crop_pad()
            self.normalize()
            self.save()          
            # self.clean_up()  
            return
        