import os
import torch
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torchio as tio
from glob import glob
from tqdm import tqdm
from radiomics import featureextractor
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, matthews_corrcoef
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize

import radiomics
radiomics.setVerbosity(60) # Quiet mode, no messages are printed to the stderr
radiomics.setVerbosity(50) # Only log messages of level “CRITICAL” are printed

def extract_features(mri_path: str, mask_path: str) -> pd.Series:
    params = {}  # Use default parameters or customize it
    extractor = featureextractor.RadiomicsFeatureExtractor(params)

    mri = tio.ScalarImage(mri_path)
    mask = tio.LabelMap(mask_path)

    mask = tio.Resample(target=mri)(mask)

    mri = mri.as_sitk()
    mask = mask.as_sitk()
    
    maximum = np.max(sitk.GetArrayFromImage(mask))
    mask = mask // maximum    
    
    # Extract features from the MRI volume
    feature_vector = extractor.execute(mri, mask)
    
    # Convert the feature vector to a dictionary and then to a pandas Series
    feature_series = pd.Series({k: float(v) for k, v in feature_vector.items() if isinstance(v, (int, float, np.ndarray)) and (k.startswith("original"))})
    
    return feature_series

def get_files(split: str) -> list:
    data_dict = torch.load(f"/media/johannes/WD Elements/Burdenko-GBM-Progression/manifest-1679410600140/{split}_patient_ids.pt", weights_only=True)

    files = []
    labels = []
    for key in data_dict.keys():
        files.extend(list(data_dict[key].keys()))
        labels.extend(list(data_dict[key].values()))


    mri_files = []
    mri_masks = []

    for file in tqdm(sorted(files)):
        patient_id = file.split("_")[0]
        pre_idx = file.split("_")[1]
        post_idx = file.split("_")[2]

        pre_image_path_t1 = glob(os.path.join("/home/johannes/Code/ResponsePrediction/data/glioblastoma/Burdenko-Dataset/", patient_id, f"{pre_idx}*", "*mrcet1*resampled.nii.gz"))[0]
        post_image_path_t1 = glob(os.path.join("/home/johannes/Code/ResponsePrediction/data/glioblastoma/Burdenko-Dataset/", patient_id, f"{post_idx}*", "*mrcet1*resampled.nii.gz"))[0]
        pre_image_path_flair = glob(os.path.join("/home/johannes/Code/ResponsePrediction/data/glioblastoma/Burdenko-Dataset/", patient_id, f"{pre_idx}*", "*flair*resampled.nii.gz"))[0]
        post_image_path_flair = glob(os.path.join("/home/johannes/Code/ResponsePrediction/data/glioblastoma/Burdenko-Dataset/", patient_id, f"{post_idx}*", "*flair*resampled.nii.gz"))[0]
        mask = glob(os.path.join("/home/johannes/Code/ResponsePrediction/data/glioblastoma/Burdenko-Dataset/", patient_id, "*", "mask*resampled.nii.gz"))[0]

        mri_files.append(pre_image_path_t1)
        mri_files.append(post_image_path_t1)
        mri_files.append(pre_image_path_flair)
        mri_files.append(post_image_path_flair)

        mri_masks.append(mask)
        mri_masks.append(mask)
        mri_masks.append(mask)
        mri_masks.append(mask)

    return mri_files, mri_masks, labels


def main():

    for split in ["train", "val", "test"]:

        mri_files, mri_masks, labels = get_files(split=split)
        
        # Extract features for all MRI volumes
        features = []
        for mri_path, mask_path in tqdm(zip(mri_files, mri_masks), total=len(mri_files)):
            features.append(extract_features(mri_path, mask_path))

        # Create a DataFrame of extracted features
        X = pd.DataFrame(features)
        y = np.array(labels)

        torch.save(features, f"{split}_features.pt")
        torch.save(X, f"{split}_X.pt")
        torch.save(y, f"{split}_y.pt")

if __name__ == "__main__":
    main()