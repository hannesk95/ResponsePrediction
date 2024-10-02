import random
import PIL.Image
import PIL.ImageFile
import numpy as np
import torch
import os
import subprocess
import mlflow
import matplotlib.pyplot as plt
from typing import Union, Optional, Tuple, List, Dict, Set
from vit3d import vit_tiny, vit_small
import torch.nn as nn
import timm

from PIL import Image
import PIL
from torchvision import transforms
from sklearn.metrics import confusion_matrix
from configparser import ConfigParser
from monai.networks.nets import (ResNetFeatures, DenseNet121, resnet, EfficientNetBN, 
                                SENet, SENet154, SEResNext50, SEResNet50, ViT)
from resnet_tencent import generate_model

def set_seed(seed: int) -> None:
    """TODO: Docstring"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"


def save_conda_env(config: ConfigParser) -> None:
    """TODO: Docstring"""

    try:
        conda_env = os.environ['CONDA_DEFAULT_ENV']
        command = f"conda env export -n {conda_env} > {config.run_dir}/environment.yml"
        subprocess.call(command, shell=True)
        mlflow.log_artifact(f"{config.run_dir}/environment.yml")
    except:
        print("Conda environment is not logged!")


def save_python_files(config: ConfigParser) -> None:
    """TODO: Docstring"""

    files = get_git_tracked_files(os.getcwd())
    _ = [mlflow.log_artifact(f"./{file}") for file in files]


def get_git_tracked_files(repo_dir: str) -> list:
    # Get the list of tracked files using Git
    try:
        result = subprocess.run(['git', '-C', repo_dir, 'ls-files'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            raise Exception(result.stderr)
        files = result.stdout.splitlines()
        return files
    except Exception as e:
        print(f"Error: {e}")
        return []

def create_confusion_matrix(y_true: list, y_pred: list) -> Tuple[PIL.ImageFile.ImageFile, float, float]:
    """TODO: Docstring"""

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(cm, cmap=plt.cm.Blues)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center', size='xx-large')

    plt.xlabel('Predicted Label', fontsize=18)
    plt.ylabel('True Label', fontsize=18)
    image_path = "./temp.png"
    plt.savefig(image_path)

    image = Image.open(image_path)
    transform = transforms.ToTensor()
    image_tensor = transform(image)

    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)

    os.remove(image_path)

    # return image_tensor, sensitivity, specificity
    return image, sensitivity, specificity

class WeightedCombinedLosses(nn.Module):

    def __init__(
        self, losses: list[nn.Module], weights: Optional[list[float]] = None
    ) -> None:
        super().__init__()
        self.losses = losses
        # equal weights if not provided
        self.weights = (
            weights
            if weights is not None
            else [1 / len(self.losses)] * len(self.losses)
        )

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = torch.zeros(1, device=preds.device)
        for w, l in zip(self.weights, self.losses):
            loss += w * l(preds, targets)

        return loss
    
class SoftF1LossMinorityClass(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.soft_f1_loss_fn = SoftF1LossWithLogits()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:

        # flip targets such that minority class is 1
        target = 1 - target
        pred = -pred  # flip logits such that minority class is < 0

        return self.soft_f1_loss_fn(pred, target)


class SoftF1LossWithLogits(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        tp = (target * pred).sum()
        # tn = ((1 - targets) * (1 - preds)).sum()
        fp = ((1 - target) * pred).sum()
        fn = (target * (1 - pred)).sum()

        p = tp / (tp + fp + 1e-16)
        r = tp / (tp + fn + 1e-16)

        soft_f1 = 2 * p * r / (p + r + 1e-16)

        soft_f1 = soft_f1.mean()

        return 1 - soft_f1


class SoftF1LossMulti(torch.nn.Module):

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.bin_loss_fn = SoftF1LossWithLogits()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:

        loss = torch.zeros(1, device=pred.device, dtype=pred.dtype)
        pred = torch.softmax(pred, dim=1)

        for i in range(self.num_classes):
            loss += self.bin_loss_fn(target[:, i], pred[:, i])

        loss /= self.num_classes

        return loss
    

class SoftMCCLoss(nn.Module):

    def forward(self, preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        tp = torch.sum(preds * labels)
        tn = torch.sum((1 - preds) * (1 - labels))
        fp = torch.sum(preds * (1 - labels))
        fn = torch.sum((1 - preds) * labels)

        numerator = tp * tn - fp * fn
        denom = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + 1e-8
        soft_mcc = numerator / denom

        loss = 1 - soft_mcc
        return loss


class SoftMCCWithLogitsLoss(SoftMCCLoss):

    def forward(self, preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        preds_sigmoid = torch.sigmoid(preds)
        return super().forward(preds_sigmoid, labels)


class SoftMCCLossMulti(nn.Module):
    """With logits."""

    def forward(self, preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # create soft confusion matrix
        preds = torch.softmax(preds, dim=1)

        # total number of correct predictions, softened by the probability of each class
        c = torch.sum(preds * labels)

        # total number of samples
        s = preds.size(0)

        # number of times each class occured in the labels
        t_k = torch.sum(labels, dim=0)

        # number of times each class was predicted
        p_k = torch.sum(preds, dim=0)

        numerator = c * s - (t_k * p_k).sum()
        denom = (
            torch.sqrt(s**2 - p_k.square().sum())
            * torch.sqrt(s**2 - t_k.square().sum())
            + 1e-8
        )

        soft_mcc = numerator / denom
        return 1 - soft_mcc
    
def replace_batchnorm_with_instancenorm(model):
    """
    Recursively replace all BatchNorm layers (BatchNorm1d, BatchNorm2d, BatchNorm3d)
    with InstanceNorm layers (InstanceNorm1d, InstanceNorm2d, InstanceNorm3d).
    
    Args:
        model (nn.Module): The model in which to replace BatchNorm layers.

    Returns:
        model (nn.Module): The modified model with InstanceNorm layers.
    """
    for name, module in model.named_children():
        # Replace BatchNorm1d with InstanceNorm1d
        if isinstance(module, nn.BatchNorm1d):
            setattr(model, name, nn.InstanceNorm1d(module.num_features, affine=module.affine, track_running_stats=False))
        # Replace BatchNorm2d with InstanceNorm2d
        elif isinstance(module, nn.BatchNorm2d):
            setattr(model, name, nn.InstanceNorm2d(module.num_features, affine=module.affine, track_running_stats=False))
        # Replace BatchNorm3d with InstanceNorm3d
        elif isinstance(module, nn.BatchNorm3d):
            setattr(model, name, nn.InstanceNorm3d(module.num_features, affine=module.affine, track_running_stats=False))
        
        # Recursively apply to submodules
        replace_batchnorm_with_instancenorm(module)
    
    return model

def get_model(config, output_neurons: int):

    match config.model_name:
        case "ResNet":
            match config.model_depth:
                case 10:
                    pretrain_path = None
                    if config.pretrained:
                        pretrain_path = "./data/sarcoma/jan/pretrain/resnet_10_23dataset.pth"
                    model = generate_model(model_depth=10, in_channels=config.channels, num_cls_classes=output_neurons, pretrain_path=pretrain_path).to(config.device)                     
                case 18:
                    if config.pretrained:
                        model = ResNetFeatures('resnet18', pretrained=True, spatial_dims=3, in_channels=1).to(config.device)
                        model.conv1 = torch.nn.Sequential(
                            torch.nn.Conv3d(config.channels, 1, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0),bias=False),
                            torch.nn.BatchNorm3d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                            torch.nn.ReLU(inplace=True),
                            model.conv1
                            ).to(config.device)

                        model = torch.nn.Sequential(model, 
                            torch.nn.AdaptiveAvgPool3d(output_size=(1, 1, 1)),
                            torch.nn.Flatten(start_dim=1, end_dim=-1),
                            torch.nn.Linear(in_features=512, out_features=output_neurons, bias=True),
                            # torch.nn.Dropout(0.4),
                            # torch.nn.Linear(in_features=512, out_features=512, bias=True),
                            # torch.nn.Dropout(0.4),
                            # torch.nn.Linear(in_features=512, out_features=256, bias=True),
                            # torch.nn.Dropout(0.4),
                            # torch.nn.Linear(in_features=256, out_features=128, bias=True),
                            # torch.nn.Dropout(0.4),
                            # torch.nn.Linear(in_features=128, out_features=output_neurons, bias=True)
                            ).to(config.device)                      

                    else:
                        model = resnet.resnet18(spatial_dims=3, n_input_channels=config.channels, num_classes=output_neurons).to(config.device)    
                        # model.fc = torch.nn.Sequential(
                        #     torch.nn.Flatten(start_dim=1, end_dim=-1),
                        #     torch.nn.Linear(in_features=model.fc.in_features, out_features=model.fc.out_features, bias=True),
                        #     # torch.nn.Dropout(0.4),
                        #     # torch.nn.Linear(in_features=512, out_features=512, bias=True),
                        #     # torch.nn.Dropout(0.4),
                        #     # torch.nn.Linear(in_features=512, out_features=256, bias=True),
                        #     # torch.nn.Dropout(0.4),
                        #     # torch.nn.Linear(in_features=256, out_features=128, bias=True),
                        #     # torch.nn.Dropout(0.4),
                        #     # torch.nn.Linear(in_features=128, out_features=model.fc.out_features, bias=True)
                        #     ).to(config.device)
                
                case 34:
                    pretrain_path = None
                    if config.pretrained:
                        pretrain_path = "./data/sarcoma/jan/pretrain/resnet_34_23dataset.pth"
                    # model = resnet.resnet34(spatial_dims=3, n_input_channels=num_channels, num_classes=output_neurons).to(config.device)
                    model = generate_model(model_depth=34, in_channels=config.channels, num_cls_classes=output_neurons, pretrain_path=pretrain_path).to(config.device) 
                case 50:
                    pretrain_path = None
                    if config.pretrained:
                        pretrain_path = "./data/sarcoma/jan/pretrain/resnet_50_23dataset.pth"
                    # model = resnet.resnet50(spatial_dims=3, n_input_channels=num_channels, num_classes=output_neurons).to(config.device)  
                    model = generate_model(model_depth=50, in_channels=config.channels, num_cls_classes=output_neurons, pretrain_path=pretrain_path).to(config.device)
                case 101:
                    pretrain_path = None
                    if config.pretrained:
                        pretrain_path = "./data/sarcoma/jan/pretrain/pretrain/resnet_101.pth"
                    # model = resnet.resnet101(spatial_dims=3, n_input_channels=num_channels, num_classes=output_neurons).to(config.device)  
                    model = generate_model(model_depth=101, in_channels=config.channels, num_cls_classes=output_neurons, pretrain_path=pretrain_path).to(config.device)
                case 152:
                    pretrain_path = None
                    if config.pretrained:
                        pretrain_path = "./data/sarcoma/jan/pretrain/pretrain/resnet_152.pth"
                    # model = resnet.resnet152(spatial_dims=3, n_input_channels=num_channels, num_classes=output_neurons).to(config.device)    
                    model = generate_model(model_depth=152, in_channels=config.channels, num_cls_classes=output_neurons, pretrain_path=pretrain_path).to(config.device)
        
        case "DenseNet":
            match config.model_depth:
                case 121:
                    model = DenseNet121(spatial_dims=3, in_channels=config.channels, out_channels=output_neurons, block_config=(6, 12, 24), 
                                        dropout_prob=0.5, act=('relu', {'inplace': True}), norm='batch').to(config.device)
                    model.class_layers = torch.nn.Sequential(
                        torch.nn.ReLU(inplace=True),      
                        torch.nn.AdaptiveAvgPool3d(output_size=1),
                        torch.nn.Flatten(start_dim=1, end_dim=-1),
                        torch.nn.Linear(in_features=model.class_layers.out.in_features, out_features=model.class_layers.out.out_features, bias=True),
                        # torch.nn.Dropout(0.4),
                        # torch.nn.Linear(in_features=512, out_features=512, bias=True),
                        # torch.nn.Dropout(0.4),
                        # torch.nn.Linear(in_features=512, out_features=256, bias=True),
                        # torch.nn.Dropout(0.4),
                        # torch.nn.Linear(in_features=256, out_features=128, bias=True),
                        # torch.nn.Dropout(0.4),
                        # torch.nn.Linear(in_features=128, out_features=model.class_layers.out.out_features, bias=True)
                        ).to(config.device)    

        case "SENet":
            match config.model_depth:
                case 154:
                    model = SENet154(spatial_dims=3, in_channels=config.channels, layers=(3, 8, 36, 3), num_classes=config.n_classes).to(config.device)
                    # model = SENet(spatial_dims=3, in_channels=config.channels, block=block, num_classes=config.n_classes).to(config.device)

        case "SEResNet":
            match config.model_depth:
                case 50:
                    model = SEResNet50(spatial_dims=3, in_channels=config.channels, layers=(3, 4, 6, 3), num_classes=config.n_classes).to(config.device)
                    # model = SENet(spatial_dims=3, in_channels=config.channels, block=block, num_classes=config.n_classes).to(config.device)

        case "SEResNext":
            match config.model_depth:
                case 50:
                    model = SEResNext50(spatial_dims=3, in_channels=config.channels, layers=(3, 4, 6, 3), num_classes=config.n_classes).to(config.device)
                    # model = SENet(spatial_dims=3, in_channels=config.channels, block=block, num_classes=config.n_classes).to(config.device)

        case "EfficientNet":
            match config.model_depth:
                case 0:
                    model = EfficientNetBN("efficientnet-b0", spatial_dims=3, in_channels=config.channels, num_classes=config.n_classes).to(config.device)                      
                case 1:
                    model = EfficientNetBN("efficientnet-b1", spatial_dims=3, in_channels=config.channels, num_classes=config.n_classes).to(config.device)                      
                case 2:
                    model = EfficientNetBN("efficientnet-b2", spatial_dims=3, in_channels=config.channels, num_classes=config.n_classes).to(config.device)                      
                case 3:
                    model = EfficientNetBN("efficientnet-b3", spatial_dims=3, in_channels=config.channels, num_classes=config.n_classes).to(config.device)                      

        case "ViT":
            match config.model_depth:
                case "monai":
                    model = ViT(in_channels=config.channels, img_size=(128, 128, 128), patch_size=16, num_classes=config.n_classes, 
                                num_layers=4, num_heads=6, classification=True).to(config.device)
                case "timm":
                    model = timm.create_model('tiny_vit_5m_224', pretrained=False).to(config.device)                
                case "tiny":
                    encoder = vit_tiny(img3d_size=128, img3d_frame=128, patch3d_size=16, patch3d_frame=16)
                    proj_in = 192
                    proj_hidden = 256

                    proj_v = nn.Sequential(
                        nn.Linear(proj_in, proj_hidden, bias=False),  # proj_in -> proj_hidden
                        nn.GELU(),
                        nn.Linear(proj_hidden, config.n_classes, bias=False),  # proj_hidden -> num_classes
                    )

                    model = nn.Sequential(encoder, proj_v).to(config.device)                    

                case "small":
                    encoder = vit_small(img3d_size=224, img3d_frame=128, patch3d_size=16, patch3d_frame=16)
                    proj_in = 384
                    proj_hidden = 256

                    proj_v = nn.Sequential(
                        nn.Linear(proj_in, proj_hidden, bias=False),  # proj_in -> proj_hidden
                        nn.GELU(),
                        nn.Linear(proj_hidden, config.n_classes, bias=False),  # proj_hidden -> num_classes
                    )

                    model = nn.Sequential(encoder, proj_v).to(config.device)   
        
        case _:
            raise ValueError(f"Given model name '{config.model_name}' or model depth '{config.model_depth}' is not implemented!")  
        
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
    return model, trainable_params
