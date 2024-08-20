import random
import numpy as np
import torch
import os
import subprocess
import mlflow
import matplotlib.pyplot as plt
from typing import Optional
import torch.nn as nn

from PIL import Image
from torchvision import transforms
from sklearn.metrics import confusion_matrix

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


def save_conda_env(config) -> None:
    """TODO: Docstring"""

    try:
        conda_env = os.environ['CONDA_DEFAULT_ENV']
        command = f"conda env export -n {conda_env} > {config.run_dir}/environment.yml"
        subprocess.call(command, shell=True)
        mlflow.log_artifact(f"{config.run_dir}/environment.yml")
    except:
        print("Conda environment is not logged!")


def create_confusion_matrix(y_true, y_pred):
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
    

# binary versions of the loss
class SoftMCCLoss(nn.Module):

    def forward(self, preds: torch.Tensor, labels: torch.Tensor):
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

    def forward(self, preds: torch.Tensor, labels: torch.Tensor):
        preds_sigmoid = torch.sigmoid(preds)
        return super().forward(preds_sigmoid, labels)


# multi-class versions of the loss
class SoftMCCLossMulti(nn.Module):
    """With logits."""

    def forward(self, preds: torch.Tensor, labels: torch.Tensor):
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