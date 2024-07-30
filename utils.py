import random
import numpy as np
import torch
import os
import subprocess
import mlflow

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

    conda_env = os.environ['CONDA_DEFAULT_ENV']
    command = f"conda env export -n {conda_env} > {config.run_dir}/environment.yml"
    subprocess.call(command, shell=True)
    mlflow.log_artifact(f"{config.run_dir}/environment.yml")