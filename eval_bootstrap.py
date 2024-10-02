import torch
import numpy as np
from param_configurator import ParamConfigurator
from dataset import BurdenkoLumiereDataset
from dataset import BurdenkoLumiereCacheDataset
from monai.data import DataLoader
from monai.data import ThreadDataLoader
from monai.transforms import PadListDataCollate
from utils import get_model
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


def main(config: ParamConfigurator, model_path: str) -> None:

    n_bootstraps = 1000
    test_bacc_list = []
    test_auc_list = []
    test_mcc_list = []
    test_f1_list = []

    for i in tqdm(range(n_bootstraps)):
    
        match config.dataloader:
            case "torch":
                test_dataset = BurdenkoLumiereDataset(config=config, split="test") 
                test_indices = torch.randint(0, len(test_dataset), (len(test_dataset),))
                test_dataset = torch.utils.data.Subset(test_dataset, test_indices)
                test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4, collate_fn=PadListDataCollate())
            case "monai":
                test_dataset, _, _ = BurdenkoLumiereCacheDataset(config=config, split="test")
                test_indices = torch.randint(0, len(test_dataset), (len(test_dataset),))
                test_dataset = torch.utils.data.Subset(test_dataset, test_indices)
                test_loader = ThreadDataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4, collate_fn=PadListDataCollate())  

        model, _ = get_model(config=config, output_neurons=config.n_classes)
        model.load_state_dict(torch.load(model_path, weights_only=True))

        model.eval()        
        with torch.no_grad():
            test_true = []
            test_pred = []
            test_prob = []
        
            for test_data in test_loader:
                
                if config.dataloader == "monai":
                        test_labels = test_data["target"][:,0,0,0,0].to(torch.long).view(-1, 1).to(config.device)
                        del test_data["target"]
                        test_images = torch.concat([tensor for tensor in test_data.values() if isinstance(tensor, torch.Tensor)], dim=1).to(config.device)

                elif config.dataloader == "torch":
                    test_images = test_data[0].to(torch.float32).to(config.device)                  
                    test_labels = test_data[1].to(torch.long).to(config.device)    

                with torch.amp.autocast(device_type=config.device, dtype=torch.float16):
                    test_outputs = model(test_images)                        
             
                test_true.extend(test_labels.cpu().numpy())                
                test_prob.extend(test_outputs.softmax(dim=1)[:,1].detach().cpu().tolist())
                test_pred.extend(test_outputs.argmax(dim=1).cpu().tolist())               
            
        test_bacc = balanced_accuracy_score(test_true, test_pred)
        test_auc = roc_auc_score(test_true, test_prob)
        test_mcc = matthews_corrcoef(test_true, test_pred)
        test_f1 = f1_score(test_true, test_pred, average="micro")

        test_bacc_list.append(test_bacc)
        test_auc_list.append(test_auc)
        test_mcc_list.append(test_mcc)
        test_f1_list.append(test_f1)
    
    # Calculate mean and std
    test_bacc_mean = np.mean(test_bacc_list)
    test_bacc_std = np.std(test_bacc_list)
    print(f"\nMean Test bACC: {test_bacc_mean}")
    print(f"Std Test bACC: {test_bacc_std}\n")
                
    test_auc_mean = np.mean(test_auc_list)
    test_auc_std = np.std(test_auc_list)
    print(f"\nMean Test AUROC: {test_auc_mean}")
    print(f"Std Test AUROC: {test_auc_std}\n")

    test_mcc_mean = np.mean(test_mcc_list)
    test_mcc_std = np.std(test_mcc_list)
    print(f"\nMean Test MCC: {test_mcc_mean}")
    print(f"Std Test MCC: {test_mcc_std}\n")

    test_f1_mean = np.mean(test_f1_list)
    test_f1_std = np.std(test_f1_list)
    print(f"\nMean Test F1: {test_f1_mean}")
    print(f"Std Test F1: {test_f1_std}\n")


if __name__ == "__main__":

    config = ParamConfigurator()
    config.model_name = "DenseNet"
    config.model_depth = 121
    config.dataloader = "torch"
    config.examination = "prepost"
    config.sequence = "T1T2" 
    config.channels = 4

    model_path = "/home/johannes/Code/ResponsePrediction/mlruns/641677121539180496/a8d4f64b7668475ca464ed60c9c1cf75/artifacts/best_metric_model_56b192bce2094e37b70f107396253755.pth"

    main(config=config, model_path=model_path)
