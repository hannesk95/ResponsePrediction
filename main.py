import os
import gc
import uuid
import monai.transforms
import torch
import monai
import mlflow
import numpy as np
import matplotlib.pyplot as plt


from tqdm import tqdm
from datetime import datetime
from utils import save_conda_env
from utils import save_python_files
from utils import create_confusion_matrix
from utils import replace_batchnorm_with_instancenorm
from dataset import ResNetDataset
from dataset import CombinedDataset
from dataset import BurdenkoDataset, BurdenkoDatasetDKFZ
from dataset import LumiereDataset
from dataset import BurdenkoLumiereDataset
from dataset import BurdenkoLumiereCacheDataset
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from preprocessor import Preprocessor
from postprocessor import Postprocessor
from monai.networks.nets import resnet
from monai.networks.nets import ResNetFeatures
from monai.optimizers import Novograd
from monai.transforms import PadListDataCollate
from monai.transforms import CutMix

from monai.networks.nets import DenseNet121
from monai.data import DataLoader, ThreadDataLoader

# from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from param_configurator import ParamConfigurator

from sklearn.metrics import roc_auc_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score

from sklearn.metrics import r2_score
from sklearn.metrics import root_mean_squared_error

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from utils import WeightedCombinedLosses
from utils import SoftF1LossMulti
from utils import SoftMCCWithLogitsLoss
from utils import get_model
from resnet_tencent import generate_model


def main(config) -> None:

    torch.cuda.empty_cache()
    gc.collect()
    save_conda_env(config)
    save_python_files(config)
    mlflow.log_params(config.__dict__)    


    # Create Dataset
    match config.dataset:
        case "sarcoma":
            # train_dataset = ResNetDataset(config=config, split="train")
            train_dataset = CombinedDataset(config=config, split="train")

            # val_dataset = ResNetDataset(config=config, split="val")
            val_dataset = CombinedDataset(config=config, split="val")

            # test_dataset = ResNetDataset(config=config, split="test")             
            test_dataset = CombinedDataset(config=config, split="test")  
        
        case "glioblastoma":
            if config.dataloader == "torch":
                train_dataset = BurdenkoLumiereDataset(config=config, split="train")
                sample_weights = train_dataset.sample_weights
                ohe = train_dataset.ohe
                val_dataset = BurdenkoLumiereDataset(config=config, split="val")
                test_dataset = BurdenkoLumiereDataset(config=config, split="test") 

                sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)##################################################################################
                train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False, sampler=sampler, drop_last=True, num_workers=4, collate_fn=PadListDataCollate())
                val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4, collate_fn=PadListDataCollate())
                test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4, collate_fn=PadListDataCollate())

            elif config.dataloader == "monai":
                train_dataset, sample_weights, ohe = BurdenkoLumiereCacheDataset(config=config, split="train")
                val_dataset, _, _ = BurdenkoLumiereCacheDataset(config=config, split="val")
                test_dataset, _, _ = BurdenkoLumiereCacheDataset(config=config, split="test")
                
                sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
                train_loader = ThreadDataLoader(train_dataset,batch_size=config.batch_size, shuffle=False, sampler=sampler, num_workers=4, drop_last=True, collate_fn=PadListDataCollate())
                val_loader = ThreadDataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4, collate_fn=PadListDataCollate())
                test_loader = ThreadDataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4, collate_fn=PadListDataCollate())             
    
    # Loss function
    imbalance_loss = {"MCC": SoftMCCWithLogitsLoss(), "F1": SoftF1LossMulti(num_classes=config.n_classes)}

    weights = None  
    if config.imbalance == "weight":
        weights = train_dataset.class_weights                
    
    loss_function = WeightedCombinedLosses([torch.nn.CrossEntropyLoss(), imbalance_loss[config.imbalance_loss]], weights=weights)       

    
    # Get Model Architecture and Depth
    model, trainable_params = get_model(config=config, output_neurons=config.n_classes)
    mlflow.log_param("trainable_params", trainable_params)
    
    # Create Optimizer
    match config.optimizer:
        case "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-3)
        case "Adam":            
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.1)
        case "AdamW":            
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.1)
        case "Novograd":
            optimizer = Novograd(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    # Create Learning Rate Scheduler
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.scheduler_step, gamma=config.scheduler_gamma)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)

    # Gradient accumulation and AMP
    scaler = torch.amp.GradScaler('cuda')

    ##############################################################
    # Training Loop
    ##############################################################
    num_epochs = config.epochs
    best_metric = -1

    best_model_id = uuid.uuid4().hex
    for epoch in range(num_epochs):
        epoch = epoch + 1
        # print(f"Epoch {epoch}/{num_epochs}")
        
        model.train()
        train_true = []
        train_pred = []
        train_prob = []
        train_epoch_loss = 0
        lr = scheduler.get_last_lr()[0]
        # cm = CutMix(batch_size=config.batch_size, alpha=0.5)
        with tqdm(train_loader, unit="batch") as tepoch:        
            for batch_idx, batch_data in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch}/{num_epochs}")

                if config.dataloader == "monai":

                    labels = batch_data["target"][:,0,0,0,0].to(torch.long).view(-1, 1).to(config.device)
                    del batch_data["target"]
                    # labels = torch.tensor(batch_data.pop("target")[:,0,0,0,0]).to(torch.long).view(-1, 1).to(config.device)
                    inputs = torch.concat([tensor for tensor in batch_data.values() if isinstance(tensor, torch.Tensor)], dim=1).to(config.device)

                elif config.dataloader == "torch":
                    inputs = batch_data[0].to(torch.float32).to(config.device)                  
                    labels = batch_data[1].to(torch.long).to(config.device)        

                with torch.amp.autocast(device_type=config.device, dtype=torch.float16):     
                    
                    # inputs_aug = cm(inputs)
                    if config.model_name == "ViT":
                        if config.model_depth == "monai":
                            outputs = model(inputs)[0]           
                        else: 
                            outputs = model(inputs)         
                    else:
                        outputs = model(inputs)                    

                    if config.task == "classification":
                        labels_ohe = labels.cpu().numpy().reshape(-1, 1)
                        labels_ohe = ohe.transform(labels_ohe)
                        labels_ohe = torch.tensor(labels_ohe).to(config.device)
                        loss = loss_function(outputs, labels_ohe)
                        # loss = loss_function(outputs, labels)
                        loss = loss / config.accumulation_steps
                    elif config.task == "regression":
                        loss = loss_function(outputs.flatten(), labels)
                
                scaler.scale(loss).backward()   

                if (batch_idx + 1) % config.accumulation_steps == 0:  
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                train_epoch_loss += loss.item() * config.accumulation_steps
                train_true.extend(labels.cpu().tolist())

                match config.task:
                    case "classification":                    
                        train_prob.extend(outputs.softmax(dim=1)[:,1].detach().cpu().tolist())
                        train_pred.extend(outputs.argmax(dim=1).cpu().tolist())
                    case "regression":
                        train_pred.extend(outputs.flatten().detach().cpu().tolist())
        
        train_loss = train_epoch_loss/len(train_loader)
        scheduler.step()
        mlflow.log_metric("learning_rate", lr, step=epoch)
        match config.task:
            case "classification":
                train_bacc = balanced_accuracy_score(train_true, train_pred)
                # train_auc = roc_auc_score(train_true, train_prob)########################
       
                train_mcc = matthews_corrcoef(train_true, train_pred)
                train_f1 = f1_score(train_true, train_pred, average="micro")
            case "regression":
                train_r2 = r2_score(train_true, train_pred)
                train_rmse = root_mean_squared_error(train_true, train_pred)*100
                # train_rmse = root_mean_squared_error(train_true, train_pred)
                # train_rmse = train_dataset.scaler.inverse_transform(np.array(train_rmse).reshape(-1, 1)).flatten().item()

        ###################################################################
        # Validation
        ###################################################################

        model.eval()
        with torch.no_grad():
            val_true = []
            val_pred = []
            val_prob = []
            val_epoch_loss = 0
            for val_data in val_loader:      

                if config.dataloader == "monai":
                    val_labels = val_data["target"][:,0,0,0,0].to(torch.long).view(-1, 1).to(config.device)
                    del val_data["target"]
                    val_images = torch.concat([tensor for tensor in val_data.values() if isinstance(tensor, torch.Tensor)], dim=1).to(config.device)

                elif config.dataloader == "torch":
                    val_images = val_data[0].to(torch.float32).to(config.device)                  
                    val_labels = val_data[1].to(torch.long).to(config.device)       

                with torch.amp.autocast(device_type=config.device, dtype=torch.float16): 
                    # val_outputs = model(val_images) 

                    if config.model_name == "ViT":
                        if config.model_depth == "monai":
                            val_outputs = model(val_images)[0]           
                        else: 
                            val_outputs = model(val_images)                   
                    else:
                        val_outputs = model(val_images)               

                    if config.task == "classification":
                        val_labels_ohe = val_labels.cpu().numpy().reshape(-1, 1)
                        val_labels_ohe = ohe.transform(val_labels_ohe)
                        val_labels_ohe = torch.tensor(val_labels_ohe).to(config.device)
                        val_loss = loss_function(val_outputs, val_labels_ohe)
                    elif config.task == "regression":
                        val_loss = loss_function(val_outputs.flatten(), val_labels)
                
                val_epoch_loss += val_loss.item()
                val_true.extend(val_labels.cpu().tolist())

                match config.task:
                    case "classification":                        
                        val_prob.extend(val_outputs.softmax(dim=1)[:,1].detach().cpu().tolist())
                        val_pred.extend(val_outputs.argmax(dim=1).cpu().tolist())
                    case "regression":
                        val_pred.extend(val_outputs.flatten().cpu().tolist())                            
            
            val_loss = val_epoch_loss/len(val_loader)
            match config.task:
                case "classification":
                    val_bacc = balanced_accuracy_score(val_true, val_pred)
                    # val_auc = roc_auc_score(val_true, val_prob)         ###############################

                    val_mcc = matthews_corrcoef(val_true, val_pred)
                    val_f1 = f1_score(val_true, val_pred, average="micro")
                case "regression":
                    val_r2 = r2_score(val_true, val_pred)
                    val_rmse = root_mean_squared_error(val_true, val_pred)*100
                    # val_rmse = root_mean_squared_error(val_true, val_pred)
                    # train_rmse = train_dataset.scaler.inverse_transform(np.array(val_rmse).reshape(-1, 1)).flatten().item()

        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)

        match config.task:
            case "classification":
                mlflow.log_metric("train_bacc", train_bacc, step=epoch)
                mlflow.log_metric("val_bacc", val_bacc, step=epoch)
                # mlflow.log_metric("train_auroc", train_auc, step=epoch)##################
                # mlflow.log_metric("val_auroc", val_auc, step=epoch)#######################
                mlflow.log_metric("train_mcc", train_mcc, step=epoch)
                mlflow.log_metric("val_mcc", val_mcc, step=epoch)
                mlflow.log_metric("train_f1", train_f1, step=epoch)
                mlflow.log_metric("val_f1", val_f1, step=epoch)             
                
                # Training Confusion Matrix
                cm = confusion_matrix(train_true, train_pred)                
                disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                disp.plot()
                plt.savefig(f'./artifacts/train_confusion_matrix_{str(epoch).zfill(3)}.png')
                mlflow.log_artifact(f'./artifacts/train_confusion_matrix_{str(epoch).zfill(3)}.png')
                os.remove(f'./artifacts/train_confusion_matrix_{str(epoch).zfill(3)}.png') 
                plt.close()               

                # Validation Confusion Matrix
                cm = confusion_matrix(val_true, val_pred)                
                disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                disp.plot()
                plt.savefig(f'./artifacts/val_confusion_matrix_{str(epoch).zfill(3)}.png')      
                mlflow.log_artifact(f'./artifacts/val_confusion_matrix_{str(epoch).zfill(3)}.png')
                os.remove(f'./artifacts/val_confusion_matrix_{str(epoch).zfill(3)}.png')          
                plt.close()   

                if val_mcc > best_metric:
                    best_metric = val_mcc
                    torch.save(model.state_dict(), f"./artifacts/best_metric_model_{best_model_id}.pth")
                    print("[INFO] Saved new best metric model")
                elif epoch == 1:
                    torch.save(model.state_dict(), f"./artifacts/best_metric_model_{best_model_id}.pth")


            case "regression":
                mlflow.log_metric("train_r2", train_r2, step=epoch)
                mlflow.log_metric("val_r2", val_r2, step=epoch)
                mlflow.log_metric("train_rmse", train_rmse, step=epoch)
                mlflow.log_metric("val_rmse", val_rmse, step=epoch)

                if val_r2 > best_metric:
                    best_metric = val_r2
                    torch.save(model.state_dict(), f"./artifacts/best_metric_model_{best_model_id}.pth")
                    print("[INFO] Saved new best metric model")
                elif epoch == 1:
                    torch.save(model.state_dict(), f"./artifacts/best_metric_model_{best_model_id}.pth")
        
    ######################################################################
    # Test
    ######################################################################
    model.eval()
    model.load_state_dict(torch.load(f"./artifacts/best_metric_model_{best_model_id}.pth", weights_only=True))
    with torch.no_grad():
        test_true = []
        test_pred = []
        test_prob = []
        test_epoch_loss = 0
        for test_data in test_loader:
            
            if config.dataloader == "monai":
                    test_labels = test_data["target"][:,0,0,0,0].to(torch.long).view(-1, 1).to(config.device)
                    del test_data["target"]
                    test_images = torch.concat([tensor for tensor in test_data.values() if isinstance(tensor, torch.Tensor)], dim=1).to(config.device)

            elif config.dataloader == "torch":
                test_images = test_data[0].to(torch.float32).to(config.device)                  
                test_labels = test_data[1].to(torch.long).to(config.device)    

            with torch.amp.autocast(device_type=config.device, dtype=torch.float16): 

                # test_outputs = model(test_images)
                if config.model_name == "ViT":
                    if config.model_depth == "monai":
                        test_outputs = model(test_images)[0]           
                    else: 
                        test_outputs = model(test_images)                   
                else:
                    test_outputs = model(test_images) 

                if config.task == "classification":
                    test_labels_ohe = test_labels.cpu().numpy().reshape(-1, 1)
                    test_labels_ohe = ohe.transform(test_labels_ohe)
                    test_labels_ohe = torch.tensor(test_labels_ohe).to(config.device)
                    test_loss = loss_function(test_outputs, test_labels_ohe)
                elif config.task == "regression":
                    test_loss = loss_function(test_outputs.flatten(), test_labels)
            
            test_epoch_loss += test_loss.item()
            test_true.extend(test_labels.cpu().numpy())

            match config.task:
                case "classification":
                    test_prob.extend(test_outputs.softmax(dim=1)[:,1].detach().cpu().tolist())
                    test_pred.extend(test_outputs.argmax(dim=1).cpu().tolist())    
                case "regression":
                    test_pred.extend(test_outputs.flatten().cpu().tolist())            
        
    test_loss = test_epoch_loss/len(test_loader)
    match config.task:
        case "classification":
            test_bacc = balanced_accuracy_score(test_true, test_pred)
            # test_auc = roc_auc_score(test_true, test_prob)######################
            test_mcc = matthews_corrcoef(test_true, test_pred)
            test_f1 = f1_score(test_true, test_pred, average="micro")
        case "regression":
            test_r2 = r2_score(test_true, test_pred)
            test_rmse = root_mean_squared_error(test_true, test_pred)*100
            # test_rmse = root_mean_squared_error(test_true, test_pred)
            # train_rmse = train_dataset.scaler.inverse_transform(np.array(test_rmse).reshape(-1, 1)).flatten().item()

    mlflow.log_metric("test_loss", test_loss)

    match config.task:
        case "classification":
            mlflow.log_metric("test_bacc", test_bacc)
            # mlflow.log_metric("test_auroc", test_auc)###################
            mlflow.log_metric("test_mcc", test_mcc)
            mlflow.log_metric("test_f1", test_f1)

            # Test Confusion Matrix
            cm = confusion_matrix(test_true, test_pred)                
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot()
            plt.savefig('./artifacts/test_confusion_matrix.png')      
            mlflow.log_artifact('./artifacts/test_confusion_matrix.png')
            os.remove('./artifacts/test_confusion_matrix.png')
            plt.close()   

        case "regression":
            mlflow.log_metric("test_r2", test_r2)      
            mlflow.log_metric("test_rmse", test_rmse)    
            
            postprocess = Postprocessor(true_labels=test_true, pred_labels=test_pred)
            test_mcc, test_bacc = postprocess()
            mlflow.log_metric("test_mcc", test_mcc)      
            mlflow.log_metric("test_bacc", test_bacc)
    
    mlflow.log_artifact(f"./artifacts/best_metric_model_{best_model_id}.pth")
    os.remove(f"./artifacts/best_metric_model_{best_model_id}.pth")

if __name__ == "__main__":

    for model_name in ["ViT"]:
        for model_depth in ["tiny"]:
            for pretrained in [False]:     
                for batch_size in [2]:                     
                    for examination in ["prepost"]:
                        for sequence in ["T1T2"]:#, "T2", "T1T2"]:                             

                            print(f"\nBegin Training - Sequence:{sequence} | Examination: {examination}\n")

                            config = ParamConfigurator()
                            config.model_name = model_name
                            config.model_depth = model_depth
                            config.pretrained = pretrained
                            config.sequence = sequence
                            config.examination = examination 


                            config.batch_size = batch_size                                                            
                            # config.accumulation_steps = accumulation_steps
                                                                    

                            match config.sequence:
                                case "T1T2":
                                    config.channels = 2
                                case _:
                                    config.channels = 1

                            match config.examination:
                                case "prepost":
                                    config.channels = config.channels * 2
                                case _:
                                    config.channels = config.channels * 1                     

                            # preprocess = Preprocessor(config=config)
                            # preprocess()

                            if os.path.exists("/dss/dsshome1/0E/ge37bud3/mlruns"):
                                mlflow.set_tracking_uri("file:///dss/dsshome1/0E/ge37bud3/mlruns")

                            # if config.model_name == "ResNet":
                            mlflow.set_experiment(f'3D{config.model_name}{config.model_depth}_{config.task}')
                            # else:
                            #     mlflow.set_experiment(f'3D{config.model_name}_{config.task}')
                            date = str(datetime.now().strftime('%Y-%m-%d_%H:%M:%S'))
                            with mlflow.start_run(run_name=date, log_system_metrics=False):
                                main(config)    

                            print(f"\nEnd Training: - Sequence:{sequence} | Examination: {examination}\n")               
