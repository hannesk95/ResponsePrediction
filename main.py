import os
import gc
import torch
import mlflow
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from datetime import datetime
from utils import save_conda_env
from utils import create_confusion_matrix
from dataset import ResNetDataset
from dataset import CombinedDataset
from preprocessor import Preprocessor
from postprocessor import Postprocessor
from monai.networks.nets import resnet
from monai.optimizers import Novograd
from torch.utils.data import DataLoader
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
from resnet_tencent import generate_model


def main(config) -> None:

    torch.cuda.empty_cache()
    gc.collect()
    save_conda_env(config)
    # save_python_files(config) # TODO: save python files tracked by git in artifacts
    mlflow.log_params(config.__dict__)    


    match config.dataset:
        case "sarcoma":
            # train_dataset = ResNetDataset(config=config, split="train")
            # val_dataset = ResNetDataset(config=config, split="val")
            # test_dataset = ResNetDataset(config=config, split="test")  
            train_dataset = CombinedDataset(config=config, split="train")
            val_dataset = CombinedDataset(config=config, split="val")
            test_dataset = CombinedDataset(config=config, split="test")  
        
        case "glioblastoma":
            raise ValueError("Not yet implemented!") # TODO

    sampler = WeightedRandomSampler(weights=train_dataset.sample_weights, 
                                    num_samples=len(train_dataset.sample_weights),
                                    replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, sampler=sampler, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    num_channels = 0
    match config.sequence:
        case "T1T2":
            num_channels += 2
        case _:
            num_channels += 1
    
    match config.examination:
        case "prepost":
            num_channels = num_channels * 2
        case _:
            num_channels = num_channels * 1
    
    config.channels = num_channels

    match config.task:
        case "classification":
            weights = None
            output_neurons = 2
            imbalance_loss = {"MCC": SoftMCCWithLogitsLoss(), "F1": SoftF1LossMulti(num_classes=output_neurons)}
            if config.imbalance == "weight":
                weights = train_dataset.class_weights                
            
            loss_function = WeightedCombinedLosses([torch.nn.CrossEntropyLoss(), imbalance_loss[config.imbalance_loss]], weights=weights)
            
        case "regression":
            loss_function = torch.nn.MSELoss()    
            output_neurons = 1

    match config.model_name:
        case "ResNet":
            match config.model_depth:
                case 10:
                    pretrain_path = None
                    if config.pretrained:
                        pretrain_path = "/home/johannes/Code/ResponsePrediction/data/sarcoma/jan/pretrain/resnet_10_23dataset.pth"
                    model = generate_model(model_depth=10, in_channels=num_channels, num_cls_classes=output_neurons, pretrain_path=pretrain_path)#.to(config.device)                     
                case 18:
                    pretrain_path = None
                    if config.pretrained:
                        pretrain_path = "/home/johannes/Code/ResponsePrediction/data/sarcoma/jan/pretrain/resnet_18_23dataset.pth"
                    # model = resnet.resnet18(spatial_dims=3, n_input_channels=num_channels, num_classes=output_neurons).to(config.device)   
                    model = generate_model(model_depth=18, in_channels=num_channels, num_cls_classes=output_neurons, pretrain_path=pretrain_path).to(config.device)
                case 34:
                    pretrain_path = None
                    if config.pretrained:
                        pretrain_path = "/home/johannes/Code/ResponsePrediction/data/sarcoma/jan/pretrain/resnet_34_23dataset.pth"
                    # model = resnet.resnet34(spatial_dims=3, n_input_channels=num_channels, num_classes=output_neurons).to(config.device)
                    model = generate_model(model_depth=34, in_channels=num_channels, num_cls_classes=output_neurons, pretrain_path=pretrain_path).to(config.device) 
                case 50:
                    pretrain_path = None
                    if config.pretrained:
                        pretrain_path = "/home/johannes/Code/ResponsePrediction/data/sarcoma/jan/pretrain/resnet_50_23dataset.pth"
                    # model = resnet.resnet50(spatial_dims=3, n_input_channels=num_channels, num_classes=output_neurons).to(config.device)  
                    model = generate_model(model_depth=50, in_channels=num_channels, num_cls_classes=output_neurons, pretrain_path=pretrain_path).to(config.device)
                case 101:
                    pretrain_path = None
                    if config.pretrained:
                        pretrain_path = "/home/johannes/Code/ResponsePrediction/data/sarcoma/jan/pretrain/pretrain/resnet_101.pth"
                    # model = resnet.resnet101(spatial_dims=3, n_input_channels=num_channels, num_classes=output_neurons).to(config.device)  
                    model = generate_model(model_depth=101, in_channels=num_channels, num_cls_classes=output_neurons, pretrain_path=pretrain_path).to(config.device)
                case 152:
                    pretrain_path = None
                    if config.pretrained:
                        pretrain_path = "/home/johannes/Code/ResponsePrediction/data/sarcoma/jan/pretrain/pretrain/resnet_152.pth"
                    # model = resnet.resnet152(spatial_dims=3, n_input_channels=num_channels, num_classes=output_neurons).to(config.device)    
                    model = generate_model(model_depth=152, in_channels=num_channels, num_cls_classes=output_neurons, pretrain_path=pretrain_path).to(config.device)
        case _:
            raise ValueError(f"Given model name '{config.model_name}' is not implemented!")  
    
    if config.task == "regression":
        model.fc = torch.nn.Sequential(torch.nn.Linear(model.fc.in_features, out_features=output_neurons, bias=True),
                                       torch.nn.Sigmoid()).to(config.device)

    match config.optimizer:
        case "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        case "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        case "Novograd":
            optimizer = Novograd(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.scheduler_step, gamma=config.scheduler_gamma)

    # Gradient accumulation and AMP
    accumulation_steps = config.accumulation_steps
    scaler = torch.amp.GradScaler('cuda')

    # Training loop
    num_epochs = config.epochs
    best_metric = -1
    for epoch in range(num_epochs):
        epoch = epoch + 1
        print(f"Epoch {epoch}/{num_epochs}")
        model.train()
        train_true = []
        train_pred = []
        train_prob = []
        train_epoch_loss = 0
        lr = scheduler.get_last_lr()[0]
        for batch_idx, batch_data in enumerate(train_loader):
            inputs = batch_data[0].to(config.device)          
            labels = batch_data[1].to(config.device)

            with torch.amp.autocast(device_type=config.device, dtype=torch.float16):                
                outputs = model(inputs)
                if config.task == "classification":
                    labels_ohe = labels.cpu().numpy().reshape(-1, 1)
                    labels_ohe = train_dataset.ohe.transform(labels_ohe)
                    labels_ohe = torch.tensor(labels_ohe).to(config.device)
                    loss = loss_function(outputs, labels_ohe)
                elif config.task == "regression":
                    loss = loss_function(outputs.flatten(), labels)
                loss = loss / accumulation_steps
            scaler.scale(loss).backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_epoch_loss += loss.item() * accumulation_steps
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
                train_auc = roc_auc_score(train_true, train_prob)
                # train_auc = 0
                train_mcc = matthews_corrcoef(train_true, train_pred)
                train_f1 = f1_score(train_true, train_pred)
            case "regression":
                train_r2 = r2_score(train_true, train_pred)
                train_rmse = root_mean_squared_error(train_true, train_pred)
                train_rmse = train_dataset.scaler.inverse_transform(np.array(train_rmse).reshape(-1, 1)).flatten().item()

        # Validation
        model.eval()
        with torch.no_grad():
            val_true = []
            val_pred = []
            val_prob = []
            val_epoch_loss = 0
            for val_data in val_loader:
                val_images = val_data[0].to(config.device)
                val_labels = val_data[1].to(config.device)               

                val_outputs = model(val_images)                

                if config.task == "classification":
                    val_labels_ohe = val_labels.cpu().numpy().reshape(-1, 1)
                    val_labels_ohe = train_dataset.ohe.transform(val_labels_ohe)
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
                    val_auc = roc_auc_score(val_true, val_prob)         
                    # val_auc = 0
                    val_mcc = matthews_corrcoef(val_true, val_pred)
                    val_f1 = f1_score(val_true, val_pred)
                case "regression":
                    val_r2 = r2_score(val_true, val_pred)
                    val_rmse = root_mean_squared_error(val_true, val_pred)
                    train_rmse = train_dataset.scaler.inverse_transform(np.array(val_rmse).reshape(-1, 1)).flatten().item()

        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)

        match config.task:
            case "classification":
                mlflow.log_metric("train_bacc", train_bacc, step=epoch)
                mlflow.log_metric("val_bacc", val_bacc, step=epoch)
                mlflow.log_metric("train_auroc", train_auc, step=epoch)
                mlflow.log_metric("val_auroc", val_auc, step=epoch)
                mlflow.log_metric("train_mcc", train_mcc, step=epoch)
                mlflow.log_metric("val_mcc", val_mcc, step=epoch)
                mlflow.log_metric("train_f1", train_f1, step=epoch)
                mlflow.log_metric("val_f1", val_f1, step=epoch)

                # img, _, _ = create_confusion_matrix(train_true, train_pred)
                # mlflow.log_image(img, key="train", step=epoch)
                
                cm = confusion_matrix(train_true, train_pred)                
                disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                disp.plot()
                plt.savefig(f'./artifacts/train_confusion_matrix_{str(epoch).zfill(3)}.png')
                mlflow.log_artifact(f'./artifacts/train_confusion_matrix_{str(epoch).zfill(3)}.png')
                os.remove(f'./artifacts/train_confusion_matrix_{str(epoch).zfill(3)}.png') 
                plt.close()               

                cm = confusion_matrix(val_true, val_pred)                
                disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                disp.plot()
                plt.savefig(f'./artifacts/val_confusion_matrix_{str(epoch).zfill(3)}.png')      
                mlflow.log_artifact(f'./artifacts/val_confusion_matrix_{str(epoch).zfill(3)}.png')
                os.remove(f'./artifacts/val_confusion_matrix_{str(epoch).zfill(3)}.png')          
                plt.close()   

                if val_f1 > best_metric:
                    best_metric = val_f1
                    torch.save(model.state_dict(), "./artifacts/best_metric_model.pth")
                    print("[INFO] Saved new best metric model")

            case "regression":
                mlflow.log_metric("train_r2", train_r2, step=epoch)
                mlflow.log_metric("val_r2", val_r2, step=epoch)
                mlflow.log_metric("train_rmse", train_rmse, step=epoch)
                mlflow.log_metric("val_rmse", val_rmse, step=epoch)

                if val_r2 > best_metric:
                    best_metric = val_r2
                    torch.save(model.state_dict(), "./artifacts/best_metric_model.pth")
                    print("[INFO] Saved new best metric model")
        

    # Test
    model.eval()
    model.load_state_dict(torch.load("./artifacts/best_metric_model.pth", weights_only=True))
    with torch.no_grad():
        test_true = []
        test_pred = []
        test_prob = []
        test_epoch_loss = 0
        for test_data in test_loader:
            test_images = test_data[0].to(config.device)
            test_labels = test_data[1].to(config.device)

            test_outputs = model(test_images)
            if config.task == "classification":
                test_labels_ohe = test_labels.cpu().numpy().reshape(-1, 1)
                test_labels_ohe = train_dataset.ohe.transform(test_labels_ohe)
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
            test_auc = roc_auc_score(test_true, test_prob)
            test_mcc = matthews_corrcoef(test_true, test_pred)
            test_f1 = f1_score(test_true, test_pred)
        case "regression":
            test_r2 = r2_score(test_true, test_pred)
            test_rmse = root_mean_squared_error(test_true, test_pred)
            train_rmse = train_dataset.scaler.inverse_transform(np.array(test_rmse).reshape(-1, 1)).flatten().item()

    mlflow.log_metric("test_loss", test_loss)

    match config.task:
        case "classification":
            mlflow.log_metric("test_bacc", test_bacc)
            mlflow.log_metric("test_auroc", test_auc)
            mlflow.log_metric("test_mcc", test_mcc)
            mlflow.log_metric("test_f1", test_f1)

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
    
    os.remove("./artifacts/best_metric_model.pth")

if __name__ == "__main__":

    for model_depth in [10, 18, 34, 50]:
        for pretrained in [False, True]:
            for task in ["classification"]:
                for sequence in ["T1", "T2", "T1T2"]:
                    for examination in ["pre", "post", "prepost"]:                

                        print(f"\nBegin Training: {task} | {sequence} | {examination}\n")

                        config = ParamConfigurator()
                        config.model_depth = model_depth
                        config.pretrained = pretrained
                        config.task = task
                        config.sequence = sequence
                        config.examination = examination

                        # preprocess = Preprocessor(config=config)
                        # preprocess()
                        mlflow.set_experiment(f'3D{config.model_name}{config.model_depth}_{config.task}')
                        date = str(datetime.now().strftime('%Y-%m-%d_%H:%M:%S'))
                        with mlflow.start_run(run_name=date, log_system_metrics=False):
                            main(config)    

                        print(f"\nEnd Training: {task} | {sequence} | {examination}\n")               
