import os
import gc
import mlflow
import uuid
from datetime import datetime
from channel_exchange.models.resnet import generate_model
from resnetce_tencent import generate_model as generate_model_tencent
from dataset import CENDataset
from torch.utils.data import DataLoader
import torch
from utils import WeightedCombinedLosses
from utils import SoftF1LossMulti, SoftMCCWithLogitsLoss
from sklearn.metrics import roc_auc_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import balanced_accuracy_score
from param_configurator_ce import ParamConfigurator
from utils import save_conda_env, save_python_files
from torch.utils.data import WeightedRandomSampler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from monai.optimizers import Novograd

def main(config):

    torch.cuda.empty_cache()
    gc.collect()
    save_conda_env(config)
    save_python_files(config) 
    mlflow.log_params(config.__dict__)   

    train_dataset = CENDataset(config=config, split="train")
    val_dataset = CENDataset(config=config, split="val")
    test_dataset = CENDataset(config=config, split="test")

    sampler = WeightedRandomSampler(weights=train_dataset.sample_weights, 
                                    num_samples=len(train_dataset.sample_weights),
                                    replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, sampler=sampler, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    
    if (config.sequence == "T1T2") & (config.examination == "prepost"):
        config.channels = 2
    else:
        config.channels = 1   

    # model = generate_model(model_depth=config.model_depth, n_input_channels=config.channels, n_classes=2).cuda()
    pretrain_path = None
    if config.pretrained:
        pretrain_path = "./data/sarcoma/jan/pretrain/resnet_50_23dataset.pth"
    model = generate_model_tencent(model_depth=50, in_channels=config.channels, num_cls_classes=2, pretrain_path=pretrain_path).to(config.device)                     
                
    match config.optimizer:
        case "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-3)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        case "AdamW":
            optimizer = torch.optim.AdamW(model.parameters())
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1.0)
        case "Novograd":
            optimizer = Novograd(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.scheduler_step, gamma=config.scheduler_gamma)

    weights = None
    imbalance_loss = {"MCC": SoftMCCWithLogitsLoss(), "F1": SoftF1LossMulti(num_classes=2)}
    if config.imbalance == "weight":
        weights = train_dataset.class_weights
    loss_function = WeightedCombinedLosses([torch.nn.CrossEntropyLoss(), imbalance_loss[config.imbalance_loss]], weights=weights)    

    # Gradient accumulation and AMP
    accumulation_steps = config.accumulation_steps
    scaler = torch.amp.GradScaler('cuda')

    # Training loop
    num_epochs = config.epochs
    best_metric = -1

    best_model_id = uuid.uuid4().hex
    for epoch in range(num_epochs):
        epoch = epoch + 1
        print(f"Epoch {epoch}/{num_epochs}")
        model.train()
        train_true = []
        train_pred = []
        train_prob = []
        train_epoch_loss = 0
        lr = scheduler.get_last_lr()[0]
        for batch_data in train_loader:    

            inputs = [batch_data[0].to(config.device), batch_data[1].to(config.device)]
            labels = batch_data[2].to(config.device)

            labels_ohe = labels.cpu().numpy().reshape(-1, 1)
            labels_ohe = train_dataset.ohe.transform(labels_ohe)
            labels_ohe = torch.tensor(labels_ohe).to(config.device)            

            with torch.amp.autocast(device_type=config.device, dtype=torch.float16):                
                outputs = model(inputs)          

                loss = 0
                for output in outputs:
                    loss += loss_function(output, labels_ohe)
            
            scaler.scale(loss).backward()            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            train_epoch_loss += loss.item()
            train_true.extend(labels.cpu().tolist())
            # train_prob.extend(torch.vstack(outputs).sum(dim=0).view(1, -1).softmax(dim=1)[:,1].detach().cpu().tolist())
            train_prob.extend(torch.sum(torch.stack(outputs), dim=0).softmax(dim=1)[:,1].detach().cpu().tolist())
            train_pred.extend(torch.sum(torch.stack(outputs), dim=0).argmax(dim=1).cpu().tolist())
        
        train_loss = train_epoch_loss/len(train_loader)        
        train_bacc = balanced_accuracy_score(train_true, train_pred)
        train_auc = roc_auc_score(train_true, train_prob)
        train_mcc = matthews_corrcoef(train_true, train_pred)

        scheduler.step()
        mlflow.log_metric("learning_rate", lr, step=epoch)            
        
         # Validation
        model.eval()
        with torch.no_grad():
            val_true = []
            val_pred = []
            val_prob = []
            val_epoch_loss = 0
            for val_data in val_loader:
                val_inputs = [val_data[0].to(config.device), val_data[1].to(config.device)]
                val_labels = val_data[2].to(config.device)      

                val_labels_ohe = val_labels.cpu().numpy().reshape(-1, 1)
                val_labels_ohe = train_dataset.ohe.transform(val_labels_ohe)
                val_labels_ohe = torch.tensor(val_labels_ohe).to(config.device)        

                with torch.amp.autocast(device_type=config.device, dtype=torch.float16):
                    val_outputs = model(val_inputs)

                    val_loss = 0
                    for val_output in val_outputs:
                        val_loss += loss_function(val_output, val_labels_ohe)                
                
                val_epoch_loss += val_loss.item()
                val_true.extend(val_labels.cpu().tolist())                
                val_prob.extend(torch.sum(torch.stack(val_outputs), dim=0).softmax(dim=1)[:,1].detach().cpu().tolist())
                val_pred.extend(torch.sum(torch.stack(val_outputs), dim=0).argmax(dim=1).cpu().tolist())                                           
            
        val_loss = val_epoch_loss/len(val_loader)            
        val_bacc = balanced_accuracy_score(val_true, val_pred)
        val_auc = roc_auc_score(val_true, val_prob)         
        val_mcc = matthews_corrcoef(val_true, val_pred)                  

        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("train_bacc", train_bacc, step=epoch)
        mlflow.log_metric("val_bacc", val_bacc, step=epoch)
        mlflow.log_metric("train_auroc", train_auc, step=epoch)
        mlflow.log_metric("val_auroc", val_auc, step=epoch)
        mlflow.log_metric("train_mcc", train_mcc, step=epoch)
        mlflow.log_metric("val_mcc", val_mcc, step=epoch)

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

        if val_mcc > best_metric:
            best_metric = val_mcc
            torch.save(model.state_dict(), f"./artifacts/best_metric_model_{best_model_id}.pth")
            print("[INFO] Saved new best metric model")

    # Test
    model.eval()
    model.load_state_dict(torch.load(f"./artifacts/best_metric_model_{best_model_id}.pth", weights_only=True))
    with torch.no_grad():
        test_true = []
        test_pred = []
        test_prob = []
        test_epoch_loss = 0
        for test_data in test_loader:
            test_inputs = [test_data[0].to(config.device), test_data[1].to(config.device)]
            test_labels = test_data[2].to(config.device)

            test_labels_ohe = test_labels.cpu().numpy().reshape(-1, 1)
            test_labels_ohe = train_dataset.ohe.transform(test_labels_ohe)
            test_labels_ohe = torch.tensor(test_labels_ohe).to(config.device)

            with torch.amp.autocast(device_type=config.device, dtype=torch.float16):
                test_outputs = model(test_inputs)
                
                test_loss = 0
                for test_output in test_outputs:
                    test_loss += loss_function(test_output, test_labels_ohe)                
            
            test_epoch_loss += test_loss.item()
            test_true.extend(test_labels.cpu().tolist())                
            test_prob.extend(torch.sum(torch.stack(test_outputs), dim=0).softmax(dim=1)[:,1].detach().cpu().tolist())
            test_pred.extend(torch.sum(torch.stack(test_outputs), dim=0).argmax(dim=1).cpu().tolist())            
        
    test_loss = test_epoch_loss/len(test_loader)    
    test_bacc = balanced_accuracy_score(test_true, test_pred)
    test_auc = roc_auc_score(test_true, test_prob)
    test_mcc = matthews_corrcoef(test_true, test_pred)      

    mlflow.log_metric("test_loss", test_loss) 
    mlflow.log_metric("test_bacc", test_bacc)
    mlflow.log_metric("test_auroc", test_auc)
    mlflow.log_metric("test_mcc", test_mcc)

    cm = confusion_matrix(test_true, test_pred)                
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig('./artifacts/test_confusion_matrix.png')      
    mlflow.log_artifact('./artifacts/test_confusion_matrix.png')
    os.remove('./artifacts/test_confusion_matrix.png')
    plt.close()          
    
    os.remove(f"./artifacts/best_metric_model_{best_model_id}.pth")

if __name__ == "__main__":  

    for model_depth in [50]:
        for pretrained in [True, False]:
            for batch_size in [2, 4, 8]:

                sequences = ["T1T2", "T1T2", "T1", "T2", "T1T2"]
                examinations = ["pre", "post", "prepost", "prepost", "prepost"]

                for i in range(5):
                    config = ParamConfigurator()
                    config.model_depth = model_depth
                    config.pretrained = pretrained
                    config.batch_size = batch_size
                    config.sequence = sequences[i]
                    config.examination = examinations[i]             
                
                    if os.path.exists("/dss/dsshome1/0E/ge37bud3/mlruns"):
                        mlflow.set_tracking_uri("file:///dss/dsshome1/0E/ge37bud3/mlruns")
            
                    mlflow.set_experiment(f'3DResNet{str(model_depth)}_channelexchange')
                    date = str(datetime.now().strftime('%Y-%m-%d_%H:%M:%S'))
                    with mlflow.start_run(run_name=date, log_system_metrics=False):
                        main(config) 

