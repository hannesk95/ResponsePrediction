from monai.data import DataLoader
from monai.networks.nets import resnet
from monai.optimizers import Novograd
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, matthews_corrcoef
from dataset import ResNetDataset

def main():

    train_dataset = ResNetDataset(data_dir="/home/johannes/Code/ResponsePrediction/data/sarcoma/preprocessed/T1_UWS_pre", split="train")
    val_dataset = ResNetDataset(data_dir="/home/johannes/Code/ResponsePrediction/data/sarcoma/preprocessed/T1_UWS_pre", split="val")
    test_dataset = ResNetDataset(data_dir="/home/johannes/Code/ResponsePrediction/data/sarcoma/preprocessed/T1_TUM_pre", split="test")

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Initialize ResNet50 model, loss function, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet.resnet50(spatial_dims=3, n_input_channels=1, num_classes=2).to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = Novograd(model.parameters(), lr=1e-4)

    # Gradient accumulation and AMP
    accumulation_steps = 8
    scaler = torch.amp.GradScaler('cuda')

    # Training loop
    num_epochs = 100
    best_metric = -1
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        model.train()
        train_true = []
        train_pred = []
        train_prob = []
        train_epoch_loss = 0
        for batch_idx, batch_data in enumerate(train_loader):
            inputs = batch_data[0].to(device)
            labels = batch_data[1].to(device)
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):                
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                loss = loss / accumulation_steps
            scaler.scale(loss).backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_epoch_loss += loss.item() * accumulation_steps
            train_true.append(labels.cpu().numpy())
            train_prob.append(outputs.softmax(dim=1)[:,1].detach().cpu().numpy())
            train_pred.append(outputs.argmax(dim=1).cpu().numpy())
        
        train_loss = train_epoch_loss/len(train_loader)
        train_bacc = balanced_accuracy_score(train_true, train_pred)
        train_auc = roc_auc_score(train_true, train_prob, average='weighted')
        train_mcc = matthews_corrcoef(train_true, train_pred)

        # Validation
        model.eval()
        with torch.no_grad():
            val_true = []
            val_pred = []
            val_prob = []
            val_epoch_loss = 0
            for val_data in val_loader:
                val_images = val_data[0].to(device)
                val_labels = val_data[1].to(device)

                val_outputs = model(val_images)
                val_loss = loss_function(val_outputs, val_labels)
                
                val_epoch_loss =+ val_loss.item()
                val_true.append(val_labels.cpu().numpy())
                val_prob.append(val_outputs.softmax(dim=1)[:,1].detach().cpu().numpy())
                val_pred.append(val_outputs.argmax(dim=1).cpu().numpy())                
            
            val_loss = val_epoch_loss/len(val_loader)
            val_bacc = balanced_accuracy_score(val_true, val_pred)
            val_auc = roc_auc_score(val_true, val_prob, average='weighted')         
            val_mcc = matthews_corrcoef(val_true, val_pred)   

        print(f"Training   loss:    {train_loss}")
        print(f"Validation loss:    {val_loss}\n")
        print(f"Training   bACC:    {train_bacc}")
        print(f"Validation bACC:    {val_bacc}\n")
        print(f"Training   AUROC:   {train_auc}")       
        print(f"Validation AUROC:   {val_auc}\n")
        print(f"Training   MCC:     {train_mcc}")       
        print(f"Validation MCC:     {val_mcc}\n")

        if val_auc > best_metric:
            best_metric = val_auc
            torch.save(model.state_dict(), "best_metric_model.pth")
            print("[INFO] Saved new best metric model")

    # Test
    model.eval()
    model.load_state_dict(torch.load("best_metric_model.pth", weights_only=True))
    with torch.no_grad():
        test_true = []
        test_pred = []
        test_prob = []
        test_epoch_loss = 0
        for test_data in test_loader:
            test_images = test_data[0].to(device)
            test_labels = test_data[1].to(device)

            test_outputs = model(test_images)
            test_loss = loss_function(test_outputs, test_labels)
            
            test_epoch_loss =+ test_loss.item()
            test_true.append(test_labels.cpu().numpy())
            test_prob.append(test_outputs.softmax(dim=1)[:,1].detach().cpu().numpy())
            test_pred.append(test_outputs.argmax(dim=1).cpu().numpy())                
        
        test_loss = test_epoch_loss/len(test_loader)
        test_bacc = balanced_accuracy_score(test_true, test_pred)
        test_auc = roc_auc_score(test_true, test_prob, average='weighted')
        test_mcc = matthews_corrcoef(test_true, test_pred)

        print(f"Test loss:    {test_loss}")
        print(f"Test bACC:    {test_bacc}")
        print(f"Test AUROC:   {test_auc}")       
        print(f"Test MCC:     {test_mcc}")       


if __name__ == "__main__":

    for task in ["classification", "regression"]:
        for sequence in ["T1", "T2", "both"]:
            for examination in ["pre", "post", "both"]:
                
                main(task, sequence, examination)
