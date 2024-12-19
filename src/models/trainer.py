import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
import joblib
import numpy as np

class Trainer(object):


    def __init__(self, model, lr, weight_decay, epochs, batch_size):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.epochs = epochs
        self.model = model.to(self.device)
        self.batch_size = batch_size

        self.criterion = nn.functional.binary_cross_entropy #pos_weight=weights[1]
        self.optimizer = torch.optim.AdamW(model.parameters(),lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )
        self.history = dict()

    def train_one_epoch(self, dataloader):
        self.model.train()
        losses = []
        accuracies = []
        f1_scores = []
    
        for it, (x, y) in enumerate(dataloader):
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            output = self.model.forward(x)
    
            # Calcul des poids pour les données déséquilibrées
            nb_pixels = y.numel()
            nb_road_pixels = y.sum()
            nb_bg_pixels = nb_pixels - nb_road_pixels
            weights = torch.where(y == 1, nb_pixels / (nb_road_pixels + 1e-6), nb_pixels / (nb_bg_pixels + 1e-6))
            
            # Calcul de la perte
            loss = self.criterion(output, y, weights)  # Weighted loss for unbalanced data
            loss.backward()
            self.optimizer.step()
    
            # Calcul des prédictions
            pred = (output >= 0.5).float()
            
            # Calcul de l'accuracy
            accuracy = pred.eq(y.view_as(pred)).sum().item() / (len(x) * x.shape[2] * x.shape[3])
            accuracies.append(accuracy)
    
            # Calcul du F1-Score pour le batch            
            f1 = f1_score(y.cpu().numpy().flatten(), pred.cpu().numpy().flatten(), zero_division=0)
            f1_scores.append(f1)
    
            # Enregistrer la perte
            losses.append(loss.item())
    
            # Affichage des logs toutes les 10 itérations
            if it % 10 == 0:
                print(
                    f"    Batch number {it+1} : "
                    f"loss = {loss.item():0.2e}, "
                    f"acc = {accuracy:0.3f}, "
                    f"f1 = {f1:0.3f}, "
                    f"lr = {self.scheduler.get_last_lr()[0]:0.3e} "
                )
        
        return losses, accuracies, f1_scores


    def validate(self, val_dataloader):
        self.model.eval()
        loss = 0
        all_preds = []
        all_labels = []
    
        with torch.no_grad():
            for x, y in val_dataloader:
                x, y = x.to(self.device), y.to(self.device)
                output = self.model.forward(x)
                loss += self.criterion(output, y).item() * len(x)
    
                pred = (output >= 0.5).float()
                all_preds.append(pred.cpu())
                all_labels.append(y.cpu())
                    
        loss /= len(val_dataloader.dataset)
    
        all_preds = torch.cat(all_preds).numpy().flatten()
        all_labels = torch.cat(all_labels).numpy().flatten()
    
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
    
        print(
            "Intermediate validation results : ",
            f"Average loss: {loss:.4f}, ",
            f"Accuracy: {100 * accuracy:.2f}%, ",
            f"F1-Score: {f1:.4f}",
            "\n"
        )
        
        return loss, accuracy, f1


    def fit(self, training_data, training_labels, val_data, val_labels):
        # First, prepare data for pytorch
        train_dataset = TensorDataset(
            torch.from_numpy(training_data).float(),
            torch.from_numpy(training_labels).float()
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=torch.cuda.is_available(), # pinned memory when using a GPU for faster transfers
            num_workers=2
        )
        val_dataset = TensorDataset(
            torch.from_numpy(val_data).float(),
            torch.from_numpy(val_labels).float()
        )
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        train_loss_history = []
        train_acc_history = []
        train_f1_history = []
        val_loss_history = []
        val_acc_history = []
        val_f1_history = []
        for ep in range(self.epochs):
            print(f"Epoch : {ep+1}/{self.epochs}")

            train_losses, train_accuracies, train_f1_scores = self.train_one_epoch(train_dataloader)
            train_loss_history.append(np.mean(train_losses))
            train_acc_history.append(np.mean(train_accuracies))
            train_f1_history.append(np.mean(train_f1_scores))
                
            val_loss, val_accuracy, val_f1 = self.validate(val_dataloader)
            val_loss_history.append(val_loss)
            val_acc_history.append(val_accuracy)
            val_f1_history.append(val_f1)

            self.scheduler.step(val_loss)

        self.history = dict(
            train_loss_history=train_loss_history,
            train_acc_history=train_acc_history,
            train_f1_history=train_f1_history,
            val_loss_history=val_loss_history,
            val_acc_history=val_acc_history,
            val_f1_history=val_f1_history
        )
        return self.history
        
    def predict(self, test_data):
        test_dataset = TensorDataset(torch.from_numpy(test_data).float())
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        self.model.eval()
        preds = torch.empty(0, device=self.device)
        with torch.no_grad():
            for x in test_dataloader:
                x = x[0].to(self.device)
                output = self.model.forward(x)
                preds = torch.cat((preds,output),0)
        preds = preds.cpu().numpy()
        return preds


    def score(self, y_true, y_pred):
        y_pred = y_pred.reshape((-1))
        y_true = y_true.reshape((-1))
        metrics = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1-Score": f1_score(y_true, y_pred, zero_division=0)
        }
        return metrics

    