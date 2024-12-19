import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
import joblib
import numpy as np

class Trainer(object):
    """
    A class to train, validate, and test a PyTorch model.
    
    Attributes:
    - model: The PyTorch model to be trained.
    - lr: Learning rate for the optimizer.
    - weight_decay: Weight decay for regularization in the optimizer.
    - epochs: Number of training epochs.
    - batch_size: Size of batches used for training and validation.
    """

    def __init__(self, model, lr, weight_decay, epochs, batch_size):
        """
        Initializes the Trainer class.

        Parameters:
        - model: The PyTorch model to train.
        - lr: Learning rate for the optimizer.
        - weight_decay: Weight decay coefficient for regularization.
        - epochs: Number of epochs to train the model.
        - batch_size: Size of the mini-batches.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available.

        self.epochs = epochs
        self.model = model.to(self.device)  # Move the model to the appropriate device.
        self.batch_size = batch_size

        self.criterion = nn.functional.binary_cross_entropy  # Loss function for binary classification.
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)  # Optimizer.
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3  # Reduce learning rate on validation loss plateau.
        )
        self.history = dict()  # Dictionary to store training and validation metrics.

    def train_one_epoch(self, dataloader):
        """
        Trains the model for one epoch.

        Parameters:
        - dataloader: DataLoader for the training dataset.

        Returns:
        - losses: List of training losses.
        - accuracies: List of accuracies for each batch.
        - f1_scores: List of F1-scores for each batch.
        """
        self.model.train()
        losses = []
        accuracies = []
        f1_scores = []
    
        for it, (x, y) in enumerate(dataloader):
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            output = self.model.forward(x)  # Forward pass.
    
            # Calculate class weights for unbalanced data.
            nb_pixels = y.numel()
            nb_road_pixels = y.sum()
            nb_bg_pixels = nb_pixels - nb_road_pixels
            weights = torch.where(y == 1, nb_pixels / (nb_road_pixels + 1e-6), nb_pixels / (nb_bg_pixels + 1e-6))
            
            loss = self.criterion(output, y, weights)  # Weighted binary cross-entropy loss.
            loss.backward()  # Backward pass.
            self.optimizer.step()  # Update model parameters.
    
            pred = (output >= 0.5).float()  # Binary predictions based on a threshold.
            
            accuracy = pred.eq(y.view_as(pred)).sum().item() / (len(x) * x.shape[2] * x.shape[3])  # Compute accuracy.
            accuracies.append(accuracy)
    
            f1 = f1_score(y.cpu().numpy().flatten(), pred.cpu().numpy().flatten(), zero_division=0)  # F1 score.
            f1_scores.append(f1)
    
            losses.append(loss.item())  # Track loss.
    
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
        """
        Validates the model on the validation dataset.

        Parameters:
        - val_dataloader: DataLoader for the validation dataset.

        Returns:
        - loss: Average validation loss.
        - accuracy: Validation accuracy.
        - f1: Validation F1 score.
        """
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
        """
        Trains the model on the training dataset and validates on the validation dataset.

        Parameters:
        - training_data: Training data.
        - training_labels: Labels for the training data.
        - val_data: Validation data.
        - val_labels: Labels for the validation data.

        Returns:
        - self.history: Dictionary containing training and validation metrics.
        """
        # Prepare training and validation datasets and loaders.
        train_dataset = TensorDataset(
            torch.from_numpy(training_data).float(),
            torch.from_numpy(training_labels).float()
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=torch.cuda.is_available(),  # Use pinned memory for faster GPU data transfer.
            num_workers=2  # Number of subprocesses for data loading.
        )
        val_dataset = TensorDataset(
            torch.from_numpy(val_data).float(),
            torch.from_numpy(val_labels).float()
        )
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # Track metrics across epochs.
        train_loss_history = []
        train_acc_history = []
        train_f1_history = []
        val_loss_history = []
        val_acc_history = []
        val_f1_history = []
        
        for ep in range(self.epochs):
            print(f"Epoch : {ep+1}/{self.epochs}")

            # Train for one epoch.
            train_losses, train_accuracies, train_f1_scores = self.train_one_epoch(train_dataloader)
            train_loss_history.append(np.mean(train_losses))
            train_acc_history.append(np.mean(train_accuracies))
            train_f1_history.append(np.mean(train_f1_scores))
                
            # Validate the model.
            val_loss, val_accuracy, val_f1 = self.validate(val_dataloader)
            val_loss_history.append(val_loss)
            val_acc_history.append(val_accuracy)
            val_f1_history.append(val_f1)

            # Adjust learning rate based on validation loss.
            self.scheduler.step(val_loss)

        # Save training and validation history.
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
        """
        Makes predictions on test data.

        Parameters:
        - test_data: Test dataset.

        Returns:
        - preds: Predictions as a NumPy array.
        """
        test_dataset = TensorDataset(torch.from_numpy(test_data).float())
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        self.model.eval()
        preds = torch.empty(0, device=self.device)
        with torch.no_grad():
            for x in test_dataloader:
                x = x[0].to(self.device)
                output = self.model.forward(x)
                preds = torch.cat((preds, output), 0)
        preds = preds.cpu().numpy()
        return preds

    def score(self, y_true, y_pred):
        """
        Computes evaluation metrics for predictions.

        Parameters:
        - y_true: Ground truth labels.
        - y_pred: Predicted labels.

        Returns:
        - metrics: Dictionary containing accuracy, precision, recall, and F1 score.
        """
        y_pred = y_pred.reshape((-1))
        y_true = y_true.reshape((-1))
        metrics = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1-Score": f1_score(y_true, y_pred, zero_division=0)
        }
        return metrics

    