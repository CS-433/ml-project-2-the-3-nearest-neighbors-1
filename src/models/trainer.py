import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score


class Trainer(object):


    def __init__(self, model, lr, epochs, batch_size):

        self.lr = lr
        self.epochs = epochs
        self.model = model
        self.batch_size = batch_size

        # self.criterion = nn.CrossEntropyLoss()
        self.criterion = nn.BCEWithLogitsLoss()  

        self.optimizer = torch.optim.Adam(model.parameters(),lr=lr)  ### WRITE YOUR CODE HERE

    def train_all(self, dataloader):

        loss = []
        for ep in range(self.epochs):
            print("\rEpoch : ",ep+1,"/",self.epochs,end='')
            new_loss = self.train_one_epoch(dataloader)
            loss += new_loss
        print()
        return loss

    def train_one_epoch(self, dataloader):

        self.model.train()
        new_loss = []
        for it, batch in enumerate(dataloader):
            x,y = batch
            logits = self.model.forward(x)
            print(torch.min(logits), torch.max(logits))
            loss = self.criterion(logits,y)
            new_loss += [loss.detach().numpy()]
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            print(loss.detach().numpy())
        return new_loss

    def predict_torch(self, dataloader):

        self.model.eval()
        pred = torch.Tensor()
        with torch.no_grad():
            for it,batch in enumerate(dataloader):
                x = batch[0]
                logits = self.model.forward(x)
                pred = torch.cat((pred,logits),0)
        return pred
    
    def fit(self, training_data, training_labels):

        # First, prepare data for pytorch
        train_dataset = TensorDataset(torch.from_numpy(training_data).float(), 
                                      torch.from_numpy(training_labels).float())
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        loss = self.train_all(train_dataloader)

        return self.predict(training_data)

    def predict(self, test_data):

        test_dataset = TensorDataset(torch.from_numpy(test_data).float())
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        preds = self.predict_torch(test_dataloader).numpy()
        return preds
    
    def score(self, y_true, y_pred, threshold=.5):
        y_pred = y_pred > threshold
        y_pred = y_pred.reshape((-1))
        y_true = y_true.reshape((-1))

        metrics = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1-Score": f1_score(y_true, y_pred, zero_division=0)
        }
        return metrics