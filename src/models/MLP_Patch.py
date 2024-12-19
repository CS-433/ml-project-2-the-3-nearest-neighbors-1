from sklearn.neural_network import MLPClassifier
from sklearn.metrics import *
import numpy as np

class MLP_patch:

    def __init__(self):
        self.model = MLPClassifier(hidden_layer_sizes=(100,100,))

    def convert_format(self, X, y):
        return X.reshape(X.shape[0],-1), y.flatten()
        
    def fit(self, X, y):
        X, y = self.convert_format(X, y)
        self.model.fit(X,y)

    def predict(self, X):
        X, _ = self.convert_format(X, np.array([]))
        return self.model.predict(X)
        
def score(y_true, y_pred):
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0)
    }
    return metrics