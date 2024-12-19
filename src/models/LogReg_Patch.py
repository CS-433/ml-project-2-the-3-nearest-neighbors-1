from sklearn import linear_model
import numpy as np

class LogReg_patch():
    
    def __init__(self):
        self.logreg = linear_model.LogisticRegression(C=1e5, class_weight="balanced")

    def features_from_input(self, X):
        feat_m = np.mean(X, axis=(1,2,3))
        feat_v = np.var(X, axis=(1,2,3))
        feat = np.c_[feat_m, feat_v]
        return feat

    def fit(self, X, y):
        feat = self.features_from_input(X)
        self.logreg.fit(feat, y.flatten())

    def predict(self, X):
        feat = self.features_from_input(X)
        return self.logreg.predict(feat)