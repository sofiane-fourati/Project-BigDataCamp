from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
import pandas as pd
import numpy as np


class Classifier(BaseEstimator):
    
    def __init__(self):
        self.model = LogisticRegression()
    def fit(self, X, y):
        X = pd.DataFrame(X)
        y.reset_index(drop=True, inplace=True)
        df = pd.concat([X, y], axis = 1)
        df_minority = df[df.CLASS==True].copy()
        df_minority_upsampled = resample(df_minority, 
                                 replace=True,    
                                 n_samples=3000,    
                                 random_state=123)
        df = pd.concat([df, df_minority_upsampled])
        y = df['CLASS'].copy()
        X = df.drop('CLASS',axis = 1).copy()
        self.model.fit(np.array(X), np.array(y))

    def predict_proba(self, X):
        proba = self.model.predict_proba(np.array(X))
        return proba
