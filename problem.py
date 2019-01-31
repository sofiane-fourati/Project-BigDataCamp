from __future__ import division, print_function
import os
import datetime

import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss, recall_score, precision_score
from sklearn.model_selection import StratifiedKFold

import rampwf as rw
from rampwf.score_types.base import BaseScoreType
from rampwf.score_types.classifier_base import ClassifierBaseScoreType
from rampwf.workflows import FeatureExtractorClassifier
from rampwf.workflows.classifier import Classifier


problem_title = 'AssurancExpertsInc'


# -----------------------------------------------------------------------------
# Worklow element
# -----------------------------------------------------------------------------



workflow = FeatureExtractorClassifier()


# -----------------------------------------------------------------------------
# Predictions type
# -----------------------------------------------------------------------------


Predictions = rw.prediction_types.make_multiclass(
    label_names=[0, 1])




# -----------------------------------------------------------------------------
# Score types
# -----------------------------------------------------------------------------

class LogLoss(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = np.inf

    def __init__(self, name='ll', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        score = log_loss(y_true, y_pred)
        return score





class Prop_Lost_Cust(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='Prop_Lost_Cust', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        y_pred = np.round(y_pred)
        return 1 - recall_score(y_true[:,1], y_pred[:,1])
		
class Precision(BaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='Precision', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        y_pred = np.round(y_pred)

        return precision_score(y_true[:,1], y_pred[:,1])
		
		




class Mixed(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = np.inf


    def __init__(self, name='mixed', precision=2):
        self.name = name
        self.precision = precision
        self.prec = Precision()
        self.prop_lost_cust = Prop_Lost_Cust()
        self.log_loss = LogLoss()

    def __call__(self, y_true, y_pred):
        alpha = 3
        beta = 1
        plc = self.prop_lost_cust(y_true, y_pred)
        prec = self.prec(y_true, y_pred)
        ll = self.log_loss(y_true, y_pred)
        return ll + alpha * (plc) + beta * (1 - prec)


score_types = [
    # mixed log-loss /Prop_Lost_Cust / Precision
    Mixed(),
    LogLoss(),
    Precision(),
    Prop_Lost_Cust()
]

# -----------------------------------------------------------------------------
# Cross-validation scheme
# -----------------------------------------------------------------------------


def get_cv(X, y):

    cv = StratifiedKFold(n_splits=5,shuffle=True, random_state = 5)

    for train, test in cv.split(X, y):
        yield (np.hstack(train), np.hstack(test))


# -----------------------------------------------------------------------------
# Training / testing data reader
# -----------------------------------------------------------------------------


def _read_data(path, type):
    data = pd.read_table("data\\AssurancExpertsInc.txt", sep="\t")
    data = data[data['STATUS']==type]
    del data['STATUS']
    data.CLASS = data.CLASS == 'Yes'
    y = data['CLASS']
    del data['CLASS']
    for i in range(43, len(data.columns)):
        prepare = {data.columns[i]: {1 : 25, 2 : 75, 3 : 150 , 4 : 350 , 5 : 750 , 6 : 3000 , 7 : 7500 , 8 : 15000 , 9 : 30000 }}
        data.replace(prepare, inplace=True)
    for i in range(5, len(data.columns)):
        prepare = {data.columns[i]: {1 : 5, 2 : 17, 3 : 30 , 4 : 43   }}
        data.replace(prepare, inplace=True)
    for i in range(5, len(data.columns)):
        prepare = {data.columns[i]: { 5 : 56,6 : 69 , 7 : 82 , 8 : 94 , 9 : 100 }}
        data.replace(prepare, inplace=True)
    prepare = {data.columns[3]: {1 : 25, 2 : 35, 3 : 45 , 4 : 55 , 5 : 65 , 6 : 75  }}
    data.replace(prepare, inplace=True)
    
    # easier but slow method
    # y = pd.Series(0, index=data.index)
    # for begin, end in labels[['begin', 'end']].itertuples(index=False):
    #     y.loc[begin:end] = 1

    # for the "quick-test" mode, use less data
    test = os.getenv('RAMP_TEST_MODE', 0)
    if test:
        N_small = 50000
        data = data[:N_small]
        y = y[:N_small]

    return data, y


def get_train_data(path='.'):
    return _read_data(path, 'Learning')


def get_test_data(path='.'):
    return _read_data(path, 'Test')
