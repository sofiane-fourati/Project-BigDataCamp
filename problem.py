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
from rampwf.workflows.feature_extractor import FeatureExtractor
from rampwf.workflows.classifier import Classifier


problem_title = 'AssurancExpertsInc'


# -----------------------------------------------------------------------------
# Worklow element
# -----------------------------------------------------------------------------


class FeatureExtractorClassifier(object):
    """
    Difference with the FeatureExtractorClassifier from ramp-workflow:
    `test_submission` wraps the y_proba in a DataFrame with the original
    index.
    """

    def __init__(self):
        self.element_names = ['feature_extractor', 'classifier']
        self.feature_extractor_workflow = FeatureExtractor(
            [self.element_names[0]])
        self.classifier_workflow = Classifier([self.element_names[1]])

    def train_submission(self, module_path, X_df, y_array, train_is=None):
        if train_is is None:
            train_is = slice(None, None, None)
        fe = self.feature_extractor_workflow.train_submission(
            module_path, X_df, y_array, train_is)
        X_train_array = self.feature_extractor_workflow.test_submission(
            fe, X_df.iloc[train_is])
        clf = self.classifier_workflow.train_submission(
            module_path, X_train_array, y_array[train_is])
        return fe, clf

    def test_submission(self, trained_model, X_df):
        fe, clf = trained_model
        X_test_array = self.feature_extractor_workflow.test_submission(
            fe, X_df)
        y_proba = self.classifier_workflow.test_submission(clf, X_test_array)

        arr = X_df.index.values.astype('datetime64[m]').astype(int)
        y = np.hstack((arr[:, np.newaxis], y_proba))
        return y


workflow = FeatureExtractorClassifier()


# -----------------------------------------------------------------------------
# Predictions type
# -----------------------------------------------------------------------------


BaseMultiClassPredictions = rw.prediction_types.make_multiclass(
    label_names=[0, 1])


class Predictions(BaseMultiClassPredictions):
    """
    Overriding parts of the ramp-workflow version to preserve the y_pred /
    y_true DataFrames.
    """

    n_columns = 3

    def __init__(self, y_pred=None, y_true=None, n_samples=None):
        # override init to not convert y_pred/y_true to arrays
        if y_pred is not None:
            self.y_pred = np.array(y_pred)
        elif y_true is not None:
            self._init_from_pred_labels(y_true)
            arr = y_true.index.values.astype('datetime64[m]').astype(int)
            self.y_pred = np.hstack((arr[:, np.newaxis], self.y_pred))
        elif n_samples is not None:
            self.y_pred = np.empty((n_samples, self.n_columns), dtype=float)
            self.y_pred.fill(np.nan)
        else:
            raise ValueError(
                'Missing init argument: y_pred, y_true, or n_samples')
        self.check_y_pred_dimensions()

    @property
    def y_pred_label_index(self):
        """Multi-class y_pred is the index of the predicted label."""
        return np.argmax(self.y_pred[:, 1:], axis=1)

    @classmethod
    def combine(cls, predictions_list, index_list=None):
        if index_list is None:  # we combine the full list
            index_list = range(len(predictions_list))
        y_comb_list = np.array(
            [predictions_list[i].y_pred for i in index_list])
        # clipping probas into [0, 1], also taking care of the case of all
        # zeros
        y_comb_list[:, :, 1:] = np.clip(
            y_comb_list[:, :, 1:], 10 ** -15, 1 - 10 ** -15)
        # normalizing probabilities
        y_comb_list[:, :, 1:] = y_comb_list[:, :, 1:] / np.sum(
            y_comb_list[:, :, 1:], axis=2, keepdims=True)
        # I expect to see RuntimeWarnings in this block
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            y_comb = np.nanmean(y_comb_list, axis=0)
        combined_predictions = cls(y_pred=y_comb)
        return combined_predictions


# -----------------------------------------------------------------------------
# Score types
# -----------------------------------------------------------------------------

class PointwiseLogLoss(BaseScoreType):
    # subclass BaseScoreType to use raw y_pred (proba's)
    is_lower_the_better = True
    minimum = 0.0
    maximum = np.inf

    def __init__(self, name='pw_ll', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        score = log_loss(y_true[:, 1:], y_pred[:, 1:])
        return score





class Prop_Lost_Cust(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='Prop_Lost_Cust', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        y_true = np.round(y_true[:, 2])
        y_pred = np.round(y_pred[:, 2])
        return 1 - recall_score(y_true, y_pred)
		
class Precision(BaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='Precision', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        y_true = np.round(y_true[:, 2])
        y_pred = np.round(y_pred[:, 2])

        return precision_score(y_true, y_pred)
		
		




class Mixed(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = np.inf


    def __init__(self, name='mixed', precision=2):
        self.name = name
        self.precision = precision
        self.prec = Precision()
        self.prop_lost_cust = Prop_Lost_Cust()
        self.pointwise_log_loss = PointwiseLogLoss()

    def __call__(self, y_true, y_pred):
        alpha = 8
        beta = 3
        plc = self.prop_lost_cust(y_true, y_pred)
        prec = self.prec(y_true, y_pred)
        ll = self.pointwise_log_loss(y_true, y_pred)
        return ll + alpha * (plc) + beta * (1 - prec)


score_types = [
    # mixed log-loss /Prop_Lost_Cust / Precision
    Mixed(),
    PointwiseLogLoss(),
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
