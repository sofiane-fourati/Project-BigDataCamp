from __future__ import division
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


class Classifier(BaseEstimator):
    def __init__(self):
        self.model = make_pipeline(StandardScaler(), LogisticRegression())

    def fit(self, X, y):
        clf_RF = RandomForestClassifier(n_estimators=50, criterion='gini', max_depth=10,
                                min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                                max_features='auto', max_leaf_nodes=None, 
                                bootstrap=True, oob_score=False, n_jobs=1, 
                                random_state=125, verbose=1, warm_start=False, class_weight=None)
        self.model.fit(X, y)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
