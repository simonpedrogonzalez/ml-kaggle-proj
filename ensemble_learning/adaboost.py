import sys, os; sys.path.insert(0, os.path.abspath('.')) if os.path.abspath('.') not in sys.path else None
from copy import deepcopy
from utils.preprocessing import CatEncodedDataFrame, CatEncodedSeries
from ensemble_learning.ensemble import Ensemble
import pandas as pd
import numpy as np

class AdaBoost(Ensemble):
    def __init__(self, learner_model, n_learners):
        self.alphas = []
        self.w = None
        super().__init__(learner_model, n_learners)

    def _build_ensemble(self, X: np.ndarray, y: np.ndarray):
        # Init weights
        n = len(X)
        self.w = np.ones(n) / n
        super()._build_ensemble(X, y)
    
    def fit_new_learner(self):
        learner = self.learner_model.copy()
        learner.fit(self.X, self.y, sample_weight=self.w)
        y_pred = learner.predict(self.X)
        error = np.sum(self.w * (y_pred != self.y.values))
        alpha = 0.5 * np.log((1 - error) / error)
        self.w = self._update_weights(self.w, self.y.values, y_pred, alpha)
        self.alphas.append(alpha)
        self.trained_learners.append(learner)
        return self
        
    def _update_weights(self, w: np.ndarray, y: np.ndarray, y_pred: np.ndarray, alpha: float):
        # If there are more than 2 classes, we
        # can't rely on y_pred*y to get the sign
        # error = -1 and ok = 1, so we make the comparison
        # and then turn it into -1 and 1
        comparison = 2 * (y_pred == y).astype(int) - 1 # -1 or 1
        new_w = w * np.exp(-alpha * comparison)
        new_w = new_w / np.sum(new_w) # normalize
        return new_w
         
    def _get_voted_predictions(self, learner_idx: list, X: np.ndarray):
        learners = [self.trained_learners[i] for i in learner_idx]
        alphas = np.array([self.alphas[i] for i in learner_idx])
        y_preds = np.empty((X.shape[0], len(learners)), dtype=int)
        alphas = np.array(alphas)
        for i, learner in enumerate(learners):
            y_preds[:, i] = learner.predict(X)
        # Get a matrix with the count of votes for each label class
        # weighted by the alpha of the learner
        voted_y_preds = np.apply_along_axis(
            lambda row: np.bincount(row, minlength=len(self.label_values), weights=alphas),
            axis=1, arr=y_preds)
        return voted_y_preds

