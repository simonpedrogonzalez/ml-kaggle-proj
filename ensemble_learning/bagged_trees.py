import sys, os; sys.path.insert(0, os.path.abspath('.')) if os.path.abspath('.') not in sys.path else None
from utils.preprocessing import CatEncodedDataFrame, CatEncodedSeries
from utils.stats import sample
from ensemble_learning.ensemble import Ensemble
import numpy as np

class BaggedTrees(Ensemble):

    def __init__(self, learner_model, n_learners, sample_size=None):
        self.sample_size = sample_size if sample_size is not None else 'bootstrap'
        super().__init__(learner_model, n_learners)

    def fit(self, X: CatEncodedDataFrame, y: CatEncodedSeries):
        if self.sample_size == 'bootstrap':
            self.sample_size = len(X)
        super().fit(X, y)

    def fit_new_learner(self):
        X, y = sample(self.sample_size, self.X, self.y, replace=True)
        tree = self.learner_model.copy()
        tree.fit(X, y)
        self.trained_learners.append(tree)
        return self

    def _get_voted_predictions(self, learners_idx: list, X: np.ndarray):
        learners = [self.trained_learners[i] for i in learners_idx]
        y_preds = np.empty((X.shape[0], len(learners)), dtype=int)
        for i, tree in enumerate(learners):
            y_preds[:, i] = tree.predict(X)
        # Get a matrix with the count of votes for each label class
        # We cannot take the mean of the predictions if there 
        # are more than 2 classes, so we count the predictions for
        # each class. It is clear that for 2 classes, this method
        # is equivalent to taking the mean
        voted_y_pred = np.apply_along_axis(
            lambda row: np.bincount(row, minlength=len(self.label_values)),
            axis=1, arr=y_preds)
        return voted_y_pred
