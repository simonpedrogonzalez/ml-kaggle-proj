import sys, os; sys.path.insert(0, os.path.abspath('.')) if os.path.abspath('.') not in sys.path else None
from utils.preprocessing import CatEncodedDataFrame, CatEncodedSeries
from utils.stats import bootstrap_sample
import numpy as np
from abc import ABC, abstractmethod

class Ensemble(ABC):
    def __init__(self, learner_model, n_learners):
        self.learner_model = learner_model
        self.initial_n_learners = n_learners
        self.trained_learners = []
        self.saved_predictions = []

    def fit(self, X: CatEncodedDataFrame, y: CatEncodedSeries):
        self.X = X
        self.y = y
        self.label_values = y.categories
        self._build_ensemble(self.X, self.y)
        return self

    def _build_ensemble(self, X, y):
        for i in range(self.initial_n_learners):
            self.fit_new_learner()
        return self

    @abstractmethod
    def fit_new_learner(self):
        pass

    @abstractmethod
    def _get_voted_predictions(self, learners_idx: list, X:np.ndarray):
        """This method will vary depending on the ensemble method.
        """
        pass
    
    def predict(self, X: CatEncodedDataFrame):
        X = X.X

        all_learner_idx = list(range(len(self.trained_learners)))
        voted_preds = self._get_voted_predictions(all_learner_idx, X)
        final_preds = np.argmax(voted_preds, axis=1)

        self.saved_predictions.append((X, voted_preds, len(self.trained_learners)))
        return final_preds
    
    def re_predict(self, prediction_index):
        """Get a new prediction by adding the predictions of the new tree only
        """
        X, old_voted_preds, old_n_learners = self.saved_predictions[prediction_index]
        new_learners_idx =list(range(old_n_learners, len(self.trained_learners)))
        new_voted_preds = self._get_voted_predictions(new_learners_idx, X)
        voted_preds = old_voted_preds + new_voted_preds
        final_preds = np.argmax(voted_preds, axis=1)
        self.saved_predictions[prediction_index] = (X, voted_preds, len(self.trained_learners))
        return final_preds