import sys, os; sys.path.insert(0, os.path.abspath('.')) if os.path.abspath('.') not in sys.path else None
import numpy as np
import pandas as pd
from utils.stats import entropy, majority_error, gini_index, unique_proba
from utils.preprocessing import CatEncodedDataFrame, CatEncodedSeries
from decision_tree.tree import FastNode

class FastID3:
    """Slighly faster version of ID3, but produces unreadable trees"""

    def __init__(self, metric, max_depth=None):
        self.metric = metric
        self.max_depth = max_depth
        self.metric_name = metric
        self.metric = {
            'infogain': entropy,
            'majerr': majority_error,
            'gini': gini_index
        }.get(metric)
        if self.metric is None:
            raise ValueError('Invalid metric')
        self.tree = None

    def copy(self):
        """Returns an untrained copy of the model"""
        return FastID3(self.metric_name, self.max_depth)

    def _preprocess(self, data, labels):
        if isinstance(data, pd.DataFrame):
            data = CatEncodedDataFrame().from_pandas(data)
        if isinstance(data, CatEncodedDataFrame):
            self.X = data.X
            self.features = data.features
            self.feature_index = data.feature_index
            self.feature_values = data.feature_values
            self.c2s = data.c2s
            self.s2c = data.s2c
        elif isinstance(data, np.ndarray):
            self.X = data
            self.features = list(range(data.shape[1]))
            self.feature_index = list(range(data.shape[1]))
            self.feature_values = {i: np.unique(data[:, i]) for i in range(data.shape[1])}
            self.c2s = { (i, v): v for i in range(data.shape[1]) for v in np.unique(data[:, i]) }
            self.s2c = self.c2s
        else:
            raise ValueError('Invalid data type')
        if isinstance(labels, pd.Series):
            labels = CatEncodedSeries().from_pandas(labels)
        if isinstance(labels, CatEncodedSeries):
            self.y = labels.values
            self.label_values = labels.categories
            self.label_index = labels.category_index
            self.lc2s = labels.c2s
            self.ls2c = labels.s2c
        elif isinstance(labels, np.ndarray):
            self.y = labels
            self.label_values = np.unique(labels)
            self.label_index = list(range(len(self.label_values)))
            self.lc2s = { i: v for i, v in enumerate(self.label_values) }
            self.ls2c = { v: i for i, v in enumerate(self.label_values) }
        else:
            raise ValueError('Invalid labels type')

    def fit(self, X: CatEncodedDataFrame, y: CatEncodedSeries, sample_weight: np.ndarray =None):
        """sample_weight: array-like of shape (n_samples,), default=None
        Sample weights. If None, then samples are equally weighted as 1.
        """
        self._preprocess(X, y)
        self.w = sample_weight if sample_weight is not None else np.ones(self.X.shape[0])
        initial_proba = unique_proba(self.y, self.w)
        self.tree = self._build_tree(self.X, self.y, self.w, self.feature_index, 0, initial_proba)
        return self

    def _build_tree(self, X: np.ndarray, y:np.ndarray, w:np.ndarray, features:list, depth:int, y_proba:tuple):

        node = FastNode()

        label_values, label_counts, label_proba = y_proba
        majority_label =  label_values[np.argmax(label_counts)]

        # Base cases
        if len(features) == 0 or \
            len(label_counts) == 1 or \
            (self.max_depth and depth >= self.max_depth):
            node.label = majority_label
            return node

        # Find best feature to split
        # Capture value subsets (sX, sy, sw) to avoid resubsetting
        # and their label probas (syps) to avoid recalculating
        feature_index, best_feature, subsets = self._pick_best_feature(X, y, w, features)

        node.feature_to_split_by = best_feature

        # Add children
        for sX, sy, sw, syps in subsets:

            if sy.size == 0:
                node.add_child(FastNode(label = majority_label))
                continue

            new_features = features.copy()
            new_features.pop(feature_index)

            node.add_child(self._build_tree(sX, sy, sw, new_features, depth + 1, syps))

        return node

    def predict(self, X: CatEncodedDataFrame):
        if self.tree is None:
            raise ValueError('Model not fitted')
        if isinstance(X, CatEncodedDataFrame):
            X = X.X
        return self._predict_encoded_ndarray(X, self.tree)
        
    def _predict_encoded_ndarray(self, X: np.ndarray, node):
        return np.array([self._predict_idx_one(row, node) for row in X])

    def _predict_idx_one(self, row: np.ndarray, node):
        if node.is_leaf:
            return node.label
        feature = node.feature_to_split_by
        value = row[feature]
        child = node.children[value] # the value is the index, assumes feature_values are 0,1,2,3...
        return self._predict_idx_one(row, child)
        
    def _pick_best_feature(self, X, y, w, features):
        metric = self.metric
        min_m = np.inf
        selected_feature_index = -1
        selected_feature_subsets = None
        n = w.sum()
        for i, f in enumerate(features):
            fvs = self.feature_values[f]
            ms = np.zeros(len(fvs))
            subsets = []
            for j, value in enumerate(fvs):
                mask = X[:, f] == value
                subset_y = y[mask]
                subset_w = w[mask]
                subset_n = subset_w.sum()
                if subset_n > 0:
                    sv, sc, sp = unique_proba(subset_y, subset_w)
                    ms[j] = metric(sp) * (subset_n / n)
                    subset_data = (X[mask], subset_y, subset_w, (sv, sc, sp)) 
                else:
                    subset_data = (X[mask], subset_y, subset_w, (0,0,0))
                subsets.append(subset_data)
            subset_total_m = np.sum(ms)
            if subset_total_m < min_m:
                min_m = subset_total_m
                selected_feature_index = i
                selected_feature_subsets = subsets
                if abs(subset_total_m) < 1e-8: # if perfect split return immediately
                    # also, np.isclose takes about 20x longer than this
                    return selected_feature_index, features[selected_feature_index], selected_feature_subsets
        return selected_feature_index, features[selected_feature_index], selected_feature_subsets