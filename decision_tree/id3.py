import sys, os; sys.path.insert(0, os.path.abspath('.')) if os.path.abspath('.') not in sys.path else None
import numpy as np
import pandas as pd
from utils.stats import gain, entropy, majority_error, gini_index, unique_proba
from utils.preprocessing import CatEncodedDataFrame, CatEncodedSeries
from decision_tree.tree import Node, ID3NodeData

class ID3:
    def __init__(self, metric, max_depth=None):
        self.metric = metric
        self.max_depth = max_depth
        self.metric = {
            'infogain': entropy,
            'majerr': majority_error,
            'gini': gini_index
        }.get(metric)
        if self.metric is None:
            raise ValueError('Invalid metric')
        self.tree = None

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

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight: np.ndarray =None):
        """sample_weight: array-like of shape (n_samples,), default=None
        Sample weights. If None, then samples are equally weighted as 1.
        """
        self._preprocess(X, y)
        self.w = sample_weight if sample_weight is not None else np.ones(self.X.shape[0])
        self.tree = self._build_tree(self.X, self.y, self.w, self.feature_index, 0)
        return self

    def _build_tree(self, X: np.ndarray, y:np.ndarray, w:np.ndarray, features:list, depth:int):

        node = Node(ID3NodeData())

        label_values, label_counts, label_proba = unique_proba(y, w)
        majority_label =  label_values[np.argmax(label_counts)]
        majority_label_name = self.lc2s[majority_label]

        # Base cases
        base_case = None
        if len(features) == 0:
            base_case = "no_features"
        if len(label_counts) == 1:
            base_case = "pure_label"
        if self.max_depth and depth >= self.max_depth:
            base_case = "max_depth"

        if base_case is not None:
            node.data.leaf_type = base_case
            node.data.label = majority_label_name
            node.data.label_index = majority_label
            node.data.label_counts = {k: v for k, v in zip(label_values, label_counts)}
            node.data.label_proba = {k: v for k, v in zip(label_values, label_proba)}
            return node

        # Find best feature to split
        metric_value, feature_index, best_feature = self._pick_best_feature(X, y, w, features)
        best_feature_name = self.features[best_feature]

        node.data.next_feature = best_feature_name
        node.data.next_feature_index = best_feature
        node.data.metric = { self.metric.__name__: metric_value }

        # Get all possible values, not only the ones currently in X
        feature_values = self.feature_values[best_feature]

        # Add children
        for value in feature_values:
            value_name = self.c2s[(best_feature_name, value)]
            subset = X[:, best_feature] == value
            X_subset = X[subset]

            if X_subset.size == 0:
                child = Node(ID3NodeData(
                    leaf_type = "empty_subset",
                    label = majority_label_name,
                    label_index = majority_label,
                    feature = best_feature_name,
                    value = value_name,
                    value_index = value,
                    feature_index = best_feature
                ))
                node.add_child(child)
                continue

            y_subset = y[subset]
            w_subset = w[subset]
            new_features = features.copy()
            new_features.pop(feature_index)

            child = self._build_tree(X_subset, y_subset, w_subset, new_features, depth + 1)

            # add the data to the child node
            child.data.feature = best_feature_name
            child.data.feature_index = best_feature
            child.data.value = value_name
            child.data.value_index = value
            child_label_values, child_label_counts, child_label_proba = unique_proba(y_subset, w_subset)
            child.data.label_counts = {k: v for k, v in zip(child_label_values, child_label_counts)}
            child.data.label_proba = {k: v for k, v in zip(child_label_values, child_label_proba)}

            node.add_child(child)

        return node

    def predict(self, X: pd.DataFrame):
        if self.tree is None:
            raise ValueError('Model not fitted')
        if isinstance(X, pd.DataFrame):
            return self._predict_df(X, self.tree)
        elif isinstance(X, CatEncodedDataFrame):
            X = X.X
        if isinstance(X, np.ndarray):
            return self._predict_encoded_ndarray(X, self.tree)
        else:
            raise ValueError('Invalid input type')
    
    def _predict_encoded_ndarray(self, X: np.ndarray, node):
        return np.array([self._predict_idx_one(row, node) for row in X])
    
    def _predict_df(self, X: pd.DataFrame, node):
        return pd.Series([self._predict_value_one(row, node) for _, row in X.iterrows()])

    def _predict_idx_one(self, row: np.ndarray, node):
        if node.is_leaf:
            return node.data.label_index
        feature = node.data.next_feature_index
        value = row[feature]
        child = node.children.get((feature, value))
        if child is not None:
            return self._predict_idx_one(row, child)
        raise ValueError(f"Value {value} not found")

    def _predict_value_one(self, row: pd.Series, node):
        if node.is_leaf:
            return node.data.label
        feature_index = node.data.next_feature_index
        feature = node.data.next_feature
        value = row[feature]
        value_index = self.s2c[(feature, value)]
        child = node.children.get((feature_index, value_index))
        if child is not None:
            return self._predict_value_one(row, child)
        raise ValueError(f"Value {value} not found")

    def _pick_best_feature(self, X, y, w, features):
        gains = np.array([
            gain(X, y, w, feature, self.feature_values[feature], self.metric) \
                for feature in features
            ])
        idx = np.argmax(gains)
        minvalue = gains[idx]
        return minvalue, idx, features[idx]
