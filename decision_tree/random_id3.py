import sys, os; sys.path.insert(0, os.path.abspath('.')) if os.path.abspath('.') not in sys.path else None
import numpy as np
from decision_tree.tree import FastNode
from decision_tree.fast_id3 import FastID3

class RandomID3(FastID3):
    """Uses a random subset of features to build the tree at each split"""

    def __init__(self, metric, feature_sample_size, max_depth=None, ):
        self.feature_sample_size = feature_sample_size
        super().__init__(metric, max_depth)

    def copy(self):
        """Returns an untrained copy of the model"""
        return RandomID3(self.metric_name, self.feature_sample_size, self.max_depth)

    def _select_features(self, features):
        if len(features) <= self.feature_sample_size:
            return features
        return np.random.choice(features, self.feature_sample_size, replace=False)
    
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

        # Randomly select features
        feature_sample = self._select_features(features)
        
        _, best_feature, subsets = self._pick_best_feature(X, y, w, feature_sample)
        # Take the index from the original feature list
        feature_index = features.index(best_feature)

        # Continue as usual
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
