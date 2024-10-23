class ID3NodeData:
    def __init__(self, feature=None, feature_index=None, value=None, value_index=None, label=None, label_index=None, label_counts:dict=None, label_proba:dict=None, metric:dict=None, leaf_type:str=None, next_feature:str=None):
        self.feature = feature
        self.value = value
        self.label = label
        self.label_counts = label_counts
        self.label_proba = label_proba
        self.metric = metric
        self.leaf_type = leaf_type
        self.next_feature = next_feature
        self.feature_index = feature_index
        self.value_index = value_index
        self.label_index = label_index

    def _repr_current_attr(self):
        if self.feature is None:
            return f"()"
        s = f"({self.feature}={self.value}"
        if self.label_counts is not None:
            s += f", {self.label_counts}"
        if self.label_proba is not None:
            s_proba = "{" + ", ".join([f"{k}: {round(v,3)}" for k, v in self.label_proba.items()]) + "}"
            s += f", label_proba={s_proba}"
        s += ")"
        return s
    
    def _repr_next_attr(self):
        s=""
        if self.metric is not None:
            s += f", {list(self.metric.keys())[0]}={round(list(self.metric.values())[0],3)}"
        if self.label is not None:
            s += f", label={self.label}"
        if self.leaf_type is not None:
            s += f", leaf_type={self.leaf_type}"
        if self.next_feature is not None:
            s += f", feature={self.next_feature}"
        return f"[{s[2:]}]"

    def __repr__(self):
        return self._repr_current_attr() + self._repr_next_attr()


class Node:
    def __init__(self, data=None):
        self.data = data
        self.children = {}
        self.is_leaf = True
        
    def __repr__(self, level=0):
        r = "\t" * level + f"{self.data}\n"
        for child in self.children.values():
            r += child.__repr__(level + 1)
        return r
    
    def add_child(self, node):
        fi, vi = node.data.feature_index, node.data.value_index
        self.children[(fi, vi)] = node
        self.is_leaf = False
    
    # def get_depth(self):
    #     if self.is_leaf():
    #         return 0
    #     return 1 + max([child.get_depth() for child in self.children])


class FastNode:
    """Slighly faster node representation that assumes feature values are 0,1,2,3...
    So the index of the children list is the value of the feature."""

    def __init__(self, label=None, feature_to_split_by=None):
        self.children = []
        self.feature_to_split_by = None
        self.label = label
        self.is_leaf = True
    
    def add_child(self, node):
        self.children.append(node)
        self.is_leaf = False

    