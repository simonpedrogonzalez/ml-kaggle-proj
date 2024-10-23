import numpy as np
import pandas as pd

def unique_proba(y: np.ndarray, w: np.ndarray):
    # assume int array and use bincount, is about 7 times faster than np.unique
    # and orders of magnitude faster than pd.Series.value_counts
    counts = np.bincount(y, weights=w)
    non_zero = counts.nonzero()
    counts = counts[non_zero]
    proba = counts / w.sum()
    values = non_zero[0]

    return values, counts, proba

def entropy(proba):
    if len(proba) == 0:
        return 0
    return -np.sum(proba * np.log2(proba))

def majority_error(proba):
    if len(proba) == 0:
        return 0
    return 1 - proba.max()

def gini_index(proba):
    if len(proba) == 0:
        return 0
    return 1 - np.sum(proba ** 2)

def gain(X: np.ndarray, y: np.ndarray, w: np.ndarray, feature: int, feature_values: list, metric_func: callable):
    X = X[:,feature]
    n = w.sum()
    _, y_counts, y_proba = unique_proba(y, w)
    total_metric = metric_func(y_proba)
    w_metric_subsets = np.zeros(len(feature_values))

    for i, value in enumerate(feature_values):
        mask = X == value
        subset_labels = y[mask]
        subset_w = w[mask]
        n_subset = subset_w.sum()

        if n_subset > 0:
            _, _, subset_proba = unique_proba(subset_labels, subset_w)
            subset_metric = metric_func(subset_proba)
            w_metric_subsets[i] = subset_metric * (n_subset / n)

    return total_metric - np.sum(w_metric_subsets)

def avg_error(y_true, y_pred):
    return np.mean(y_true != y_pred)

def sample(size: int, X: np.ndarray, y: np.ndarray, replace: bool=False):
    """Input is assumed to support len and indexing"""
    idx = np.random.choice(range(len(X)), size=size, replace=replace)
    return X[idx], y[idx]

def bootstrap_sample(X: np.ndarray, y: np.ndarray):
    """Input is assumed to support len and indexing"""
    return sample(len(X), X, y, replace=True)

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def cost(y_true, y_pred):
    return 1/2 * np.mean((y_true - y_pred) ** 2)

def endless_batch_generator(X: np.ndarray, y: np.ndarray, batch_size: int, random: bool=True):
    """Endless batch generator: starts over when it reaches the end"""
    while True:
        batch_gen = batch_generator(X, y, batch_size, random)
        while True:
            try:
                yield next(batch_gen)
            except StopIteration:
                break

def batch_generator(X: np.ndarray, y: np.ndarray, batch_size: int, random: bool=True):
    """Batch generator"""
    n = len(X)
    n_batches = n // batch_size
    indices = np.arange(n)
    
    if random:
        np.random.shuffle(indices)

    for i in range(n_batches):
        batch = indices[i*batch_size:(i+1)*batch_size]
        yield X[batch], y[batch]

