import sys, os; sys.path.insert(0, os.path.abspath('.')) if os.path.abspath('.') not in sys.path else None
from time import time
import pandas as pd
from data.datasets import play_tennis_dataset
from utils.preprocessing import dataset_to_cat_encoded_dataset, transform_num_to_bin_median
from utils.stats import avg_error
from decision_tree.fast_id3 import FastID3
from ensemble_learning.bagged_trees import BaggedTrees

def train_test_run(adaboost, data, n):
    train, train_labels, test, test_labels = data.train, data.train_labels, data.test, data.test_labels
    if n == 1:
        adaboost.fit(train, train_labels)
        train_pred = adaboost.predict(train)
    else:
        adaboost.fit_new_learner()
        train_pred = adaboost.re_predict(0)
    
    failed_indexes = [i for i, (p, l) in enumerate(zip(train_pred, train_labels.values)) if p != l]
    train_error = avg_error(train_pred, train_labels.values)
    return train_error, None

def report(data):
    max_n = 15 # achieves stable 0 in between 1 and 10 depending on the random choice of features
    results = []

    adaboost = BaggedTrees(FastID3('infogain'), 1)
    for n in range(1, max_n + 1):
        t0 = time()
        train_error, test_error = train_test_run(adaboost, data, n)
        et = round(time() - t0, 2)
        progress = n / max_n
        print(f"n: {n}, Train Error: {round(train_error, 3)}, Time: {et}s, Progress: {progress * 100:.2f}%")
        results.append([n, train_error, test_error])


data = play_tennis_dataset()
data = dataset_to_cat_encoded_dataset(data)

report(data)