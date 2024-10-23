import sys, os; sys.path.insert(0, os.path.abspath('.')) if os.path.abspath('.') not in sys.path else None
from time import time
import pandas as pd
from data.datasets import bank_dataset
from utils.preprocessing import dataset_to_cat_encoded_dataset, transform_num_to_bin_median
from utils.stats import avg_error
from decision_tree.fast_id3 import FastID3
from ensemble_learning.adaboost import AdaBoost

def train_test_run(adaboost, data, n):
    train, train_labels, test, test_labels = data.train, data.train_labels, data.test, data.test_labels
    if n == 1:
        adaboost.fit(train, train_labels)
        train_pred = adaboost.predict(train)
        test_pred = adaboost.predict(test)
    else:
        adaboost.fit_new_learner()
        train_pred = adaboost.re_predict(0)
        test_pred = adaboost.re_predict(1)

    train_error = avg_error(train_pred, train_labels.values)
    test_error = avg_error(test_pred, test_labels.values)
    return train_error, test_error

def report(data):
    max_n = 500
    results = []

    adaboost = AdaBoost(FastID3('infogain', 1), 1)
    for n in range(1, max_n + 1):
        t0 = time()
        train_error, test_error = train_test_run(adaboost, data, n)
        et = round(time() - t0, 2)
        progress = n / max_n
        print(f"n: {n}, Train Error: {round(train_error, 3)}, Test Error: {round(test_error, 3)}, Time: {et}s, Progress: {progress * 100:.2f}%")
        results.append([n, train_error, test_error])

    print("Exporting report for exercise 2...")
    df = pd.DataFrame(results, columns=['n', 'train_error', 'test_error'])
    df.to_csv('ensemble_learning/reports/h2e2a_report.csv', index=False)
    df.to_latex('ensemble_learning/reports/h2e2a_report.tex', index=False, longtable=True)


data = bank_dataset()
data.train = transform_num_to_bin_median(data.train)
data.test = transform_num_to_bin_median(data.test)
data = dataset_to_cat_encoded_dataset(data)

report(data)