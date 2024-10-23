import sys, os; sys.path.insert(0, os.path.abspath('.')) if os.path.abspath('.') not in sys.path else None
from time import time
import pandas as pd
from data.datasets import bank_dataset
from utils.preprocessing import dataset_to_cat_encoded_dataset, transform_num_to_bin_median
from utils.stats import avg_error
from decision_tree.fast_id3 import FastID3
from ensemble_learning.adaboost import AdaBoost
import numpy as np


def train_test_run(adaboost, data, n):
    train, train_labels, test, test_labels = data.train, data.train_labels, data.test, data.test_labels
    last_w = adaboost.w
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

def errors_of_each_learner(adaboost, data, ws):
    train, train_labels, test, test_labels = data.train, data.train_labels, data.test, data.test_labels
    learners = adaboost.trained_learners
    train_errors = []
    test_errors = []
    w_train_errors = []
    w_test_errors = []

    def w_error(y, y_pred, w):
        return np.sum(w * (y != y_pred))

    for i, learner in enumerate(learners):
        train_pred = learner.predict(train)
        test_pred = learner.predict(test)
        train_error = avg_error(train_pred, train_labels.values)
        test_error = avg_error(test_pred, test_labels.values)
        train_errors.append(train_error)
        test_errors.append(test_error)
        w_train_error = w_error(train_labels.values, train_pred, ws[i])
        w_test_error = w_error(test_labels.values, test_pred, ws[i])
        w_train_errors.append(w_train_error)
        w_test_errors.append(w_test_error)

    return train_errors, test_errors, w_train_errors, w_test_errors

def report(data):
    max_n = 500
    results = []
    ws = []

    adaboost = AdaBoost(FastID3('infogain', 1), 1)
    for n in range(1, max_n + 1):
        t0 = time()
        train_error, test_error = train_test_run(adaboost, data, n)
        et = round(time() - t0, 2)
        progress = n / max_n
        print(f"n: {n}, Train Error: {round(train_error, 3)}, Test Error: {round(test_error, 3)}, Time: {et}s, Progress: {progress * 100:.2f}%")
        results.append([n, train_error, test_error])
        ws.append(adaboost.w)

    print("Exporting report for exercise 2a...")
    df = pd.DataFrame(results, columns=['n', 'train_error', 'test_error'])
    df.to_csv('ensemble_learning/reports/h2e2a_1st_report.csv', index=False)

    ws = [[0] * len(ws[0])] + ws[:-1]
    ws = np.array(ws)

    print("Evaluating single learner errors...")
    train_errors, test_errors, w_train_errors, w_test_errors = errors_of_each_learner(adaboost, data, ws)
    learners = list(range(1, len(train_errors) + 1))
    print("Exporting single learner errors...")
    df = pd.DataFrame({'t': learners, 'train_error': train_errors, 'test_error': test_errors})
    df.to_csv('ensemble_learning/reports/h2e2a_2nd_report.csv', index=False)
    df = pd.DataFrame({'t': learners, 'train_error': w_train_errors, 'test_error': w_test_errors})
    df.to_csv('ensemble_learning/reports/h2e2a_4th_report.csv', index=False)
    
    # single tree run
    print("Evaluating single tree run...")
    id3 = FastID3('infogain')
    id3.fit(data.train, data.train_labels)
    train_pred = id3.predict(data.train)
    test_pred = id3.predict(data.test)
    train_error = avg_error(train_pred, data.train_labels.values)
    test_error = avg_error(test_pred, data.test_labels.values)
    print(f"Train Error: {round(train_error, 3)}, Test Error: {round(test_error, 3)}")
    df = pd.DataFrame([[train_error, test_error]], columns=['train_error', 'test_error'])
    df.to_csv('ensemble_learning/reports/h2e2a_3rd_report.csv', index=False)

    

data = bank_dataset()
data.train = transform_num_to_bin_median(data.train)
data.test = transform_num_to_bin_median(data.test)
data = dataset_to_cat_encoded_dataset(data)

report(data)