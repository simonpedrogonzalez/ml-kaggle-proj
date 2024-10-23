import sys, os; sys.path.insert(0, os.path.abspath('.')) if os.path.abspath('.') not in sys.path else None
import pandas as pd
from time import time
from decision_tree.fast_id3 import FastID3
from utils.stats import avg_error
from utils.preprocessing import dataset_to_cat_encoded_dataset, transform_num_to_bin_median, impute_mode
from data.datasets import bank_dataset

def preprocess(data):
    data.train = transform_num_to_bin_median(data.train)
    data.test = transform_num_to_bin_median(data.test)
    data_with_unknown = dataset_to_cat_encoded_dataset(data)
    data.train = impute_mode(data.train, "unknown")
    data.test = impute_mode(data.test, "unknown")
    data_with_imp_values = dataset_to_cat_encoded_dataset(data)
    return data_with_unknown, data_with_imp_values

def train_test_run(data, metric, max_depth):
    train, train_labels, test, test_labels = data.train, data.train_labels, data.test, data.test_labels
    id3 = FastID3(metric, max_depth).fit(train, train_labels)
    train_pred = id3.predict(train)
    test_pred = id3.predict(test)
    train_error = avg_error(train_pred, train_labels.values)
    test_error = avg_error(test_pred, test_labels.values)
    return train_error, test_error

def report(data, exercise):
    metrics = ['infogain', 'majerr', 'gini']
    max_depths = range(1, 17)
    rows = []
    total = len(metrics) * len(max_depths)
    for i, metric in enumerate(metrics):
        for j, max_depth in enumerate(max_depths):
            t0 = time()
            train_error, test_error = train_test_run(data, metric, max_depth)
            t1 = time()
            print(f"Progress: {i * len(max_depths) + j + 1}/{total}, Metric: {metric}, Max Depth: {max_depth}, Time: {t1 - t0:.2f}s")            
            rows.append([metric, max_depth, train_error, test_error])
    print(f"Exporting report for exercise 3 {exercise}...")
    df = pd.DataFrame(rows, columns=['metric', 'max_depth', 'train_error', 'test_error'])
    df.to_csv(f"decision_tree/reports/h1e{exercise}_report.csv", index=False)
    df.to_latex(f"decision_tree/reports/h1e{exercise}_report.tex", index=False, longtable=True)

data = bank_dataset()
data_with_unknown, data_with_imp_values = preprocess(data)
report(data_with_unknown, '3b')
report(data_with_imp_values, '3c')
