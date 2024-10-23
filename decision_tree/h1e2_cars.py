import sys, os; sys.path.insert(0, os.path.abspath('.')) if os.path.abspath('.') not in sys.path else None
import pandas as pd
from decision_tree.fast_id3 import FastID3
from utils.stats import avg_error
from utils.preprocessing import dataset_to_cat_encoded_dataset
from data.datasets import cars_dataset

def train_test_run(data, metric, max_depth):
    train, test, train_labels, test_labels = data.train, data.test, data.train_labels, data.test_labels
    id3 = FastID3(metric, max_depth).fit(train, train_labels)
    train_pred = id3.predict(train)
    test_pred = id3.predict(test)
    train_error = avg_error(train_pred, train_labels.values)
    test_error = avg_error(test_pred, test_labels.values)
    return train_error, test_error

def report(data):
    metrics = ['infogain', 'majerr', 'gini']
    max_depths = range(1, 7)
    rows = []
    total = len(metrics) * len(max_depths)
    for i, metric in enumerate(metrics):
        for j, max_depth in enumerate(max_depths):
            train_error, test_error = train_test_run(data, metric, max_depth)
            rows.append([metric, max_depth, train_error, test_error])
            print(f"Progress: {i * len(max_depths) + j + 1}/{total}, Metric: {metric}, Max Depth: {max_depth}")            
    df = pd.DataFrame(rows, columns=['metric', 'max_depth', 'train_error', 'test_error'])
    print("Exporting report for exercise 2...")
    df.to_csv('decision_tree/reports/h1e2_report.csv', index=False)
    df.to_latex('decision_tree/reports/h1e2_report.tex', index=False, longtable=True)

data = cars_dataset()
data = dataset_to_cat_encoded_dataset(data)
report(data)
