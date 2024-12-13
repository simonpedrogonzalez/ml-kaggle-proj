import sys, os; sys.path.insert(0, os.path.abspath('.')) if os.path.abspath('.') not in sys.path else None
from ensemble_learning.adaboost import AdaBoost
from ensemble_learning.bagged_trees import BaggedTrees
from decision_tree.fast_id3 import FastID3
from decision_tree.random_id3 import RandomID3
from data.datasets import income_level_dataset
from utils.stats import avg_error
from utils.preprocessing import dataset_to_cat_encoded_dataset, transform_num_to_cat, CatEncodedDataFrame, CatEncodedSeries
from time import time
import pandas as pd
import numpy as np

import seaborn as sns   
import matplotlib.pyplot as plt


data = income_level_dataset()

data.apply_transform_on_all_data(transform=lambda x: transform_num_to_cat(x, strat='qcut'))
data.apply_transform_on_all_data(transform=lambda x: CatEncodedDataFrame().from_pandas(x))
data.apply_transform_on_all_labels(transform=lambda x: CatEncodedSeries().from_pandas(x))

# data = dataset_to_cat_encoded_dataset(data)

n_features = len(data.train.features)

tree_models = {
    'fully_expanded': {
        'best': {
            'metric': 'majerr',
            'depth': 14
        },
        'default': {
            'metric': 'infogain',
            'depth': 14
        }
    },
    'best': {
        'best': {
            'metric': 'majerr',
            'depth': 4
        },
        'default': {
            'metric': 'gini',
            'depth': 3
        }
    },
    'stump': {
        'best': {
            'metric': 'majerr',
            'depth': 1
        },
        'default': {
            'metric': 'infogain',
            'depth': 1
        }
    }
}


def get_fast_id3(depth_type, model_type):
    return FastID3(tree_models[depth_type][model_type]['metric'], tree_models[depth_type][model_type]['depth'])

def get_random_id3(depth_type, model_type, feature_percent):
    feature_percents = {
        '20%': int(n_features * 0.2),
        '50%': int(n_features * 0.5),
        '100%': n_features
    }
    return RandomID3(tree_models[depth_type][model_type]['metric'], tree_models[depth_type][model_type]['depth'], feature_percents[feature_percent])

adamodels = {
    'ada_stump_best': AdaBoost(get_fast_id3('stump', 'best'), 1),
    'ada_stump_default': AdaBoost(get_fast_id3('stump', 'default'), 1),
    'ada_best_best': AdaBoost(get_fast_id3('best', 'best'), 1),
    'ada_best_default': AdaBoost(get_fast_id3('best', 'default'), 1),
}

rfmodels = {
    'rf_best_best_20%': BaggedTrees(get_random_id3('best', 'best', '20%'), 1),
    'rf_best_best_50%': BaggedTrees(get_random_id3('best', 'best', '50%'), 1),
    'rf_best_default_20%': BaggedTrees(get_random_id3('best', 'default', '20%'), 1),
    'rf_best_default_50%': BaggedTrees(get_random_id3('best', 'default', '50%'), 1),
    'rf_fully_expanded_best_20%': BaggedTrees(get_random_id3('fully_expanded', 'best', '20%'), 1),
    'rf_fully_expanded_best_50%': BaggedTrees(get_random_id3('fully_expanded', 'best', '50%'), 1),
    'rf_fully_expanded_default_20%': BaggedTrees(get_random_id3('fully_expanded', 'default', '20%'), 1),
    'rf_fully_expanded_default_50%': BaggedTrees(get_random_id3('fully_expanded', 'default', '50%'), 1)
}

all_models = {**adamodels, **rfmodels}

def train_test_run_one_model(data, model, n):
    train, train_labels, test, test_labels = data.train, data.train_labels, data.test, data.test_labels
    if n == 1:
        model.fit(train, train_labels)
        train_pred = model.predict(train)
        test_pred = model.predict(test)
    else:
        model.fit_new_learner()
        train_pred = model.re_predict(0)
        test_pred = model.re_predict(1)
    train_error = avg_error(train_pred, train_labels.values)
    test_error = avg_error(test_pred, test_labels.values)
    return train_error, test_error


def run(data):
    rows = []
    max_n = 500
    total = len(all_models) * max_n

    for name, model in all_models.items():
        for n in range(1, max_n + 1):
            t0 = time()
            train_error, test_error = train_test_run_one_model(data, model, n)
            et = round(time() - t0, 2)
            progress = len(rows) / total
            print(f"{name}, n: {n}, Train Error: {round(train_error, 3)}, Test Error: {round(test_error, 3)}, Time: {et}s, Progress: {progress * 100:.2f}%")
            rows.append([name, n, train_error, test_error])

    df = pd.DataFrame(rows, columns=['model', 'n', 'train_error', 'test_error'])
    df.to_csv('kaggle_proj/reports/ensemble.csv', index=False)
    
def report(data):

    max_n = 500
    df = pd.read_csv('kaggle_proj/reports/first_attempt_reports/ensemble.csv')


    # plot all ada models
    ada_df = df[df['model'].str.contains('ada')]
    plt.figure(figsize=(5, 5))
    sns.lineplot(data=ada_df, x='n', y='test_error', hue='model')
    plt.title('AdaBoost Test Error')
    plt.savefig('kaggle_proj/reports/ada_test_error2.png')

    # plot all rf models
    rf_df = df[df['model'].str.contains('rf')]
    plt.figure(figsize=(5, 5))
    sns.lineplot(data=rf_df, x='n', y='test_error', hue='model')
    plt.title('Random Forest Test Error')
    plt.savefig('kaggle_proj/reports/rf_test_error2.png')

    # plot best model of each type where best is the one with the lowest avg test error on the last 100 iterations
    best_models = []
    for name, model in all_models.items():
        test_errors = df[df['model'] == name]['test_error']
        avg_test_error = test_errors[-100:].mean()
        best_models.append((name, avg_test_error))
    best_models.sort(key=lambda x: x[1])
    best_models = best_models[:3]
    best_df = df[df['model'].isin([name for name, _ in best_models])]
    plt.figure(figsize=(5, 5))
    sns.lineplot(data=best_df, x='n', y='test_error', hue='model')
    plt.title('Best Models Test Error')
    plt.savefig('kaggle_proj/reports/best_models_test_error2.png')

    # best of all
    best_model = best_models[0][0]
    # use best model for to_predict
    model = all_models[best_model]
    model.fit(data.train, data.train_labels)
    train_pred = model.predict(data.train)
    test_pred = model.predict(data.test)
    train_error = avg_error(train_pred, data.train_labels.values)
    test_error = avg_error(test_pred, data.test_labels.values)
    pred_to_predict = model.predict(data.to_predict)
    best_name = best_model
    df2 = pd.DataFrame({'train_error': [train_error], 'test_error': [test_error], 'best_model': [best_name]})
    df2.to_csv('kaggle_proj/reports/best_model_report.csv', index=False)
    # map predictions to original values
    map_dict = data.train_labels.c2s
    mapped_pred_to_predict = [map_dict[pred] for pred in pred_to_predict]
    ids = np.arange(1, len(mapped_pred_to_predict) + 1)
    df3 = pd.DataFrame({'ID': ids, 'Prediction': mapped_pred_to_predict})
    df3.to_csv('kaggle_proj/reports/predictions.csv', index=False)


# run(data)
report(data)