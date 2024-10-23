import sys, os; sys.path.insert(0, os.path.abspath('.')) if os.path.abspath('.') not in sys.path else None
from ensemble_learning.adaboost import AdaBoost
from ensemble_learning.bagged_trees import BaggedTrees
from decision_tree.fast_id3 import FastID3
from decision_tree.id3 import ID3
from decision_tree.random_id3 import RandomID3
from data.datasets import income_level_dataset
from utils.stats import avg_error
from utils.preprocessing import dataset_to_cat_encoded_dataset, transform_num_to_cat, CatEncodedDataFrame, CatEncodedSeries
import seaborn as sns
import matplotlib.pyplot as plt
from time import time
import pandas as pd
# from kaggle_proj.pre3_feature_eng import pre3_feature_eng_as_nominal

report_path = 'kaggle_proj/reports/'

def fully_expanded_depth(data):
    try:
        df = pd.read_csv(report_path + 'fully_exp_trees.csv')
        return df['depth'].max()
    except FileNotFoundError:
        train, train_labels, test, test_labels = data.train, data.train_labels, data.test, data.test_labels
        res = []
        for m in ['infogain', 'gini', 'majerr']:
            id3 = ID3(metric=m).fit(train, train_labels)
            depth = id3.tree.get_depth()
            train_pred = id3.predict(train)
            test_pred = id3.predict(test)
            train_error = avg_error(train_pred, train_labels.values)
            test_error = avg_error(test_pred, test_labels.values)
            res.append({
                'metric': m,
                'depth': depth,
                'train_error': train_error,
                'test_error': test_error
            })
        df = pd.DataFrame(res)
        df.to_csv(report_path + 'fully_exp_trees.csv', index=False)
        depth = df['depth'].max()
        return depth

def train_test_run_one_model(data, metric, n):
    train, train_labels, test, test_labels = data.train, data.train_labels, data.test, data.test_labels
    tree = FastID3(metric, max_depth=n)
    tree.fit(train, train_labels)
    train_pred = tree.predict(train)
    test_pred = tree.predict(test)
    train_error = avg_error(train_pred, train_labels.values)
    test_error = avg_error(test_pred, test_labels.values)
    return train_error, test_error

def report(data):

    # fet_depth = fully_expanded_depth(data)
    # depths = list(range(1, fet_depth + 1))
    # metrics = ['infogain', 'gini', 'majerr']
    # res = []
    # total = len(depths) * len(metrics)

    # for depth in depths:
    #     for metric in metrics:
    #         t0 = time()
    #         train_err, test_err = train_test_run_one_model(data, metric, depth)
    #         tf = round(time() - t0, 3)
    #         progress = round(len(res) / total, 2)

    #         print(f"Depth: {depth}, Metric: {metric}, Train Error: {round(train_err, 3)}, Test Error: {round(test_err, 3)}, Time: {tf}s, Progress: {progress * 100:.2f}%")
            
    #         res.append({
    #             'depth': depth,
    #             'metric': metric,
    #             'train_error': train_err,
    #             'test_error': test_err
    #         })

    # df = pd.DataFrame(res)
    # df.to_csv(report_path + 'single_tree.csv', index=False)

    df = pd.read_csv(report_path + 'first_attempt_reports/single_tree.csv')

    # fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    # sns.lineplot(data=df, x='depth', y='train_error', hue='metric', ax=ax)
    # # set title and save image
    # ax.set_title('Single Tree Training Error vs Depth')
    # ax.set_ylabel('Error')
    # ax.set_xlabel('Depth')
    # plt.savefig(report_path + 'single_tree_train_error2.png')

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    sns.lineplot(data=df, x='depth', y='test_error', hue='metric', ax=ax)
    # set title and save image
    ax.set_title('Single Tree Test Error vs Depth')
    ax.set_ylabel('Error')
    ax.set_xlabel('Depth')
    plt.savefig(report_path + 'single_tree_test_error2.png')

data = income_level_dataset()

# data.apply_transform_on_all_data(transform=lambda x: transform_num_to_cat(x, strat='qcut'))
# data.apply_transform_on_all_data(transform=lambda x: CatEncodedDataFrame().from_pandas(x))
# data.apply_transform_on_all_labels(transform=lambda x: CatEncodedSeries().from_pandas(x))

# data = pre3_feature_eng_as_nominal()

report(data)