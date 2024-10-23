import sys, os; sys.path.insert(0, os.path.abspath('.')) if os.path.abspath('.') not in sys.path else None
from time import time
import pandas as pd
from data.datasets import bank_dataset
from utils.preprocessing import dataset_to_cat_encoded_dataset, transform_num_to_bin_median
from utils.stats import avg_error, sample
from decision_tree.fast_id3 import FastID3
from ensemble_learning.bagged_trees import BaggedTrees
import numpy as np

def report(data):
    train, train_labels, test, test_labels = data.train, data.train_labels, data.test, data.test_labels
    repetitions = 100
    n_samples = 1000
    max_n = 500
    # get a prediction for each one of the experiments with different samples
    bt_test_predictions = np.zeros((len(test), repetitions))
    single_learner_test_predictions = np.zeros((len(test), repetitions))

    for i in range(repetitions):
        train_sample, train_sample_labels = sample(n_samples, train, train_labels, replace=False)
        print(f"Progress: {round((i + 1) / repetitions * 100, 2)}%")
        # predict test with the bagged trees
        learner = FastID3('infogain')
        bt = BaggedTrees(learner, max_n)
        bt.fit(train_sample, train_sample_labels)
        bt_test_predictions[:, i] = bt.predict(test)
        # predict test with the first single learner
        first_learner = bt.trained_learners[0]
        single_learner_test_predictions[:, i] = first_learner.predict(test)
    
    # transform from 0 1 labels to -1 1 labels for both predictions and test_labels
    bt_test_predictions = np.where(bt_test_predictions == 0, -1, 1)
    single_learner_test_predictions = np.where(single_learner_test_predictions == 0, -1, 1)
    test_labels = np.where(test_labels.values == 0, -1, 1)

    # take the average of the predictions for all 100 runs
    bt_mean_estimation = np.mean(bt_test_predictions, axis=1)
    sl_mean_estimation = np.mean(single_learner_test_predictions, axis=1)

    sl_bias = (sl_mean_estimation - test_labels)**2
    bt_bias = (bt_mean_estimation - test_labels)**2

    sl_var = np.var(single_learner_test_predictions, axis=1)
    bt_var = np.var(bt_test_predictions, axis=1)

    sl_mean_bias = np.mean(sl_bias)
    bt_mean_bias = np.mean(bt_bias)

    sl_mean_var = np.mean(sl_var)
    bt_mean_var = np.mean(bt_var)

    sl_error_est = sl_mean_bias + sl_mean_var
    bt_error_est = bt_mean_bias + bt_mean_var

    # profilactic check
    # ssss = []
    # for i in range(bt_test_predictions.shape[1]):
    #     ss = ((bt_test_predictions[:, i] - test_labels)**2).sum() / len(test_labels)
    #     ssss.append(ss)
    # should_be_error = np.array(ssss).mean()
    # assert np.isclose(should_be_error, bt_error_est)

    df = pd.DataFrame({
        'single_tree_bias': [sl_mean_bias],
        'bagged_tree_bias': [bt_mean_bias],
        'single_tree_variance': [sl_mean_var],
        'bagged_tree_variance': [bt_mean_var],
        'single_tree_error_estimation': [sl_error_est],
        'bagged_tree_error_estimation': [bt_error_est]
    })

    print(f"Results:\n{df}")
    print("Exporting report for exercise 2c...")
    df.to_csv('ensemble_learning/reports/h2e2c_report.csv', index=False)

data = bank_dataset()
data.train = transform_num_to_bin_median(data.train)
data.test = transform_num_to_bin_median(data.test)
data = dataset_to_cat_encoded_dataset(data)

report(data)