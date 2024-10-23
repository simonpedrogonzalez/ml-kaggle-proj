import sys, os; sys.path.insert(0, os.path.abspath('.')) if os.path.abspath('.') not in sys.path else None
from time import time
import pandas as pd
from data.datasets import credit_card_default_dataset
from utils.preprocessing import dataset_to_cat_encoded_dataset, transform_num_to_cat
from utils.stats import avg_error
from decision_tree.random_id3 import RandomID3
from decision_tree.fast_id3 import FastID3
from ensemble_learning.bagged_trees import BaggedTrees
from ensemble_learning.adaboost import AdaBoost

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

def train_test_run(models, data, n):
    train, train_labels, test, test_labels = data.train, data.train_labels, data.test, data.test_labels
    errors = []
    # models is a dictionary with the model name as key and the model as value
    for model_name, model in models.items():
        train_error, test_error = train_test_run_one_model(data, model, n)
        errors.append({'model': model_name, 'train_error': train_error, 'test_error': test_error})
    return errors

def report(data):
    ab = AdaBoost(FastID3('infogain', 1), 1)
    bt = BaggedTrees(FastID3('infogain'), 1)

    # take 20% of the features for random forest
    n_features = len(data.train.features)
    number_of_features_rf = int(n_features * 0.2)
    rf = BaggedTrees(RandomID3('infogain', feature_sample_size=number_of_features_rf), 1)

    models = {'adaboost': ab, 'bagged_trees': bt, 'random_forest': rf}

    errors = []
    max_n = 500

    for n in range(1, max_n + 1):
        t0 = time()
        step_errors = train_test_run(models, data, n)
        et = round(time() - t0, 2)
        progress = round(n / max_n, 2)
        
        error_message = f"{n} "
        for se in step_errors:
            model_name = se['model']
            train_error = se['train_error']
            test_error = se['test_error']
            error_message += f", {model_name} train_e: {round(train_error, 3)}, t_e: {round(test_error, 3)}"
            errors.append([n, model_name, train_error, test_error])
        error_message += f", Time: {et}s, Progress: {progress * 100:.2f}%"
        print(error_message)

    print("Exporting report for exercise 3...")
    df = pd.DataFrame(errors, columns=['n', 'model', 'train_error', 'test_error'])
    df.to_csv('ensemble_learning/reports/h2e3_report.csv', index=False)

    single_tree = bt.trained_learners[0]
    train_pred = single_tree.predict(data.train)
    test_pred = single_tree.predict(data.test)
    train_error = avg_error(train_pred, data.train_labels.values)
    test_error = avg_error(test_pred, data.test_labels.values)
    print(f"Single tree train error: {round(train_error, 3)}, test error: {round(test_error, 3)}")
    df2 = pd.DataFrame({'train_error': [train_error], 'test_error': [test_error]})
    df2.to_csv('ensemble_learning/reports/h2e3_single_tree_report.csv', index=False)


data = credit_card_default_dataset()
# Use 4 bins for each numerical feature based on quantiles
data.train = transform_num_to_cat(data.train, strat='qcut')
data.test = transform_num_to_cat(data.test, strat='qcut')
data = dataset_to_cat_encoded_dataset(data)

report(data)