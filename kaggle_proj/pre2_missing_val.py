import sys, os; sys.path.insert(0, os.path.abspath('.')) if os.path.abspath('.') not in sys.path else None
import numpy as np
import pandas as pd
from kaggle_proj.pre1_read_data import pre1_read_data

"""Test different ways of handling missing values"""

all_data, all_train, to_predict = pre1_read_data()


all_train = all_train.replace('?', np.nan)
print(f"Current len of all_train: {all_train.shape[0]}")
print(f"Percentage of missing values in train: {all_train.isnull().sum().sum() / all_train.shape[0]}")
to_predict = to_predict.replace('?', np.nan)
print(f"Current len of to_predict: {to_predict.shape[0]}")
print(f"Percentage of missing values in to_predict: {to_predict.isnull().sum().sum() / to_predict.shape[0]}")

def pre2_impute_mode():
    # impute missing values with mode
    new_all_train = all_train.fillna(all_train.mode().iloc[0])
    # remove income>50k row from to_predict because it has nans
    new_to_predict = to_predict.drop('income>50K', axis=1)
    new_to_predict = new_to_predict.fillna(new_to_predict.mode().iloc[0])
    # add income>50k column to new_to_predict again
    new_to_predict['income>50K'] = np.nan
    new_all_data = pd.concat([new_all_train, new_to_predict])
    
    return new_all_data, new_all_train, new_to_predict


