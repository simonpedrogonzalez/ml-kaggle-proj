import sys, os; sys.path.insert(0, os.path.abspath('.')) if os.path.abspath('.') not in sys.path else None
from data.datasets import income_level_dataset_plain
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from kaggle_proj.preprocess1 import preprocess1

"""Correlation matrix of one hot encoded data"""

all_data, all_train, to_predict = preprocess1()

# one hot encoding all_data cat cols
all_train = pd.get_dummies(all_train)

# check if there is a string value somewhere
for col in all_train.columns:
    if all_train[col].dtype == 'object':
        print(col)

# correlation matrix
fig, ax = plt.subplots(figsize=(100, 100))
corr = all_train.corr()
sns.heatmap(corr, annot=True, fmt='.2f')
# export
plt.savefig('kaggle_proj/reports/one_hot_corr_matrix.png')
