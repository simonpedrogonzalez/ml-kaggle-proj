import sys, os; sys.path.insert(0, os.path.abspath('.')) if os.path.abspath('.') not in sys.path else None
from data.datasets import income_level_dataset_plain
import numpy as np
import pandas as pd

"""Read data, set categories and order categories"""

data = income_level_dataset_plain()

df = data.train
df_predict = data.to_predict
df_predict['income>50K'] = np.nan

all_vars = ['age', 'workclass', 'fnlwgt', 'education', 'education.num', 'marital.status', 'occupation', 'relationship',
            'race', 'sex', 'capital.gain', 'capital.loss', 'hours.per.week', 'native.country', "income>50K"]

num_vars = ['age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']
cat_vars = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', "income>50K", 
            'sex', 'race', 'native.country']

target = "income>50K"


train_indices = range(df.shape[0])
to_predict_indices = range(df.shape[0], df.shape[0] + df_predict.shape[0])

df_all = pd.concat([df, df_predict])

# reindex
df_all.reset_index(drop=True, inplace=True)
# drop 'id' column
df_all.drop('ID', axis=1, inplace=True)


df = df_all


for var in cat_vars:
    df[var] = df[var].astype('category')



# make a dict of all categories
# cat_dict = {col: df[col].cat.categories for col in cat_vars}

# reorder the categories
# workclass order: Private, Self-emp-not-inc, Self-emp-inc, Local-gov, State-gov, Federal-gov, Without-pay, Never-worked, ?
df['workclass'] = df['workclass'].cat.reorder_categories(
    ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Local-gov', 'State-gov', 'Federal-gov', 'Without-pay',
    'Never-worked', '?'])
# education education_levels = ['Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th', '11th', '12th', 'HS-grad', 'Some-college', 'Assoc-voc', 'Assoc-acdm', 'Bachelors', 'Masters', 'Prof-school', 'Doctorate']
df['education'] = df['education'].cat.reorder_categories(
    ['Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th', '11th', '12th', 'HS-grad', 'Some-college',
    'Assoc-voc', 'Assoc-acdm', 'Bachelors', 'Masters', 'Prof-school', 'Doctorate'])

# marital_status = ['Never-married', 'Married-civ-spouse', 'Married-AF-spouse', 'Married-spouse-absent', 'Separated', 'Divorced', 'Widowed']
df['marital.status'] = df['marital.status'].cat.reorder_categories(
    ['Never-married', 'Married-civ-spouse', 'Married-AF-spouse', 'Married-spouse-absent', 'Separated', 'Divorced',
    'Widowed'])

# relationship = ['Husband', 'Wife', 'Own-child', 'Other-relative', 'Unmarried', 'Not-in-family']

df['relationship'] = df['relationship'].cat.reorder_categories(
    ['Husband', 'Wife', 'Own-child', 'Other-relative', 'Unmarried', 'Not-in-family'])

# race = ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', ]

df['race'] = df['race'].cat.reorder_categories(
    ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'])


all_train = df.iloc[train_indices]
to_predict = df.iloc[to_predict_indices]

def pre1_read_data():
    return df, all_train, to_predict