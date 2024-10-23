import sys, os; sys.path.insert(0, os.path.abspath('.')) if os.path.abspath('.') not in sys.path else None
import numpy as np
import pandas as pd
from kaggle_proj.pre2_missing_val import pre2_impute_mode

"""Transform and select variables"""

all_data, all_train, to_predict = pre2_impute_mode()

# Drop duplicates in all_train?
# all_train = all_train.drop_duplicates(keep='first')

# Capital diff
all_data = all_data.assign(capital_diff = all_data['capital.gain'] - all_data['capital.loss'])
all_data = all_data.drop(['capital.gain', 'capital.loss'], axis=1)

# Freq encode native.country
# freq = all_data['native.country'].value_counts(normalize=True)
# all_data['native.country'] = all_data['native.country'].map(freq)

# Target encode native.country
all_train_copy = all_train.copy()
# income>50K as numeric, currently is cat 0, 1
all_train_copy['income>50K'] = all_train_copy['income>50K'].astype(int)
target_mean = all_train_copy['income>50K'].mean()
target_encode = all_train_copy.groupby('native.country')['income>50K'].mean()
# fill na with target mean
target_encode = target_encode.fillna(target_mean)
all_data['native.country'] = all_data['native.country'].map(target_encode)


# Binning Age
age_q = 6
quartile_edges = all_data['age'].quantile([i/age_q for i in range(age_q + 1)])
quartile_edges = quartile_edges.to_list()
quartile_edges[0] = 0
quartile_edges = [int(edge) for edge in quartile_edges]
quartile_edges[-1] = np.inf
quartile_labels = [f'{quartile_edges[i]}-{quartile_edges[i+1]}' for i in range(age_q)]
all_data['age'] = pd.qcut(all_data['age'], q=age_q, labels=quartile_labels)
# ordinal encode
all_data['age'] = all_data['age'].cat.codes

# Binning hours per week in a different work week categories
# less than 20, 20-40, 40-60, 60+
bins = [0, 20, 40, 60, np.inf]
labels = ['<20', '20-40', '40-60', '60+']
all_data['hours.per.week'] = pd.cut(all_data['hours.per.week'], bins=bins, labels=labels)
# ordinal encode
all_data['hours.per.week'] = all_data['hours.per.week'].cat.codes

# Drop fnlwgt, education
all_data = all_data.drop(['fnlwgt', 'education'], axis=1)

# Num vars (or ordinal vars)
num_vars = ['capital_diff', 'native.country', 'age', 'hours.per.week', 'education.num']
# scale num vars
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
all_data[num_vars] = scaler.fit_transform(all_data[num_vars])


# nominal vars
cat_vars = ['workclass', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'income>50K']
# one hot encode
all_data = pd.get_dummies(all_data, columns=cat_vars)

print(all_data.head())

def pre3_feature_eng():
    new_train = all_data.iloc[:all_train.shape[0]]
    new_predict = all_data.iloc[all_train.shape[0]:]
    return all_data, new_train, new_predict



