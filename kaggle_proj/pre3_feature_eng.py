import sys, os; sys.path.insert(0, os.path.abspath('.')) if os.path.abspath('.') not in sys.path else None
import numpy as np
import pandas as pd
from kaggle_proj.pre2_missing_val import pre2_impute_mode, do_nothing
from data.datasets import Dataset

"""Transform and select variables"""





def pre3_feature_eng_as_dummies():
    # all_data, all_train, to_predict = pre2_impute_mode()
    all_data, all_train, to_predict = do_nothing()

    # Drop duplicates in all_train?
    # all_train = all_train.drop_duplicates(keep='first')

    # Capital diff
    all_data = all_data.assign(capital_diff = all_data['capital.gain'] - all_data['capital.loss'])
    all_data = all_data.drop(['capital.gain', 'capital.loss'], axis=1)

    # Freq encode native.country
    # freq = all_data['native.country'].value_counts(normalize=True)
    # all_data['native.country'] = all_data['native.country'].map(freq)

    # BEGIN OLD
    # Target encode native.country
    all_train_copy = all_train.copy()
    # # income>50K as numeric, currently is cat 0, 1
    # all_train_copy['income>50K'] = all_train_copy['income>50K'].astype(int)
    # target_mean = all_train_copy['income>50K'].mean()
    # target_encode = all_train_copy.groupby('native.country')['income>50K'].mean()
    # # fill na with target mean
    # target_encode = target_encode.fillna(target_mean)
    # all_data['native.country'] = all_data['native.country'].map(target_encode)
    # END OLD

    # Frequency encoding
    count_encode = all_train['native.country'].value_counts()  # Raw counts
    all_data['native.country'] = all_data['native.country'].map(count_encode).fillna(0)


    # Binning Age
    age_q = 7
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
    # bins = [0, 20, 40, 60, np.inf]
    # labels = ['<20', '20-40', '40-60', '60+']
    # all_data['hours.per.week'] = pd.cut(all_data['hours.per.week'], bins=bins, labels=labels)
    # ordinal encode
    # all_data['hours.per.week'] = all_data['hours.per.week'].cat.codes

    # begin test
    hours_q = 7
    quartile_edges = all_data['hours.per.week'].quantile([i / hours_q for i in range(hours_q + 1)]).unique()
    quartile_edges = quartile_edges.tolist()
    quartile_edges[0] = 0
    quartile_edges = [int(edge) for edge in quartile_edges]
    quartile_edges[-1] = np.inf
    quartile_labels = [f'{quartile_edges[i]}-{quartile_edges[i+1]}' for i in range(len(quartile_edges) - 1)]
    all_data['hours.per.week'] = pd.qcut(
        all_data['hours.per.week'],
        q=len(quartile_labels),
        duplicates='drop',
    )
    all_data['hours.per.week'] = all_data['hours.per.week'].cat.codes
    # end test




    # Drop fnlwgt, education
    all_data = all_data.drop(['fnlwgt', 'education'], axis=1)


    # Num vars (or ordinal vars)
    num_vars = ['capital_diff', 'native.country', 'age', 'hours.per.week', 'education.num']
    # scale num vars

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    all_data[num_vars] = scaler.fit_transform(all_data[num_vars])

    # # nominal vars
    # num categories
    # marital.status 7
    # occupation 15
    # relationship 6
    # race 5
    # sex 2
    cat_vars = ['workclass', 'marital.status', 'occupation', 'relationship', 'race', 'sex']
    # one hot encode
    all_data = pd.get_dummies(all_data, columns=cat_vars)

    new_train = all_data.iloc[:all_train.shape[0]]
    new_predict = all_data.iloc[all_train.shape[0]:]
    return all_data, new_train, new_predict


def pre3_feature_eng_keep_unencoded_categorical():
    # all_data, all_train, to_predict = pre2_impute_mode()
    all_data, all_train, to_predict = do_nothing()

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
    cat_vars = ['workclass', 'marital.status', 'occupation', 'relationship', 'race', 'sex']
    # convert all to astyoe str
    for col in cat_vars:
        all_data[col] = all_data[col].astype(str)


    new_train = all_data.iloc[:all_train.shape[0]]
    new_predict = all_data.iloc[all_train.shape[0]:]

    return all_data, new_train, new_predict


# def pre3_feature_eng_as_nominal():
#     # something went wrong here

#     # 80-20 split
#     new_all_data = all_data.copy()
#     # make intervals for the capital_diff
#     from utils.preprocessing import transform_num_to_cat

#     for col in new_all_data.columns:
#         if col == 'capital_diff':
#             continue
#         new_all_data[col] = new_all_data[col].astype('category')
#         new_all_data[col] = new_all_data[col].cat.codes
#         new_all_data[col] = new_all_data[col].astype('category')

#     new_all_data = transform_num_to_cat(new_all_data)

#     from utils.preprocessing import CatEncodedDataFrame, CatEncodedSeries
#     new_all_c = CatEncodedDataFrame().from_pandas(new_all_data[new_all_data.columns.difference(['income>50K'])])
#     new_all_c_labels = CatEncodedSeries().from_pandas(new_all_data['income>50K'])

#     # take 80 percent of the data for training
#     # the rest for testing
#     train_indices = np.random.choice(range(new_all_c.X.shape[0]), int(0.8 * new_all_c.X.shape[0]), replace=False)
#     new_train_c = new_all_c._empty_copy()
#     new_train_c.X = new_all_c.X[train_indices]


#     new_test_c = new_all_c._empty_copy()
#     new_test_c.X = new_all_c.X[~train_indices]

#     new_train_labels_c = new_all_c_labels._empty_copy()
#     new_train_labels_c.values = new_all_c_labels.values[train_indices]

#     new_test_labels_c = new_all_c_labels._empty_copy()
#     new_test_labels_c.values = new_all_c_labels.values[~train_indices]

#     new_to_predict_c = new_all_c._empty_copy()
#     new_to_predict_c.X = new_all_c.X[new_all_c.X.shape[0]:]

#     return Dataset(new_train_c, new_test_c, new_train_labels_c, new_test_labels_c, new_to_predict_c)


