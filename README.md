# ml-kaggle-proj
This is the Kaggle Project developed by Simon Gonzalez CS5350/6350 in University of Utah

# ml-library
This is a machine learning library developed by Simon Gonzalez for CS5350/6350 in University of Utah.

## Run Homework reports

Every homework has a `run_h*.sh` script that runs the homework and saves the output to a file in the corresponding `reports` folder. Take into account that **many scripts can take a long time to run**.

- HW1 Decision Tree: run `run_h1.sh`
- HW2 Bagging and Boosting: run `run_h2.sh`

## Library usage

It is not a package, it's meant to be cloned and run locally from the root.

### Loading data

1. Loading a dataset: all the used datasets are included in the library and can be loaded as follows:
```python
from data.datasets import play_tennis_dataset
data = play_tennis_dataset()
```
Each `Dataset` object has 2 `pd.DataFrame` objects (train and test) and 2 `pd.Series` objects (train_labels and test_labels).

### Preprocessing

Convert `Dataset` to categorical encoded `Dataset`, containing `CatEncodedDataset` and `CatEncodedSeries` instead of `pd.DataFrame` and `pd.Series`. Theese objects hold the data in `np.array` `int8` format for farster processing and also keep track of the original feature names and values.

```python
# Transform a full Dataset object
from utils.preprocessing import dataset_to_cat_encoded_dataset
data = dataset_to_cat_encoded_dataset(data)
# Transform single pd.DataFrame
from utils.preprocessing import CatEncodedDataset
data = CatEncodedDataset().from_pandas(df)
```
There are other functions, check `utils.preprocessing`.
### Decision Tree

General usage:

```python
from decision_tree.some_tree import MyTree

tree = MyTree(metric='infogain', max_depth=3).fit(train, train_labels)
y_pred = tree.predict(test)
```

Available trees:

```python
from decision_tree.fast_id3 import FastID3
# Uses CatEncodedDataset and CatEncodedSeries
from decision_tree.id3 import ID3
# ID3 can handle regular pd.DataFrame and pd.Series
# Also, produces a tree that can be printed to console.
from decision_tree.random_id3 import RandomID3
# Random ID3 selects a random subset of features
# from which to choose the best feature to split on
# at each node.
```

### Bagging and Boosting

General usage:

```python
from ensemble_learning.some_ensemble import MyEnsemble
from decision_tree.some_model import MyLearner
learner = MyLearner()
ensemble = MyEnsemble(learner_model=learner, n_learners=100)
ensemble.fit(train, train_labels)
y_pred = ensemble.predict(test)
# Add one additional learner, so it would be 101.
ensemble.fit_new_learner()
```
Available ensembles:
```python
from ensemble_learning.adaboost import AdaBoost
from ensemble_learning.bagged_trees import BaggedTrees
from decision_tree.fast_id3 import FastID3
from decision_tree.random_id3 import RandomID3

# Adaboost
stump = FastID3('infogain', 1)
adaboost = AdaBoost(stump, 100)
# Regular bagged trees
id3 = FastID3('infogain')
bagged_trees = BaggedTrees(id3, 100)
# Ramdom Forest
rid3 = RandomID3('infogain', feature_sample_size=3)
random_forest = BaggedTrees(rid3, 100)
```

### Linear regression

General usage:

```python
from linear_regression.some_regressor import MyRegressor

regressor = MyRegressor(max_iter=1000, lr=0.01, atol=1e-5, batch_size=32)
regressor.fit(train, train_labels)
y_pred = regressor.predict(test)
```

Available regressors:

```python
from linear_regression.linear_regressor import AnalytucalRegressor
from linear_regression.gradient_descent import \
     BatchGradientDescent, StochasticGradientDescent
```
