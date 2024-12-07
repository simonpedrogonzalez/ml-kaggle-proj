import optuna
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import numpy as np
import pickle
import logging

# Load data
from pre3_feature_eng import pre3_feature_eng_as_dummies
all_data, new_train, new_predict = pre3_feature_eng_as_dummies()