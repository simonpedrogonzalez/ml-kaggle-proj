from pre3_feature_eng import pre3_feature_eng_as_dummies
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split

from scipy.stats import uniform, randint
import numpy as np
import pickle
import pandas as pd


# Load data
all_data, new_train, new_predict = pre3_feature_eng_as_dummies()


# Split data
X = new_train.drop('income>50K', axis=1)
y = new_train['income>50K']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

# best model
with open('best_xgb_model.pkl', 'rb') as file:
    best_model = pickle.load(file)

pred = best_model.predict(X_test)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

pred_proba = best_model.predict_proba(X_test)
print(roc_auc_score(y_test, pred_proba[:, 1]))

# train with all new_train

best_model.fit(X, y)

# predict new_predict and write a pd df with ID and prediction

X_predict = new_predict.drop('income>50K', axis=1)

pred = best_model.predict(X_predict)
pred_proba = best_model.predict_proba(X_predict)

pred_df = pd.DataFrame(pred_proba[:, 1], columns=['Prediction'])

# add ID column 1 to n

pred_df['ID'] = range(1, len(pred_df) + 1)

pred_df.to_csv('boosting_predictions_best_model.csv', index=False)

