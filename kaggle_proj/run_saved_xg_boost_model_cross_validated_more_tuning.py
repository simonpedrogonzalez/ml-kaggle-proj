from pre3_feature_eng import pre3_feature_eng_as_dummies
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split

from scipy.stats import uniform, randint
import numpy as np
import pandas as pd



# Load data
all_data, new_train, new_predict = pre3_feature_eng_as_dummies()

# Split data

params = {'alpha': 0.2206221805728857, 'colsample_bytree': 0.7050997393917311, 'gamma': 0.25227398092194825, 'lambda': 1.4849340795599348, 'learning_rate': 0.1551739370120038, 'max_depth': 4, 'min_child_weight': 1, 'n_estimators': 218, 'subsample': 0.9400050852911308}

xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=0, **params)

# Split data
X = new_train.drop('income>50K', axis=1)
y = new_train['income>50K']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

# Train
xgb_model.fit(X_train, y_train)

# Predict
pred = xgb_model.predict(X_test)
pred_proba = xgb_model.predict_proba(X_test)

# Print the results
print(f"Confusion Matrix:\n{confusion_matrix(y_test, pred)}")
print(f"Classification Report:\n{classification_report(y_test, pred)}")
print(f"ROC AUC Score:\n{roc_auc_score(y_test, pred_proba[:, 1])}")


# Train with all new_train data and predict to_predict
xgb_model.fit(X, y)

X_predict = new_predict.drop('income>50K', axis=1)

pred = xgb_model.predict(X_predict)

pred_proba = xgb_model.predict_proba(X_predict)

pred_df = pd.DataFrame(pred_proba[:, 1], columns=['Prediction'])

# add ID column 1 to n

pred_df['ID'] = range(1, len(pred_df) + 1)

# date
import datetime

date_as_string = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

pred_df.to_csv(f"boosting_predictions_best_model_cross_valid_and_bayes_tuned_{date_as_string}.csv", index=False)
