from pre3_feature_eng import pre3_feature_eng_as_dummies
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split

from scipy.stats import uniform, randint
import numpy as np


# Load data
all_data, new_train, new_predict = pre3_feature_eng_as_dummies()


# Split data
X = new_train.drop('income>50K', axis=1)
y = new_train['income>50K']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)


xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=0)

# Define wider parameter ranges for a long random search
param_distributions = {
    'alpha': uniform(0, 10),                # L1 regularization, wide range
    'colsample_bytree': uniform(0.5, 0.5),  # Feature sampling ratio between 0.5 and 1.0
    'gamma': uniform(0, 0.5),               # Regularization, a larger range for conservative trees
    'lambda': uniform(0, 10),               # L2 regularization, wide range
    'learning_rate': uniform(0.01, 0.2),    # Learning rate from small to moderate
    'max_depth': randint(3, 10),            # Depth of trees, allowing shallow and deep trees
    'min_child_weight': randint(1, 10),     # Minimum sum of instance weight, affecting tree growth
    'n_estimators': randint(50, 300),       # Number of boosting rounds, from few to many
    'subsample': uniform(0.6, 0.4)          # Subsample from 0.6 to 1.0
}

def report_best_scores(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")



def display_scores(scores):
    print("Scores: {0}\nMean: {1:.3f}\nStd: {2:.3f}".format(scores, np.mean(scores), np.std(scores)))


n_iter=16800 # it's going to be a long fun night of tuning!!!
search = RandomizedSearchCV(xgb_model, param_distributions=param_distributions, random_state=42, n_iter=n_iter, cv=3, verbose=3, n_jobs=4, return_train_score=True)



search.fit(X, y)

report_best_scores(search.cv_results_, 1)

print('done')

# Best model:

best_model = search.best_estimator_
rebuild_best_model = xgb.XGBClassifier(objective="binary:logistic", random_state=0, **search.best_params_)
rebuild_best_model.fit(X_train, y_train)

pred = rebuild_best_model.predict(X_test)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

pred_proba = rebuild_best_model.predict_proba(X_test)
print(roc_auc_score(y_test, pred_proba[:, 1]))

# export the results
import pickle
with open('best_xgb_model.pkl', 'wb') as f:
    pickle.dump(rebuild_best_model, f)
    f.close()

# export the conf matrix, classification report and roc auc score
with open('best_xgb_model_results.txt', 'w') as f:
    f.write(f"Confusion Matrix:\n{confusion_matrix(y_test, pred)}\n")
    f.write(f"Classification Report:\n{classification_report(y_test, pred)}\n")
    f.write(f"ROC AUC Score:\n{roc_auc_score(y_test, pred_proba[:, 1])}\n")
    f.close()

# export the best parameters
with open('best_xgb_model_params.txt', 'w') as f:
    f.write(f"Best Parameters:\n{search.best_params_}\n")
    f.close()

print('done')
