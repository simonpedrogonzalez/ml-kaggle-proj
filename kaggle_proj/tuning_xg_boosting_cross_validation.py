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

X = new_train.drop('income>50K', axis=1)
y = new_train['income>50K']

def objective(trial):
    # Search space reduced around the values that proved good
    # in the random search
    params = {
        'alpha': trial.suggest_float('alpha', 0.2, 0.8), # centered around 0.422
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.8), # around 0.748
        'gamma': trial.suggest_float('gamma', 0.2, 0.3), # around 0.272
        'lambda': trial.suggest_float('lambda', 1.0, 2.5), # around 1.684
        'learning_rate': trial.suggest_float('learning_rate', 0.1, 0.2), # around 0.162
        'max_depth': trial.suggest_int('max_depth', 3, 5), # range around 4
        'min_child_weight': trial.suggest_int('min_child_weight', 0, 3), # around 1
        'n_estimators': trial.suggest_int('n_estimators', 200, 250), # around 222
        'subsample': trial.suggest_float('subsample', 0.85, 0.95) #around 0.887
    }
    
    model = xgb.XGBClassifier(objective="binary:logistic", random_state=0, **params)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    
    scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=1)
    return np.mean(scores)

def report(study, trial):
        print(f"Trial {trial.number}: {trial.value}, parameters: {trial.params}")

# default optuna algo is TPE
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=3000, n_jobs=-1, callbacks=[report])

print("Best parameters:", study.best_params)
print("Best cross-validated AUC:", study.best_value)

best_model = xgb.XGBClassifier(objective="binary:logistic", random_state=0, **study.best_params)
best_model.fit(X, y)

with open('best_xgb_model_cross_valid_tuning.pkl', 'wb') as f:
    pickle.dump(best_model, f)

# date
import datetime
date_as_string = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# Save the best parameters
with open(f'best_xgb_model_cross_valid_tuning_params{date_as_string}.txt', 'w') as f:
    f.write(f"Best Parameters:\n{study.best_params}\n")
    f.write(f"Best Cross-Validated AUC:\n{study.best_value:.4f}\n")

print("Hyperparameter tuning complete. Best model saved.")
