import optuna
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
import numpy as np
import logging
import xgboost as xgb
import os

# Limit thread usage for better compatibility
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"

logging.basicConfig(level=logging.INFO)

from pre3_feature_eng import pre3_feature_eng_with_poly_vars
all_data, new_train, new_predict = pre3_feature_eng_with_poly_vars()

X = new_train.drop('income>50K', axis=1)
y = new_train['income>50K']

def objective(trial):
    # n pca components
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

    n_components = trial.suggest_int("n_components", 1, X.shape[1])

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    
    model = xgb.XGBClassifier(objective="binary:logistic", random_state=0, **params)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    roc_auc_scores = cross_val_score(model, X_pca, y, cv=cv, scoring="roc_auc", n_jobs=1)
    
    mean_roc_auc = np.mean(roc_auc_scores)    
    return mean_roc_auc

def report(study, trial):
    print(f"Trial {trial.number}: {trial.value}, parameters: {trial.params}")

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=1000, n_jobs=-1, callbacks=[report])

print("Best Trial:")
best_trial = study.best_trial
print(f"  AUC-ROC: {best_trial.value:.4f}")
for key, value in best_trial.params.items():
    if key == "n_components":
        print(f"    {key}: {value} / {X.shape[1]}")
    print(f"    {key}: {value}")

# best_pca = PCA(n_components=best_trial.params["n_components"])
# X_pca = best_pca.fit_transform(X)

# final_model = xgb.XGBClassifier(
#     max_depth=best_trial.params["max_depth"],
#     learning_rate=best_trial.params["learning_rate"],
#     n_estimators=best_trial.params["n_estimators"],
#     subsample=best_trial.params["subsample"],
#     colsample_bytree=best_trial.params["colsample_bytree"],
#     objective="binary:logistic",
#     use_label_encoder=False,
#     random_state=0,
#     eval_metric="auc"
# )
# final_model.fit(X_pca, y)

# Save the best parameters
import datetime
date_as_string = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
with open(f'best_xgboost_model_params_poly_pca_{date_as_string}.txt', 'w') as f:
    f.write(f"Best Parameters:\n{best_trial.params}\n")
    f.write(f"Best Cross-Validated AUC:\n{best_trial.value:.4f}\n")
