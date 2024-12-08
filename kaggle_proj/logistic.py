import optuna
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
import numpy as np
import logging

import os

# Limit thread usage for better compatibility
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"


logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger()

from pre3_feature_eng import pre3_feature_eng_with_poly_vars
all_data, new_train, new_predict = pre3_feature_eng_with_poly_vars()

X = new_train.drop('income>50K', axis=1)
y = new_train['income>50K']

def objective(trial):
    # n pca components
    n_components = trial.suggest_int("n_components", 1, X.shape[1])
    # logistic regularization
    C = trial.suggest_float("C", 0.08, 10, log=True)


    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    
    model = LogisticRegression(solver='lbfgs', C=C, random_state=0, max_iter=1000)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    roc_auc_scores = cross_val_score(model, X_pca, y, cv=cv, scoring="roc_auc", n_jobs=1)
    
    mean_roc_auc = np.mean(roc_auc_scores)    
    return mean_roc_auc

def report(study, trial):
    print(f"Trial {trial.number}: {trial.value}, parameters: {trial.params}")


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=3000, n_jobs=-1, callbacks=[report])

print("Best Trial:")
best_trial = study.best_trial
print(f"  AUC-ROC: {best_trial.value:.4f}")
for key, value in best_trial.params.items():
    if key == "n_components":
        print(f"    {key}: {value} / {X.shape[1]}")
    print(f"    {key}: {value}")

best_pca = PCA(n_components=best_trial.params["n_components"])
X_pca = best_pca.fit_transform(X)

final_model = LogisticRegression(solver='lbfgs', C=best_trial.params["C"], random_state=0, max_iter=1000)
final_model.fit(X_pca, y)

# import pickle
# with open("best_logistic_model.pkl", "wb") as f:
#     pickle.dump(final_model, f)
# with open("best_pca.pkl", "wb") as f:
#     pickle.dump(best_pca, f)

# date
import datetime
date_as_string = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# Save the best parameters
with open(f'best_logistic_model_params{date_as_string}.txt', 'w') as f:
    f.write(f"Best Parameters:\n{best_trial.params}\n")
    f.write(f"Best Cross-Validated AUC:\n{best_trial.value:.4f}\n")

