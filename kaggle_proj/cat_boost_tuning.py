from pre3_feature_eng import pre3_feature_eng_as_dummies, pre3_feature_eng_keep_unencoded_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split
from catboost import CatBoostClassifier


# Load data
# all_data, new_train, new_predict = pre3_feature_eng_keep_unencoded_categorical()
all_data, new_train, new_predict = pre3_feature_eng_as_dummies()

# Split data
X = new_train.drop('income>50K', axis=1)
y = new_train['income>50K']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

# Train model
param_grid = {
    'iterations': [500, 1000, 1500, 2000],
    'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1],
    'depth': [4, 6, 8, 10],
    'l2_leaf_reg': [1, 3, 5, 7, 9],
    'border_count': [32, 64, 128, 256],
    'bagging_temperature': [0, 0.25, 0.5, 0.75, 1],
    'random_strength': [0, 1, 2, 4],
    'subsample': [0.7, 0.8, 0.9, 1],
    'colsample_bylevel': [0.7, 0.8, 0.9, 1],
    'boosting_type': ['Plain', 'Ordered'],
    'grow_policy': ['SymmetricTree', 'Depthwise', 'Lossguide']
}

catboost_model = CatBoostClassifier(random_state=0)

# Perform the grid search
grid_search_result = catboost_model.grid_search(
    param_grid, 
    X=X_train, y=y_train,  # Replace with your training data
    cv=5,                   # 5-fold cross-validation
    partition_random_seed=42,
    refit=True,             # Refit the best model after grid search
    calc_cv_statistics=True
)

# Display the best parameters and score
print("Best Parameters:", grid_search_result['params'])
print("Best Score:", grid_search_result['cv_results']['test-Accuracy-mean'].max())

# save results
grid_search_result['cv_results'].to_csv('catboost_grid_search_results.csv', index=False)

# Get the best model
best_model = grid_search_result['cv_results']['model'].values[0]

# Use the best model to predict
pred = best_model.predict(X_test)

print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
proba = best_model.predict_proba(X_test)
print(roc_auc_score(y_test, proba[:, 1]))

