from pre3_feature_eng import pre3_feature_eng_as_dummies, pre3_feature_eng_keep_unencoded_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split
from catboost import CatBoostClassifier
import pandas as pd


# Load data
# all_data, new_train, new_predict = pre3_feature_eng_keep_unencoded_categorical()
all_data, new_train, new_predict = pre3_feature_eng_as_dummies()

# Split data
X = new_train.drop('income>50K', axis=1)
y = new_train['income>50K']

X_train = X
y_train = y

X_test = new_predict
X_test = X_test.drop('income>50K', axis=1)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

# Train model

# cat_vars = ['workclass', 'marital.status', 'occupation', 'relationship', 'race', 'sex']
# cat_model = CatBoostClassifier(cat_features=cat_vars, random_state=0)

cat_model = CatBoostClassifier(random_state=0)

cat_model.fit(X_train, y_train)

pred = cat_model.predict(X_test)

pred_proba = cat_model.predict_proba(X_test)

pred_df = pd.DataFrame(pred_proba[:, 1], columns=['Prediction'])

# add ID column 1 to n

pred_df['ID'] = range(1, len(pred_df) + 1)

# date
import datetime

date_as_string = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

pred_df.to_csv(f"cat_boost_{date_as_string}.csv", index=False)

