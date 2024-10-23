from pre3_feature_eng import pre3_feature_eng_as_dummies
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import classification_report, roc_auc_score

# Load data
all_data, new_train, new_predict = pre3_feature_eng_as_dummies()

# Split data
X = new_train.drop('income>50K', axis=1)
y = new_train['income>50K']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

# Train model
rf = RandomForestClassifier(n_estimators=100, random_state=0)
rf.fit(X_train, y_train)

pred = rf.predict(X_test)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

pred_proba = rf.predict_proba(X_test)
print(roc_auc_score(y_test, pred_proba[:, 1]))




