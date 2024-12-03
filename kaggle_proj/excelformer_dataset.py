import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pre3_feature_eng import pre3_feature_eng_as_dummies
import torch_frame
from torch_frame.data import Dataset, DataLoader
from torch_frame.transforms import MutualInformationSort
from torch_frame.typing import TaskType


batch_size = 128


# Load data
all_data, new_train, new_predict = pre3_feature_eng_as_dummies()

# Split the data into training+validation and test sets (80-20 split)
X = new_train.drop('income>50K', axis=1)
y = new_train['income>50K']
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

# # Split training+validation into training and validation sets (80-20 split)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=0, stratify=y_train_val)

# X_train = X
# y_train = y

# # Convert non-float64 columns from bool to int
for col in X_train.columns:
    if X_train[col].dtype == 'bool':
        X_train[col] = X_train[col].astype(int)
        X_val[col] = X_val[col].astype(int)
        X_test[col] = X_test[col].astype(int)
        new_predict[col] = new_predict[col].astype(int)



# Map all features (numerical + dummy encoded) to `stype.numerical`
col_to_stype = {
    **{col: torch_frame.numerical for col in X_train.columns},  # All columns in X_train are numerical
    "target": torch_frame.categorical  # Specify the target column as categorical
}

# Create training, validation, and testing datasets
train_dataset = Dataset(X_train.assign(target=y_train), col_to_stype=col_to_stype, target_col="target")
val_dataset = Dataset(X_val.assign(target=y_val), col_to_stype=col_to_stype, target_col="target")
test_dataset = Dataset(X_test.assign(target=y_test), col_to_stype=col_to_stype, target_col="target")

col_to_stype_no_target = {
    **{col: torch_frame.numerical for col in X_train.columns},
}

predict_dataset = Dataset(new_predict, col_to_stype=col_to_stype_no_target)

# Materialize datasets
train_dataset.materialize()
val_dataset.materialize()
test_dataset.materialize()
predict_dataset.materialize()
# dataset = train_dataset
# train_dataset = dataset[:0.7]  # First 80% for training
# test_dataset = dataset[0.7:]  # Last 20% for testing
# val_dataset = dataset[0.8:]  # Last 20% for testing

# Apply Mutual Information Sorting to the training set only


mutual_info_sort = MutualInformationSort(task_type=TaskType.BINARY_CLASSIFICATION)
mutual_info_sort.fit(train_dataset.tensor_frame, train_dataset.col_stats)

# Sort training features
sorted_train_tensor_frame = mutual_info_sort(train_dataset.tensor_frame)

# Use original tensor frames for validation and testing
val_tensor_frame = mutual_info_sort(val_dataset.tensor_frame)
test_tensor_frame = mutual_info_sort(test_dataset.tensor_frame)
predict_tensor_frame = mutual_info_sort(predict_dataset.tensor_frame)

# Create DataLoaders
train_loader = DataLoader(sorted_train_tensor_frame, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_tensor_frame, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_tensor_frame, batch_size=batch_size, shuffle=False)
predict_loader = DataLoader(predict_tensor_frame, batch_size=batch_size, shuffle=False)

def loaders():
    return train_loader, test_loader, val_loader

def datasets():
    return train_dataset, test_dataset, val_dataset

def get_predict_loader():
    return predict_loader