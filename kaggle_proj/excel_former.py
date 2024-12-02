import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pre3_feature_eng import pre3_feature_eng_as_dummies

# Load data
all_data, new_train, new_predict = pre3_feature_eng_as_dummies()


# Split data
X = new_train.drop('income>50K', axis=1)
y = new_train['income>50K']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
# Numerical features

from torch_frame.data import Dataset, DataLoader
import torch_frame

# iter all non float64 columns and convert from bool to int
for col in X_train.columns:
    if X_train[col].dtype == 'bool':
        X_train[col] = X_train[col].astype(int)

# Map columns to their respective semantic types (stype)
# Map all features (numerical + dummy encoded) to `stype.numerical`
col_to_stype = {
    **{col: torch_frame.numerical for col in X_train.columns},  # All columns in X_train are numerical
    "target": torch_frame.categorical  # Specify the target column as categorical
}

# Set up the dataset with semantic types and target column
dataset = Dataset(X_train.assign(target=y_train), col_to_stype=col_to_stype, target_col="target")

# Materialize the dataset for PyTorch Frame
dataset.materialize()

# Split the dataset into training and testing subsets
train_dataset = dataset[:0.8]  # First 80% for training
test_dataset = dataset[0.8:]  # Last 20% for testing

# Create a DataLoader for batching and shuffling
train_loader = DataLoader(train_dataset.tensor_frame, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset.tensor_frame, batch_size=128, shuffle=False)

from torch_frame.nn.models import ExcelFormer
import torch
from torch_frame import stype

# Device setup (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Initialize ExcelFormer
model = ExcelFormer(
    in_channels=48,  # Number of numerical features
    out_channels=dataset.num_classes,# Number of target classes
    num_cols=len(dataset.feat_cols),# Total number of columns
    num_layers=4,  # Customize based on complexity
    num_heads=8,  # Number of attention heads
    col_stats=dataset.col_stats,  # Column statistics
    col_names_dict=dataset.tensor_frame.col_names_dict,  # Column names
).to(device)


import torch.nn.functional as F
from torch.optim import Adam

# Loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()  # Use BinaryCrossEntropy for binary classification
optimizer = Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):  # Adjust number of epochs as needed
    model.train()
    for tf in train_loader:
        tf = tf.to(device)  # Send TensorFrame to GPU/CPU
        optimizer.zero_grad()
        preds = model(tf)  # Forward pass
        loss = criterion(preds, tf.y)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update parameters
    print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
