
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pre3_feature_eng import pre3_feature_eng_as_dummies
# Load data
all_data, new_train, new_predict = pre3_feature_eng_as_dummies()
# Split data
X = new_train.drop('income>50K', axis=1)
y = new_train['income>50K']


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
X_train = X
y_train = y



# Numerical features
from torch_frame.data import Dataset, DataLoader
import torch_frame
# iter all non float64 columns and convert from bool to int
for col in X_train.columns:
    if X_train[col].dtype == 'bool':
        X_train[col] = X_train[col].astype(int)
# Map all features (numerical + dummy encoded) to stype.numerical
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


from torch_frame.transforms import MutualInformationSort
from torch_frame.typing import TaskType
mutual_info_sort = MutualInformationSort(task_type=TaskType.BINARY_CLASSIFICATION)  # Adjust task_type as needed
mutual_info_sort.fit(train_dataset.tensor_frame, train_dataset.col_stats)
# train_dataset.tensor_frame = mutual_info_sort(train_dataset.tensor_frame)
sorted_tensor_frame = mutual_info_sort(train_dataset.tensor_frame)



# Create a DataLoader for batching and shuffling
train_loader = DataLoader(sorted_tensor_frame, batch_size=512, shuffle=True)
test_loader = DataLoader(sorted_tensor_frame, batch_size=512, shuffle=False)


import os
from torch_frame.nn.models import ExcelFormer
import torch
from torch_frame import stype

# Paths for saving and loading model checkpoints
model_checkpoint_path = "excelformer_checkpoint.pth"

# Device setup (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize ExcelFormer
model = ExcelFormer(
    in_channels=256,  # Number of numerical features
    out_channels=dataset.num_classes,  # Number of target classes
    num_cols=len(dataset.feat_cols),  # Total number of columns
    num_layers=4,  # Customize based on complexity
    num_heads=8,  # Number of attention heads
    col_stats=dataset.col_stats,  # Column statistics
    col_names_dict=dataset.tensor_frame.col_names_dict,  # Column names
    mixup="feature",  # Mixup strategy
).to(device)

# Load model if checkpoint exists
if os.path.exists(model_checkpoint_path):
    print(f"Loading model checkpoint from {model_checkpoint_path}...")
    checkpoint = torch.load(model_checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    print("Model checkpoint loaded.")
else:
    print("No checkpoint found. Starting training from scratch.")



from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
def evaluate_roc_auc(model, loader, device):
    model.eval()  # Set model to evaluation mode
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for tf in loader:
            tf = tf.to(device)
            logits = model(tf)  # Logits are of shape (batch_size, 2)

            # Get probabilities for the positive class (class 1)
            probabilities = F.softmax(logits, dim=1)[:, 1].cpu().numpy()  # Extract class 1 probabilities

            all_probs.extend(probabilities)  # Add probabilities for positive class
            all_labels.extend(tf.y.cpu().numpy())  # Labels should match batch size

    # Ensure number of labels matches number of probabilities
    assert len(all_labels) == len(all_probs), "Mismatch in number of labels and probabilities"
    return roc_auc_score(all_labels, all_probs)



import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

# Loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()  # Use BinaryCrossEntropy for binary classification
optimizer = Adam(model.parameters(), lr=0.001)
lr_scheduler = ExponentialLR(optimizer, gamma=0.95)

# Load optimizer and scheduler state if checkpoint exists
if os.path.exists(model_checkpoint_path):
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
else:
    start_epoch = 0

# Training loop
num_epochs = 100
for epoch in range(start_epoch, num_epochs):
    model.train()
    for tf in train_loader:
        tf = tf.to(device)
        optimizer.zero_grad()
        preds = model(tf)  # Forward pass
        loss = criterion(preds, tf.y)  # Compute loss
        loss.backward()
        optimizer.step()
    lr_scheduler.step()  # Update learning rate

    # Save model checkpoint
    print(f"Saving model checkpoint for epoch {epoch + 1}...")
    # torch.save({
    #     "epoch": epoch,
    #     "model_state_dict": model.state_dict(),
    #     "optimizer_state_dict": optimizer.state_dict(),
    #     "lr_scheduler_state_dict": lr_scheduler.state_dict()
    # }, model_checkpoint_path)

    # Evaluate ROC AUC on validation or test data
    roc_auc = evaluate_roc_auc(model, test_loader, device)
    print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}, ROC AUC: {roc_auc:.4f}")