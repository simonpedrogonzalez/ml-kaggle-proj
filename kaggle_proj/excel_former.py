# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from pre3_feature_eng import pre3_feature_eng_as_dummies
# # Load data
# all_data, new_train, new_predict = pre3_feature_eng_as_dummies()
# # Split data
# X = new_train.drop('income>50K', axis=1)
# y = new_train['income>50K']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
# # Numerical features
# from torch_frame.data import Dataset, DataLoader
# import torch_frame
# # iter all non float64 columns and convert from bool to int
# for col in X_train.columns:
#     if X_train[col].dtype == 'bool':
#         X_train[col] = X_train[col].astype(int)
#         X_test[col] = X_test[col].astype(int)
# # Map all features (numerical + dummy encoded) to `stype.numerical`
# col_to_stype = {
#     **{col: torch_frame.numerical for col in X_train.columns},  # All columns in X_train are numerical
#     "target": torch_frame.categorical  # Specify the target column as categorical
# }


# # Set up the dataset with semantic types and target column
# train_dataset = Dataset(X_train.assign(target=y_train), col_to_stype=col_to_stype, target_col="target")
# # Materialize the dataset for PyTorch Frame
# train_dataset.materialize()
# test_dataset = Dataset(X_test.assign(target=y_test), col_to_stype=col_to_stype, target_col="target")
# test_dataset.materialize()


# from torch_frame.transforms import MutualInformationSort
# from torch_frame.typing import TaskType
# mutual_info_sort = MutualInformationSort(task_type=TaskType.BINARY_CLASSIFICATION)  # Adjust task_type as needed
# mutual_info_sort.fit(train_dataset.tensor_frame, train_dataset.col_stats)
# # train_dataset.tensor_frame = mutual_info_sort(train_dataset.tensor_frame)
# sorted_tensor_frame = mutual_info_sort(train_dataset.tensor_frame)

# # Sort features in the training tensor frame
# sorted_train_tensor_frame = mutual_info_sort(train_dataset.tensor_frame)

# # Use the original test tensor frame without sorting
# test_tensor_frame = test_dataset.tensor_frame

# batch_size = 512

# # Create DataLoaders
# train_loader = DataLoader(sorted_train_tensor_frame, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(test_tensor_frame, batch_size=batch_size, shuffle=False)

# import os
# from torch_frame.nn.models import ExcelFormer
import torch
# from torch_frame import stype
from excelformer_dataset import loaders, datasets
from excel_former_model import get_best_validation_model, get_train_model, save_train_model, save_best_validation_model
import tqdm



# # Initialize ExcelFormer
# model = ExcelFormer(
#     in_channels=256,  # Number of numerical features
#     out_channels=test_dataset.num_classes,  # Number of target classes
#     num_cols=len(test_dataset.feat_cols),  # Total number of columns
#     num_layers=5,  # Customize based on complexity
#     num_heads=4,  # Number of attention heads
#     col_stats=test_dataset.col_stats,  # Column statistics
#     col_names_dict=test_dataset.tensor_frame.col_names_dict,  # Column names
#     mixup="feature",  # Mixup strategy
#     residual_dropout=0.,
#     diam_dropout=0.3,
#     aium_dropout=0.,
# ).to(device)

# from sklearn.metrics import roc_auc_score
import torch.nn.functional as F

# def evaluate_roc_auc(model, loader, device):
#     model.eval()  # Set model to evaluation mode
#     all_labels = []
#     all_probs = []

#     with torch.no_grad():
#         for tf in loader:
#             tf = tf.to(device)
#             logits = model(tf)  # Logits are of shape (batch_size, 2)

#             # Get probabilities for the positive class (class 1)
#             probabilities = F.softmax(logits, dim=1)[:, 1].cpu().numpy()  # Extract class 1 probabilities

#             all_probs.extend(probabilities)  # Add probabilities for positive class
#             all_labels.extend(tf.y.cpu().numpy())  # Labels should match batch size

#     # Ensure number of labels matches number of probabilities
#     assert len(all_labels) == len(all_probs), "Mismatch in number of labels and probabilities"
#     return roc_auc_score(all_labels, all_probs)


from torchmetrics import AUROC

def test(model, loader, device):
    model.eval()  # Set model to evaluation mode
    metric = AUROC(task="binary").to(device)  # Initialize TorchMetrics AUROC for binary classification

    with torch.no_grad():
        for tf in loader:
            tf = tf.to(device)
            logits = model(tf)  # Logits are of shape (batch_size, 2)

            # Get probabilities for the positive class (class 1)
            probabilities = F.softmax(logits, dim=1)[:, 1]  # Extract class 1 probabilities

            # Update the AUROC metric
            metric.update(probabilities, tf.y)

    # Compute and return the final AUROC
    return metric.compute().item()





# # Load model if checkpoint exists
# if os.path.exists(model_checkpoint_path):
#     print(f"Loading model checkpoint from {model_checkpoint_path}...")
#     checkpoint = torch.load(model_checkpoint_path)
#     model.load_state_dict(checkpoint["model_state_dict"])
#     print("Model checkpoint loaded.")
#     print("Evaluating loaded model...")
#     roc_auc = evaluate_roc_auc(model, test_loader, device)
#     model_epoch = checkpoint["epoch"]
#     print(f"Epoch {model_epoch}, ROC AUC: {roc_auc:.4f}")
# else:
#     print("No checkpoint found. Starting training from scratch.")







# import torch.nn.functional as F
# from torch.optim import Adam
# from torch.optim.lr_scheduler import ExponentialLR

# # Loss function and optimizer
# criterion = torch.nn.CrossEntropyLoss()  # Use BinaryCrossEntropy for binary classification
# optimizer = Adam(model.parameters(), lr=0.001)
# lr_scheduler = ExponentialLR(optimizer, gamma=0.95)

# # Load optimizer and scheduler state if checkpoint exists
# if os.path.exists(model_checkpoint_path):
#     optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
#     lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
#     start_epoch = checkpoint["epoch"] + 1
# else:
#     start_epoch = 0

def report(epoch, metrics):
    print(f"Epoch {epoch}, Loss: {metrics['loss']:.4f}, Test AUROC: {metrics['test_roc_auc']:.4f}, Val AUROC: {metrics['val_roc_auc']:.4f}")


def train(model, loader, device, optimizer, criterion, use_mixup=True, epoch=None):
   
    model.train()  # Set model to training mode
    loss_accum = total_count = 0  # Initialize accumulated loss and sample count

    for tf in tqdm.tqdm(loader, desc=f"Epoch {epoch}"):
        tf = tf.to(device)
        optimizer.zero_grad()

        if use_mixup:
            # Use mixup for data augmentation
            preds, y_mixedup = model(tf, mixup_encoded=True)
            loss = criterion(preds, y_mixedup)
        else:
            # Standard training without mixup
            preds = model(tf)
            loss = criterion(preds, tf.y)

        loss.backward()
        optimizer.step()

        # Accumulate loss and sample count
        loss_accum += float(loss) * len(tf.y)
        total_count += len(tf.y)

    # Calculate average loss
    avg_loss = loss_accum / total_count
    return avg_loss


train_dataset, test_dataset, val_dataset = datasets()
train_loader, test_loader, val_loader = loaders()
model, criterion, optimizer, lr_scheduler, start_epoch, device, metrics = get_train_model(train_dataset)
best_model = get_best_validation_model(val_dataset)
best_metrics = best_model[-1]
if metrics is None:
    best_val_auc_roc = 0
else:
    best_val_auc_roc = best_metrics['val_roc_auc']

# Paths for saving and loading model checkpoints
# model_checkpoint_path = "excelformer_checkpoint.pth"

num_epochs = 300
use_mixup = True  # Toggle mixup as needed

for epoch in range(start_epoch, num_epochs):
    # Train for one epoch
    avg_loss = train(model, train_loader, device, optimizer, criterion, use_mixup=use_mixup, epoch=epoch)

    # Step the learning rate scheduler
    lr_scheduler.step()

    # Evaluate metrics
    test_roc_auc = test(model, test_loader, device)
    val_roc_auc = test(model, val_loader, device)

    metrics = {
        "test_roc_auc": test_roc_auc,
        "val_roc_auc": val_roc_auc,
        "loss": avg_loss
    }

    report(epoch, metrics)

    # Save model checkpoint
    save_train_model(epoch, model, optimizer, lr_scheduler, metrics)
    if val_roc_auc > best_val_auc_roc:
        save_best_validation_model(epoch, model, optimizer, lr_scheduler, metrics)
        best_val_auc_roc = val_roc_auc