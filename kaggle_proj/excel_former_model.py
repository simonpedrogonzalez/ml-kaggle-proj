import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch_frame.nn.models import ExcelFormer
import os
import torch


def get_model(model_checkpoint_path, test_dataset):
    # Paths for saving and loading model checkpoints

    # Device setup (use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize ExcelFormer
    model = ExcelFormer(
        in_channels=48,  # Number of numerical features
        out_channels=test_dataset.num_classes,  # Number of target classes
        num_cols=len(test_dataset.feat_cols),  # Total number of columns
        num_layers=2,  # Customize based on complexity
        num_heads=4,  # Number of attention heads
        col_stats=test_dataset.col_stats,  # Column statistics
        col_names_dict=test_dataset.tensor_frame.col_names_dict,  # Column names
        mixup="feature",  # Mixup strategy
        residual_dropout=0.,
        diam_dropout=0.3,
        aium_dropout=0.,
    ).to(device)

    # Load model if checkpoint exists
    if os.path.exists(model_checkpoint_path):
        print(f"Loading model checkpoint from {model_checkpoint_path}...")
        checkpoint = torch.load(model_checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        model_epoch = checkpoint["epoch"]
        print("Model checkpoint loaded from epoch", model_epoch)
        print("Metrics:", checkpoint["metrics"])
    else:
        print("No checkpoint found. Starting training from scratch.")

    # import torch.nn.functional as F
    # from torch.optim import Adam
    # from torch.optim.lr_scheduler import ExponentialLR

    # Loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()  # Use BinaryCrossEntropy for binary classification
    optimizer = Adam(model.parameters(), lr=0.001)
    lr_scheduler = ExponentialLR(optimizer, gamma=0.95)

    # Load optimizer and scheduler state if checkpoint exists
    if os.path.exists(model_checkpoint_path):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        metrics = checkpoint["metrics"]
    else:
        start_epoch = 0
        metrics = None
    
    return model, criterion, optimizer, lr_scheduler, start_epoch, device, metrics

def save_model(model_checkpoint_path, epoch, model, optimizer, lr_scheduler, metrics):
    print(f"Saving model checkpoint for epoch {epoch}...")
    torch.save({
        "epoch": epoch,
        "metrics": metrics,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state_dict": lr_scheduler.state_dict(),
    }, model_checkpoint_path)

train_checkpoint_path = "excelformer_train_checkpoint.pth"
best_val_checkpoint_path = "excelformer_best_val_checkpoint.pth"

def report(epoch, metrics):
    print(f"Epoch {epoch}, Loss: {metrics['loss']:.4f}, Test AUROC: {metrics['test_roc_auc']:.4f}, Val AUROC: {metrics['val_roc_auc']:.4f}")


def get_best_validation_model(dataset):
    print(f"Loading best validation model...")
    model, criterion, optimizer, lr_scheduler, start_epoch, device, metrics = get_model(
        best_val_checkpoint_path,
        dataset)
    if metrics is None:
        print("No best validation model found.")
    else:
        print("Best validation model loaded.")
        report(start_epoch, metrics)
    return model, criterion, optimizer, lr_scheduler, start_epoch, device, metrics

def get_train_model(dataset):
    print(f"Loading training model...")
    model, criterion, optimizer, lr_scheduler, start_epoch, device, metrics = get_model(
        train_checkpoint_path,
        dataset)
    if metrics is None:
        print("No training model found.")
    else:
        print("Training model loaded.")
        report(start_epoch, metrics)
    return model, criterion, optimizer, lr_scheduler, start_epoch, device, metrics

def save_best_validation_model(epoch, model, optimizer, lr_scheduler, metrics):
    print(f"Saving best validation model: Validation AUROC: {metrics['val_roc_auc']:.4f}")
    save_model(best_val_checkpoint_path, epoch, model, optimizer, lr_scheduler, metrics)

def save_train_model(epoch, model, optimizer, lr_scheduler, metrics):
    save_model(train_checkpoint_path, epoch, model, optimizer, lr_scheduler, metrics)

