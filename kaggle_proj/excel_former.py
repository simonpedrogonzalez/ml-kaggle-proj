import torch
from excelformer_dataset import loaders, datasets
from excel_former_model import get_best_validation_model, get_train_model, save_train_model, save_best_validation_model
import tqdm
import torch.nn.functional as F
from torchmetrics import AUROC

train_dataset, test_dataset, val_dataset = datasets()
train_loader, test_loader, val_loader = loaders()
model, criterion, optimizer, lr_scheduler, start_epoch, device, metrics = get_train_model(test_dataset)
best_model = get_best_validation_model(test_dataset)
best_metrics = best_model[-1]
if metrics is None:
    best_val_auc_roc = 0
else:
    best_val_auc_roc = best_metrics['val_roc_auc']
num_epochs = 300
use_mixup = False  # Toggle mixup as needed

def test(loader):
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
    
    # print('withsklearn', evaluate_roc_auc(model, loader, device))

    # Compute and return the final AUROC
    return metric.compute().item()

def report(epoch, metrics):
    print(f"Epoch {epoch}, Loss: {metrics['loss']:.4f}, Test AUROC: {metrics['test_roc_auc']:.4f}, Val AUROC: {metrics['val_roc_auc']:.4f}")

def train(use_mixup=True, epoch=None):
   
    model.train()  # Set model to training mode
    loss_accum = total_count = 0  # Initialize accumulated loss and sample count

    for tf in tqdm.tqdm(train_loader, desc=f"Epoch {epoch}"):
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

        # print('loss', loss)
        # Accumulate loss and sample count
        loss_accum += float(loss) * len(tf.y)
        total_count += len(tf.y)

    # Calculate average loss
    avg_loss = loss_accum / total_count
    return avg_loss

for epoch in range(start_epoch, num_epochs):
    # Train for one epoch
    avg_loss = train(use_mixup=use_mixup, epoch=epoch)

    # Step the learning rate scheduler
    lr_scheduler.step()

    # Evaluate metrics
    test_roc_auc = test(test_loader)
    val_roc_auc = test(val_loader)

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