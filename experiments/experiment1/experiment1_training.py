#########################################################################
# SlumberNet-style Sleep Stage Classification using Residual CNNs
# PyTorch rewrite for preprocessed Sleep-EDF .npz data
#########################################################################

import os
import csv
import copy
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

from sklearn.model_selection import KFold
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    confusion_matrix,
    cohen_kappa_score,
    explained_variance_score,
    log_loss,
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =========================================================
# Configuration
# =========================================================

input_directory = r"C:\Users\ryanz\Documents\somneuro\sleepnet\data\preprocessed"
data_path = os.path.join(input_directory, "sleep_edf_all_epochs.npz")
output_directory = os.path.join(r"C:\Users\ryanz\Documents\somneuro\sleepnet\data\output", "k-fold_models_pytorch/")
os.makedirs(output_directory, exist_ok=True)

seed = 154727
num_folds = 5
num_epochs = 50
learning_rate = 1e-4
batch_size = 64
weight_decay = 1e-4
num_workers = 0

augment_data = True
dropout_rate = 0.1

n_resnet_blocks = 7
n_feature_maps = 8
kernel_expansion_fct = 1
kernel_y = 2

use_initial_pool = True
initial_pool_kernel = (4, 1)  # reduces time dimension e.g. 3000 -> 750

CLASS_NAMES = ["W", "N1", "N2", "N3", "REM"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))

# =========================================================
# Reproducibility
# =========================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(seed)

# =========================================================
# Save run parameters
# =========================================================

parameters = {
    "Framework": "PyTorch",
    "Number of folds for k-fold cross-validation": num_folds,
    "Number of Resnet blocks": n_resnet_blocks,
    "Number of epochs": num_epochs,
    "Batch size": batch_size,
    "Learning rate (initial)": learning_rate,
    "Weight decay": weight_decay,
    "Number of feature maps in top layer": n_feature_maps,
    "Kernel y dimension (1 or 2)": kernel_y,
    "Kernel factor size multiple": kernel_expansion_fct,
    "Training data augmented": "yes" if augment_data else "no",
    "Dropout rate": dropout_rate,
    "Initial pooling used": use_initial_pool,
    "Initial pooling kernel": str(initial_pool_kernel),
}
pd.DataFrame(parameters, index=[0]).to_csv(
    os.path.join(output_directory, "run_parameters.csv"),
    index=False
)

# =========================================================
# Load data
# =========================================================

data = np.load(data_path, allow_pickle=True)

# Preprocessing output:
# X shape: (N, C, T)
# y shape: (N,)
# record_id shape: (N,)
X = data["X"].astype(np.float32)
y = data["y"].astype(np.int64)
record_ids = data["record_id"]
channel_names = list(data["channel_names"])
sfreq = float(data["sfreq"][0])

print("Loaded arrays:")
print("  X raw shape:", X.shape)
print("  y shape:", y.shape)
print("  record_ids shape:", record_ids.shape)
print("  channel_names:", channel_names)
print("  sfreq:", sfreq)

n_samples, n_channels, n_timepoints = X.shape
nb_classes = len(np.unique(y))

if nb_classes != len(CLASS_NAMES):
    CLASS_NAMES = [str(i) for i in range(nb_classes)]

print("n_samples:", n_samples)
print("n_channels:", n_channels)
print("n_timepoints:", n_timepoints)
print("nb_classes:", nb_classes)

# =========================================================
# Dataset / augmentation
# =========================================================

class SleepDataset(Dataset):
    """
    Expects X in shape (N, C, T).
    Returns tensors in shape (1, T, C) for Conv2d:
      PyTorch Conv2d expects (B, in_channels, H, W)
      Here:
        in_channels = 1
        H = time
        W = channels
    """

    def __init__(self, X, y, augment=False):
        self.X = X
        self.y = y
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def _augment(self, x):
        # x shape: (C, T)

        # Mild amplitude scaling per channel
        amp = np.random.uniform(0.9, 1.1, size=(x.shape[0], 1)).astype(np.float32)
        x = x * amp

        # Small temporal shift, applied jointly across channels
        shift = np.random.randint(-200, 201)
        x = np.roll(x, shift=shift, axis=1)

        return x

    def __getitem__(self, idx):
        x = self.X[idx].copy()  # (C, T)
        target = self.y[idx]

        if self.augment:
            x = self._augment(x)

        # Convert (C, T) -> (T, C)
        x = np.transpose(x, (1, 0))  # (T, C)

        # Add PyTorch conv channel dim -> (1, T, C)
        x = np.expand_dims(x, axis=0)

        return torch.tensor(x, dtype=torch.float32), torch.tensor(target, dtype=torch.long)

# =========================================================
# Model
# =========================================================

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_y=2, kernel_expansion_fct=1, dropout_rate=0.0):
        super().__init__()

        self.conv_x = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kernel_y, 8 * kernel_expansion_fct),
            padding="same",
        )
        self.bn_x = nn.BatchNorm2d(out_channels)

        self.conv_y = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(kernel_y, 5 * kernel_expansion_fct),
            padding="same",
        )
        self.bn_y = nn.BatchNorm2d(out_channels)

        self.conv_z = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(kernel_y, 3 * kernel_expansion_fct),
            padding="same",
        )
        self.bn_z = nn.BatchNorm2d(out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), padding="same"),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.BatchNorm2d(out_channels)

        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x

        out = self.conv_x(x)
        out = self.bn_x(out)
        out = self.dropout(out)
        out = self.relu(out)

        out = self.conv_y(out)
        out = self.bn_y(out)
        out = self.dropout(out)
        out = self.relu(out)

        out = self.conv_z(out)
        out = self.bn_z(out)

        identity = self.shortcut(identity)
        out = out + identity
        out = self.dropout(out)
        out = self.relu(out)
        return out


class FinalResidualBlock(nn.Module):
    def __init__(self, channels, kernel_y=2, kernel_expansion_fct=1, dropout_rate=0.0):
        super().__init__()

        self.conv_x = nn.Conv2d(
            channels, channels,
            kernel_size=(kernel_y, 8 * kernel_expansion_fct),
            padding="same"
        )
        self.bn_x = nn.BatchNorm2d(channels)

        self.conv_y = nn.Conv2d(
            channels, channels,
            kernel_size=(kernel_y, 8 * kernel_expansion_fct),
            padding="same"
        )
        self.bn_y = nn.BatchNorm2d(channels)

        self.conv_z = nn.Conv2d(
            channels, channels,
            kernel_size=(kernel_y, 8 * kernel_expansion_fct),
            padding="same"
        )
        self.bn_z = nn.BatchNorm2d(channels)

        self.shortcut_bn = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = self.shortcut_bn(x)

        out = self.conv_x(x)
        out = self.bn_x(out)
        out = self.relu(out)

        out = self.conv_y(out)
        out = self.bn_y(out)
        out = self.relu(out)

        out = self.conv_z(out)
        out = self.bn_z(out)

        out = out + identity
        out = self.dropout(out)
        out = self.relu(out)
        return out


class SlumberNetPT(nn.Module):
    def __init__(
        self,
        nb_classes,
        n_resnet_blocks=7,
        n_feature_maps=8,
        kernel_y=2,
        kernel_expansion_fct=1,
        dropout_rate=0.0,
        use_initial_pool=True,
        initial_pool_kernel=(4, 1),
    ):
        super().__init__()

        self.use_initial_pool = use_initial_pool
        if self.use_initial_pool:
            self.initial_pool = nn.AvgPool2d(kernel_size=initial_pool_kernel)

        blocks = []
        in_channels = 1

        # Repeated residual blocks with channel expansion
        for i in range(n_resnet_blocks - 1):
            out_channels = n_feature_maps * (2 ** i)
            blocks.append(
                ResidualBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_y=kernel_y,
                    kernel_expansion_fct=kernel_expansion_fct,
                    dropout_rate=dropout_rate,
                )
            )
            in_channels = out_channels

        self.res_blocks = nn.Sequential(*blocks)

        self.final_block = FinalResidualBlock(
            channels=in_channels,
            kernel_y=kernel_y,
            kernel_expansion_fct=kernel_expansion_fct,
            dropout_rate=dropout_rate,
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels, nb_classes)

    def forward(self, x):
        # x shape: (B, 1, T, C)
        if self.use_initial_pool:
            x = self.initial_pool(x)

        x = self.res_blocks(x)
        x = self.final_block(x)

        x = self.gap(x)             # (B, F, 1, 1)
        x = torch.flatten(x, 1)     # (B, F)
        x = self.fc(x)              # (B, nb_classes)
        return x

# =========================================================
# Training / evaluation helpers
# =========================================================

def train_one_epoch(model, loader, criterion, optimizer, device, epoch_num=None, num_epochs=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar_desc = f"Train Epoch {epoch_num}/{num_epochs}" if epoch_num is not None else "Train"
    pbar = tqdm(loader, desc=pbar_desc, leave=False)

    for xb, yb in pbar:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * xb.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)

        current_loss = running_loss / total
        current_acc = correct / total
        pbar.set_postfix(loss=f"{current_loss:.4f}", acc=f"{current_acc:.4f}")

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

@torch.no_grad()
def evaluate(model, loader, criterion, device, epoch_num=None, num_epochs=None, split_name="Val"):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_targets = []
    all_preds = []
    all_probs = []

    pbar_desc = f"{split_name} Epoch {epoch_num}/{num_epochs}" if epoch_num is not None else split_name
    pbar = tqdm(loader, desc=pbar_desc, leave=False)

    for xb, yb in pbar:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        logits = model(xb)
        loss = criterion(logits, yb)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        running_loss += loss.item() * xb.size(0)
        correct += (preds == yb).sum().item()
        total += yb.size(0)

        all_targets.append(yb.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

        current_loss = running_loss / total
        current_acc = correct / total
        pbar.set_postfix(loss=f"{current_loss:.4f}", acc=f"{current_acc:.4f}")

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)

    return epoch_loss, epoch_acc, y_true, y_pred, y_prob

def fit_fold(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs, checkpoint_path):
    history = {
        "loss": [],
        "accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    best_val_acc = -np.inf
    best_state = None
    epochs_without_improvement = 0
    early_stop_patience = 8

    epoch_bar = tqdm(range(num_epochs), desc="Epochs", leave=True)

    for epoch in epoch_bar:
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch_num=epoch + 1, num_epochs=num_epochs
        )

        val_loss, val_acc, _, _, _ = evaluate(
            model, val_loader, criterion, device,
            epoch_num=epoch + 1, num_epochs=num_epochs, split_name="Val"
        )

        history["loss"].append(train_loss)
        history["accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        if scheduler is not None:
            scheduler.step(val_loss)

        epoch_bar.set_postfix(
            train_loss=f"{train_loss:.4f}",
            train_acc=f"{train_acc:.4f}",
            val_loss=f"{val_loss:.4f}",
            val_acc=f"{val_acc:.4f}",
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, checkpoint_path)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= early_stop_patience:
            print("Early stopping triggered.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history

def save_history_plots(history, output_directory, fold_num):
    plt.figure()
    plt.plot(history["accuracy"])
    plt.plot(history["val_accuracy"])
    plt.title("Model accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"], loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, f"fold_{fold_num}_accuracy_over_epoch.png"))
    plt.savefig(os.path.join(output_directory, f"fold_{fold_num}_accuracy_over_epoch.pdf"))
    plt.show()
    plt.close()

    plt.figure()
    plt.plot(history["loss"])
    plt.plot(history["val_loss"])
    plt.title("Model loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"], loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, f"fold_{fold_num}_loss_over_epoch.png"))
    plt.savefig(os.path.join(output_directory, f"fold_{fold_num}_loss_over_epoch.pdf"))
    plt.show()
    plt.close()


def save_confusion_matrix_plot(conf_matrix, class_names, output_directory, fold_num):
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_matrix,
        annot=True,
        cmap="YlGnBu",
        fmt="g",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(f"Confusion Matrix for Fold {fold_num}")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, f"fold_{fold_num}_confusion_matrix.png"))
    plt.savefig(os.path.join(output_directory, f"fold_{fold_num}_confusion_matrix.pdf"))
    plt.show()
    plt.close()

# =========================================================
# Record-wise k-fold cross-validation
# =========================================================

precisions, recalls, f1_scores, supports = [], [], [], []
accuracies, kappas, losses, explained_variances, confusion_matrices = [], [], [], [], []

unique_records = np.unique(record_ids)
print(f"Unique recordings: {len(unique_records)}")

if num_folds > len(unique_records):
    raise ValueError(
        f"num_folds={num_folds} is greater than number of unique recordings={len(unique_records)}"
    )

kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)

fold_iterator = list(enumerate(kf.split(unique_records), start=1))

for fold_num, (train_record_idx, test_record_idx) in tqdm(
    fold_iterator,
    total=num_folds,
    desc="Cross-validation folds"
):
    print(f"\n========== Fold {fold_num} ==========")

    train_records = unique_records[train_record_idx]
    test_records = unique_records[test_record_idx]

    train_mask = np.isin(record_ids, train_records)
    test_mask = np.isin(record_ids, test_records)

    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]

    print(
        f"Fold {fold_num}: "
        f"train_records={len(train_records)}, test_records={len(test_records)}, "
        f"X_train={X_train.shape}, y_train={y_train.shape}, "
        f"X_test={X_test.shape}, y_test={y_test.shape}"
    )

    train_dataset = SleepDataset(X_train, y_train, augment=augment_data)
    test_dataset = SleepDataset(X_test, y_test, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = SlumberNetPT(
        nb_classes=nb_classes,
        n_resnet_blocks=n_resnet_blocks,
        n_feature_maps=n_feature_maps,
        kernel_y=kernel_y,
        kernel_expansion_fct=kernel_expansion_fct,
        dropout_rate=dropout_rate,
        use_initial_pool=use_initial_pool,
        initial_pool_kernel=initial_pool_kernel,
    ).to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        min_lr=1e-7,
    )

    best_model_filepath = os.path.join(output_directory, f"fold_{fold_num}_model_best.pt")

    model, history = fit_fold(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=num_epochs,
        checkpoint_path=best_model_filepath,
    )

    # Save final model
    torch.save(model.state_dict(), os.path.join(output_directory, f"fold_{fold_num}_model_last.pt"))

    # Save history
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(output_directory, f"fold_{fold_num}_history.csv"), index=False)

    # Evaluate best model
    test_loss, test_acc, y_true, y_pred, y_prob = evaluate(model, test_loader, criterion, device, split_name="Test")
    # Metrics
    precision, recall, f1_score, support = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred, labels=np.arange(nb_classes))
    kappa = cohen_kappa_score(y_true, y_pred)
    y_true_one_hot = np.eye(nb_classes)[y_true]
    fold_log_loss = log_loss(y_true_one_hot, y_prob, labels=np.arange(nb_classes))
    exp_var = explained_variance_score(y_true, y_pred)

    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1_score)
    supports.append(support)
    accuracies.append(acc)
    kappas.append(kappa)
    losses.append(fold_log_loss)
    explained_variances.append(exp_var)
    confusion_matrices.append(conf_matrix)

    print(
        f"Fold {fold_num} results | "
        f"loss={test_loss:.4f} acc={acc:.4f} "
        f"precision={precision:.4f} recall={recall:.4f} "
        f"f1={f1_score:.4f} kappa={kappa:.4f}"
    )

    save_confusion_matrix_plot(conf_matrix, CLASS_NAMES, output_directory, fold_num)
    save_history_plots(history, output_directory, fold_num)

# =========================================================
# Save aggregate results
# =========================================================

metrics = {
    "Precision": precisions,
    "Recall": recalls,
    "F1-Score": f1_scores,
    "Support": supports,
    "Accuracy": accuracies,
    "Cohen_Kappa": kappas,
    "Log_Loss": losses,
    "Explained_Variance": explained_variances,
}

metrics_results_df = pd.DataFrame(metrics)
metrics_results_df.to_csv(os.path.join(output_directory, "kfold_metrics.csv"), index=False)

for i, matrix in enumerate(confusion_matrices, start=1):
    with open(os.path.join(output_directory, f"confusion_matrix_fold_{i}.csv"), mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        for row in matrix:
            writer.writerow(row)

summary_df = pd.DataFrame({
    "metric": ["Precision", "Recall", "F1-Score", "Accuracy", "Cohen_Kappa", "Log_Loss", "Explained_Variance"],
    "mean": [
        np.mean(precisions),
        np.mean(recalls),
        np.mean(f1_scores),
        np.mean(accuracies),
        np.mean(kappas),
        np.mean(losses),
        np.mean(explained_variances),
    ],
    "std": [
        np.std(precisions),
        np.std(recalls),
        np.std(f1_scores),
        np.std(accuracies),
        np.std(kappas),
        np.std(losses),
        np.std(explained_variances),
    ]
})
summary_df.to_csv(os.path.join(output_directory, "kfold_metric_summary.csv"), index=False)

print("\nCross-validation complete.")
print(summary_df)