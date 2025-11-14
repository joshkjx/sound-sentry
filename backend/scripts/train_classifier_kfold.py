import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
from sklearn.metrics import roc_curve
from imblearn.over_sampling import SMOTE
from .utils import (
    DATA_DIR, FEATURES_OUTPUT_FILE, LABELS_OUTPUT_FILE, DEVICE
)
# Prevent terminal clearing on Windows
os.environ['LOKY_MAX_CPU_COUNT'] = '1'
os.environ['JOBLIB_START_METHOD'] = 'spawn'

VERBOSE = False

# Set number of folds
K_FOLDS = 10

# Binary neural network for real/fake classification.
# Same as training_classifier
class BinaryClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, dropout_rate: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(1)


# K-FOLD: Helper function to train a single model (to avoid code repetition)
def train_model(model, criterion, optimizer, features_train, labels_train, features_val, labels_val):
    best_val_loss = float('inf')
    patience = 100  # Stop if no improvement for 100 epochs
    counter = 0

    features_train_tensor = torch.tensor(features_train, dtype=torch.float32)
    labels_train_tensor = torch.tensor(labels_train, dtype=torch.float32)
    features_val_tensor = torch.tensor(features_val, dtype=torch.float32)
    labels_val_tensor = torch.tensor(labels_val, dtype=torch.float32)

    for epoch in range(10000):
        model.train()
        optimizer.zero_grad()
        output = model(features_train_tensor.to(DEVICE))
        loss = criterion(output, labels_train_tensor.to(DEVICE))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_output = model(features_val_tensor.to(DEVICE))
            val_loss = criterion(
                val_output, labels_val_tensor.to(DEVICE)).item()

        if (epoch + 1) % 100 == 0:
            print(
                f"Epoch {epoch+1}: train_loss={loss.item():.6f}, val_loss={val_loss:.4f}")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                if VERBOSE:
                    print(f"Early stopping at epoch {epoch+1}")
                break
    return model


# K-FOLD: Helper function to scale data and apply augmentations
def preprocess_fold_data(features_train, labels_train, features_val):
    # Fit scaler *only* on this fold's training data
    feature_scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    feature_scaler.fit(features_train)

    # Check and fix low-variance features for this fold
    per_feature_std = features_train.std(axis=0)
    low_var_threshold = 0.01
    low_var_mask = per_feature_std < low_var_threshold
    num_low_var = low_var_mask.sum()

    if num_low_var > 0:
        if VERBOSE:
            print(
                f"   Fixing {num_low_var} low-variance features for this fold...")
        if feature_scaler.scale_ is not None:
            feature_scaler.scale_[low_var_mask] = 1.0
        else:
            print("   Error: Scaler not fitted properly")

    # Apply scaling
    features_train_scaled = feature_scaler.transform(features_train)
    # Scale validation data with the *training* scaler
    features_val_scaled = feature_scaler.transform(features_val)

    # Apply SMOTE first (before noise augmentation)
    # This prevents SMOTE from interpolating between artificially similar noisy samples
    smote = SMOTE(random_state=4347)
    resample_result = smote.fit_resample(features_train_scaled, labels_train)
    features_train_balanced, labels_train_balanced = resample_result[0], resample_result[1]

    # Augment with noise AFTER SMOTE
    augmented_features_list = [features_train_balanced]
    augmented_labels_list = [labels_train_balanced]
    for sigma in [0.005, 0.01, 0.02]:
        noise = features_train_balanced + \
            np.random.randn(*features_train_balanced.shape) * sigma
        augmented_features_list.append(noise)
        augmented_labels_list.append(labels_train_balanced)

    features_train_res = np.vstack(augmented_features_list)
    labels_train_res = np.hstack(augmented_labels_list)

    return features_train_res, labels_train_res, features_val_scaled, feature_scaler


# K-FOLD: Helper function to evaluate a trained model
def evaluate_model(model, features_test, labels_test):
    features_test_tensor = torch.tensor(features_test, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        test_output = model(features_test_tensor.to(DEVICE))
        test_prob = torch.sigmoid(test_output).cpu().numpy()
        test_pred = (test_prob > 0.5).astype(int)

    test_acc = accuracy_score(labels_test, test_pred)
    test_auc = roc_auc_score(labels_test, test_prob)
    test_f1 = f1_score(labels_test, test_pred)

    fpr, tpr, thresholds = roc_curve(labels_test, test_prob)
    fnr = 1 - tpr
    eer_threshold_idx = np.nanargmin(np.absolute(fnr - fpr))
    eer = fpr[eer_threshold_idx]
    eer_threshold = thresholds[eer_threshold_idx]

    return test_acc, test_auc, test_f1, eer, eer_threshold


if __name__ == "__main__":
    # Best hyperparameter set
    BEST_LR = 0.0005
    BEST_WEIGHT_DECAY = 1e-4
    BEST_HIDDEN_SIZE = 512
    BEST_DROPOUT_RATE = 0.5

    print("--- Using Hyperparameters ---")
    print(f"Learning Rate: {BEST_LR}")
    print(f"Weight Decay: {BEST_WEIGHT_DECAY}")
    print(f"Hidden Size: {BEST_HIDDEN_SIZE}")
    print(f"Dropout Rate: {BEST_DROPOUT_RATE}")
    print(f"K-Folds: {K_FOLDS}")
    print("-----------------------------")

    # Load all data
    features_path = os.path.join(DATA_DIR, FEATURES_OUTPUT_FILE)
    labels_path = os.path.join(DATA_DIR, LABELS_OUTPUT_FILE)
    features = np.load(features_path)
    labels = np.load(labels_path)

    # Create the outer 80/20 train_val/test split
    # 80% for K-Fold training/validation, 20% for final hold-out test
    features_train_val, features_test_holdout, labels_train_val, labels_test_holdout = train_test_split(
        features, labels, test_size=0.2, stratify=labels, random_state=4347)

    print(f"\nTotal samples: {len(features)}")
    print(
        f"Using {len(features_train_val)} for {K_FOLDS}-Fold Cross-Validation.")
    print(f"Using {len(features_test_holdout)} for final hold-out test set.")

    # Start the K-Fold Validation loop
    print(f"\n--- Starting {K_FOLDS}-Fold Cross-Validation ---")
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=4347)

    fold_val_accs = []
    fold_val_aucs = []
    fold_val_f1s = []
    fold_val_eers = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(features_train_val, labels_train_val)):
        print(f"\n--- Fold {fold+1}/{K_FOLDS} ---")

        # Get data for this fold
        features_train_fold, features_val_fold = features_train_val[
            train_idx], features_train_val[val_idx]
        labels_train_fold, labels_val_fold = labels_train_val[train_idx], labels_train_val[val_idx]

        print(
            f"Train: {len(features_train_fold)}, Val: {len(features_val_fold)}")

        # Preprocess data (Scale, Augment, SMOTE)
        # Scaler is fit *only* on features_train_fold
        features_train_res, labels_train_res, features_val_scaled, _ = \
            preprocess_fold_data(features_train_fold,
                                 labels_train_fold, features_val_fold)

        print(f"Train (after augment/SMOTE): {len(features_train_res)}")

        # Initialize model for this fold
        model_fold = BinaryClassifier(input_dim=features.shape[1],
                                      hidden_size=BEST_HIDDEN_SIZE,
                                      dropout_rate=BEST_DROPOUT_RATE).to(DEVICE)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(
            model_fold.parameters(),
            lr=BEST_LR,
            betas=(0.9, 0.999),
            weight_decay=BEST_WEIGHT_DECAY
        )

        # Train the model
        model_fold = train_model(
            model_fold, criterion, optimizer,
            features_train_res, labels_train_res,  # Train set
            features_val_scaled, labels_val_fold   # Validation set for early stopping
        )

        # Evaluate on this fold's validation set
        val_acc, val_auc, val_f1, val_eer, _ = evaluate_model(
            model_fold, features_val_scaled, labels_val_fold
        )
        print(
            f"Fold {fold+1} Val Metrics: Acc={val_acc:.4f}, AUC={val_auc:.4f}, F1={val_f1:.4f}, EER={val_eer:.4f}")

        # Store metrics
        fold_val_accs.append(val_acc)
        fold_val_aucs.append(val_auc)
        fold_val_f1s.append(val_f1)
        fold_val_eers.append(val_eer)

    # K-Fold Loop Finished
    print("\n--- K-Fold Validation Summary ---")
    print(
        f"Avg Val Accuracy: {np.mean(fold_val_accs):.4f} +/- {np.std(fold_val_accs):.4f}")
    print(
        f"Avg Val AUC:      {np.mean(fold_val_aucs):.4f} +/- {np.std(fold_val_aucs):.4f}")
    print(
        f"Avg Val F1:       {np.mean(fold_val_f1s):.4f} +/- {np.std(fold_val_f1s):.4f}")
    print(
        f"Avg Val EER:      {np.mean(fold_val_eers):.4f} +/- {np.std(fold_val_eers):.4f}")
