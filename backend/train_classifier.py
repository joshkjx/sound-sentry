import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os
from sklearn.metrics import roc_curve
from imblearn.over_sampling import SMOTE
from utils import (
    DATA_DIR, FEATURES_OUTPUT_FILE, LABELS_OUTPUT_FILE,
    SCALER_OUTPUT_FILE, MODEL_OUTPUT_FILE, DEVICE
)
# Prevent terminal clearing on Windows
os.environ['LOKY_MAX_CPU_COUNT'] = '1'
os.environ['JOBLIB_START_METHOD'] = 'spawn'

VERBOSE = False

# Binary neural network for real/fake classification.
# Differences from original DeepSonar:
# - Added dropout (0.3) to each hidden layer for regularisation (paper had no dropout).
# - Uses fixed hidden size (256) x4 instead of decreasing (512-256-128-64).


class BinaryClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int = 256, dropout_rate: float = 0.3):
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


if __name__ == "__main__":
    
    # Best hyperparameter set obtained from gridsearch
    BEST_LR = 0.0005
    BEST_WEIGHT_DECAY = 1e-5
    BEST_HIDDEN_SIZE = 512
    BEST_DROPOUT_RATE = 0.5
    
    print("--- Using Hyperparameters ---")
    print(f"Learning Rate: {BEST_LR}")
    print(f"Weight Decay: {BEST_WEIGHT_DECAY}")
    print(f"Hidden Size: {BEST_HIDDEN_SIZE}")
    print(f"Dropout Rate: {BEST_DROPOUT_RATE}")
    print("-----------------------------")
    # Load data
    features_path = os.path.join(DATA_DIR, FEATURES_OUTPUT_FILE)
    labels_path = os.path.join(DATA_DIR, LABELS_OUTPUT_FILE)
    features = np.load(features_path)
    labels = np.load(labels_path)

    if VERBOSE:
        print("\nLabel distribution:")
        print(pd.Series(labels).value_counts())

    # Split data: 60/20/20 with stratification
    features_train, features_temp, labels_train, labels_temp = train_test_split(
        features, labels, test_size=0.4, stratify=labels, random_state=4347)
    features_val, features_test, labels_val, labels_test = train_test_split(
        features_temp, labels_temp, test_size=0.5, stratify=labels_temp,
        random_state=4347)

    if VERBOSE:
        print(f"\nTrain set: {features_train.shape[0]} samples")
        print(f"Val set: {features_val.shape[0]} samples")
        print(f"Test set: {features_test.shape[0]} samples")

        print(f"Train features before scaling:")
        print(f"  Mean: {features_train.mean():.2f}")
        print(f"  Std: {features_train.std():.2f}")
        print(f"  Min: {features_train.min():.2f}")
        print(f"  Max: {features_train.max():.2f}")

    # Check per-feature variance
    per_feature_std = features_train.std(axis=0)
    if VERBOSE:
        print(f"\nPer-feature std statistics:")
        print(f"  Min: {per_feature_std.min():.6f}")
        print(f"  Max: {per_feature_std.max():.2f}")
        print(f"  Mean: {per_feature_std.mean():.2f}")

    # Identify low-variance features
    low_var_threshold = 0.01  # Features with std < 0.01 are nearly constant
    low_var_mask = per_feature_std < low_var_threshold
    num_low_var = low_var_mask.sum()

    if num_low_var > 0:
        print(
            f"\nWARNING: {num_low_var} features have very low variance (std < {low_var_threshold})")
        print(f"   These features are nearly constant and will be handled specially.")
        print(
            f"   Low-variance feature indices: {np.where(low_var_mask)[0][:10]}...")

    # Fit scaler with explicit parameters
    # Scale features (fits on train only)
    # Differences from original DeepSonar:
    # - Added scaling for better NN performance.
    feature_scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    feature_scaler.fit(features_train)

    # Fix low-variance features by updating their scale to 1.0 to avoid division by tiny numbers
    if num_low_var > 0:
        print(f"\n   Fixing {num_low_var} low-variance features...")
        if feature_scaler.scale_ is not None:
            original_scale = feature_scaler.scale_.copy()
            feature_scaler.scale_[low_var_mask] = 1.0
            print(f"   Before fix - min scale: {original_scale.min():.6f}")
            print(
                f"   After fix - min scale: {feature_scaler.scale_.min():.6f}")
        else:
            print("   Error: Scaler not fitted properly")

    # Verify scaler parameters
    if VERBOSE:
        print(f"\nScaler parameters:")
        if feature_scaler.mean_ is not None:
            print(f"  Mean (avg): {feature_scaler.mean_.mean():.2f}")
        else:
            print("  Mean: None (scaler not fitted)")
        if feature_scaler.scale_ is not None:
            print(f"  Scale (avg): {feature_scaler.scale_.mean():.2f}")
            print(f"  Scale (min): {feature_scaler.scale_.min():.2f}")
            print(f"  Scale (max): {feature_scaler.scale_.max():.2f}")
        else:
            print("  Scale: None (scaler not fitted)")

    # Apply scaling
    features_train_scaled = feature_scaler.transform(features_train)
    features_val = feature_scaler.transform(features_val)
    features_test = feature_scaler.transform(features_test)

    # Verify scaling worked
    if VERBOSE:
        print(f"\nTrain features after scaling:")
        print(f"  Mean: {features_train_scaled.mean():.6f} (should be ~0)")
        print(f"  Std: {features_train_scaled.std():.6f} (should be ~1)")
        print(f"  Min: {features_train_scaled.min():.2f}")
        print(f"  Max: {features_train_scaled.max():.2f}")

    if abs(features_train_scaled.mean()) > 0.01 or abs(features_train_scaled.std() - 1.0) > 0.1:
        print("\nERROR: Scaling verification failed!")
        print("   Features are not properly standardized.")
        exit(1)

    # Save scaler
    scaler_path = os.path.join(DATA_DIR, SCALER_OUTPUT_FILE)
    joblib.dump(feature_scaler, scaler_path)
    print(f"\nScaler saved to {scaler_path}")

    # Augment with noise (doubles training data)
    print("\nAugmenting training data with various noise levels...")

    # Create 3 augmented versions with different noise levels
    augmented_features_list = [features_train_scaled]  # Original
    augmented_labels_list = [labels_train]

    # 1. Noise augmentation (keeps full length)
    print("  Adding noise augmentation...")
    for noise_level, sigma in [('light', 0.005), ('medium', 0.01), ('heavy', 0.02)]:
        noise = features_train_scaled + \
            np.random.randn(*features_train_scaled.shape) * sigma
        augmented_features_list.append(noise)
        augmented_labels_list.append(labels_train)
        print(
            f"    - {noise_level} noise (σ={sigma}): {noise.shape[0]} samples")

    # Combine all
    features_train_augmented = np.vstack(augmented_features_list)
    labels_train_augmented = np.hstack(augmented_labels_list)

    print(
        f"After noise augmentation: {features_train_augmented.shape[0]} samples")
    print(f"  Original: {features_train_scaled.shape[0]}")
    print(f"  Light noise: {features_train_scaled.shape[0]}")
    print(f"  Medium noise: {features_train_scaled.shape[0]}")
    print(f"  Heavy noise: {features_train_scaled.shape[0]}")

    # Classification with imbalanced classes by performing over-sampling.
    # Differences from original DeepSonar:
    # - Apply SMOTE to train data for imbalance
    smote = SMOTE(random_state=4347)
    resample_result = smote.fit_resample(
        features_train_augmented, labels_train_augmented)
    features_train_res, labels_train_res = resample_result[0], resample_result[1]

    if VERBOSE:
        print(f"After SMOTE: {features_train_res.shape[0]} samples")
        print(f"Label distribution after SMOTE:")
        print(pd.Series(np.array(labels_train_res)).value_counts())

    # Convert to tensors
    features_train_tensor = torch.tensor(
        features_train_res, dtype=torch.float32)
    labels_train_tensor = torch.tensor(labels_train_res, dtype=torch.float32)
    features_val_tensor = torch.tensor(features_val, dtype=torch.float32)
    labels_val_tensor = torch.tensor(labels_val, dtype=torch.float32)
    features_test_tensor = torch.tensor(features_test, dtype=torch.float32)
    labels_test_tensor = torch.tensor(labels_test, dtype=torch.float32)

    # Setup model, loss, optimiser
    # Differences from original DeepSonar:
    # - Adam instead of SGD, added weight_decay.
    model = BinaryClassifier(input_dim=features.shape[1]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()  # Handles logits directly
    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001,            # faster learning
        betas=(0.9, 0.999),  # beta1 matches the SGD momentum
        weight_decay=1e-4    # more regularization than L2
    )

    best_val_loss = float('inf')
    patience = 100  # Stop if no improvement for 100 epochs
    counter = 0
    for epoch in range(1000):
        model.train()
        optimizer.zero_grad()
        output = model(features_train_tensor.to(DEVICE))
        loss = criterion(output, labels_train_tensor.to(DEVICE))
        loss.backward()
        # Differences from original DeepSonar:
        # Added clipping.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_output = model(features_val_tensor.to(DEVICE))
            val_prob = torch.sigmoid(val_output).cpu().numpy()
            val_pred = (val_prob > 0.5).astype(int)
            val_loss = criterion(
                val_output, labels_val_tensor.to(DEVICE)).item()
            val_acc = accuracy_score(labels_val, val_pred)
            val_auc = roc_auc_score(labels_val, val_prob)
            val_f1 = f1_score(labels_val, val_pred)

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}: train_loss={loss.item():.6f}, "
                  f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f}, "
                  f"val_auc={val_auc:.3f}, val_f1={val_f1:.3f}")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Test evaluation
    model.eval()
    with torch.no_grad():
        test_output = model(features_test_tensor.to(DEVICE))
        test_prob = torch.sigmoid(test_output).cpu().numpy()
        test_pred = (test_prob > 0.5).astype(int)

    test_acc = accuracy_score(labels_test, test_pred)
    test_auc = roc_auc_score(labels_test, test_prob)
    test_f1 = f1_score(labels_test, test_pred)
    print(f"\nTest Accuracy: {test_acc:.4f}, AUC: {test_auc:.4f}, "
          f"F1: {test_f1:.4f}")

    # EER calculation
    # Differences: Added EER metric.
    fpr, tpr, thresholds = roc_curve(labels_test, test_prob)
    fnr = 1 - tpr
    eer_threshold_idx = np.nanargmin(np.absolute(fnr - fpr))
    eer = fpr[eer_threshold_idx]
    eer_threshold = thresholds[eer_threshold_idx]
    print(f"\nEER: {eer:.4f} at threshold={eer_threshold:.4f}")

    # Save model
    model_path = os.path.join(DATA_DIR, MODEL_OUTPUT_FILE)
    torch.save({
        'model_state_dict': model.state_dict(),
        'threshold': float(eer_threshold),
        'eer': float(eer),
        'test_acc': float(test_acc),
        'test_auc': float(test_auc),
        'test_f1': float(test_f1),
        'input_dim': int(features.shape[1]),
        'hyperparameters': {
            'lr': BEST_LR,
            'weight_decay': BEST_WEIGHT_DECAY,
            'hidden_size': BEST_HIDDEN_SIZE,
            'dropout_rate': BEST_DROPOUT_RATE
        }
    }, model_path)
    print(f"Model and metadata saved to {model_path}.")
