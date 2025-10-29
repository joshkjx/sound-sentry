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
    SCALER_OUTPUT_FILE, DEVICE
)
import itertools
import copy
from torch.utils.data import DataLoader, TensorDataset
# Prevent terminal clearing on Windows
os.environ['LOKY_MAX_CPU_COUNT'] = '1'
os.environ['JOBLIB_START_METHOD'] = 'spawn'

VERBOSE = False
VERBOSE_TRAINING = False

from train_classifier import BinaryClassifier
    

def run_trial(params: dict, data_tensors: dict, input_dim: int) -> tuple:
    """
    Runs a single training trial with the given hyperparameters.
    
    Returns:
        (best_val_auc, best_model_state)
    """

    # Unpack data tensors
    features_train_tensor = data_tensors['features_train_tensor'].to(DEVICE)
    labels_train_tensor = data_tensors['labels_train_tensor'].to(DEVICE)
    features_val_tensor = data_tensors['features_val_tensor'].to(DEVICE)
    labels_val_tensor = data_tensors['labels_val_tensor'].to(DEVICE)

    # Setup model, loss, optimiser from params
    model = BinaryClassifier(
        input_dim=input_dim,
        hidden_size=params['hidden_size'],
        dropout_rate=params['dropout_rate']
    ).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=params['lr'],
        weight_decay=params['weight_decay'],
        betas=(0.9, 0.999)
    )

    best_val_loss = float('inf')
    best_val_auc = -1.0
    best_model_state = None

    patience = 100
    counter = 0

    for epoch in range(1000):
        model.train()
        optimizer.zero_grad()
        output = model(features_train_tensor)
        loss = criterion(output, labels_train_tensor)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_output = model(features_val_tensor)
            val_prob = torch.sigmoid(val_output).cpu().numpy()
            val_pred = (val_prob > 0.5).astype(int)
            val_loss = criterion(val_output, labels_val_tensor).item()
            val_acc = accuracy_score(labels_val_tensor.cpu(), val_pred)
            val_auc = roc_auc_score(labels_val_tensor.cpu(), val_prob)
            val_f1 = f1_score(labels_val_tensor.cpu(), val_pred)

        # Early stopping check (based on your original logic)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_auc = val_auc  # <-- Store the AUC when loss is best
            # <-- Save the best model weights
            best_model_state = copy.deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                # print(f"  Early stopping at epoch {epoch+1}")
                break

    # print(f"  -> Trial Result: Best Val AUC = {best_val_auc:.4f} (at Val Loss = {best_val_loss:.4f})")
    return best_val_auc, best_model_state

if __name__ == "__main__":
    # Load data
    features_path = os.path.join(DATA_DIR, FEATURES_OUTPUT_FILE)
    labels_path = os.path.join(DATA_DIR, LABELS_OUTPUT_FILE)
    features = np.load(features_path)
    labels = np.load(labels_path)
    input_dim = features.shape[1]

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
        print(f"\nWARNING: {num_low_var} features have very low variance (std < {low_var_threshold})")
        print(f"   These features are nearly constant and will be handled specially.")
        print(f"   Low-variance feature indices: {np.where(low_var_mask)[0][:10]}...")
    
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
            print(f"   After fix - min scale: {feature_scaler.scale_.min():.6f}")
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
        noise = features_train_scaled + np.random.randn(*features_train_scaled.shape) * sigma
        augmented_features_list.append(noise)
        augmented_labels_list.append(labels_train)
        print(f"    - {noise_level} noise (σ={sigma}): {noise.shape[0]} samples")

    # Combine all
    features_train_augmented = np.vstack(augmented_features_list)
    labels_train_augmented = np.hstack(augmented_labels_list)

    print(f"After noise augmentation: {features_train_augmented.shape[0]} samples")
    print(f"  Original: {features_train_scaled.shape[0]}")
    print(f"  Light noise: {features_train_scaled.shape[0]}")
    print(f"  Medium noise: {features_train_scaled.shape[0]}")
    print(f"  Heavy noise: {features_train_scaled.shape[0]}")

    # Classification with imbalanced classes by performing over-sampling.
    # Differences from original DeepSonar:
    # - Apply SMOTE to train data for imbalance 
    smote = SMOTE(random_state=4347)
    resample_result = smote.fit_resample(features_train_augmented, labels_train_augmented)
    features_train_res, labels_train_res = resample_result[0], resample_result[1]

    if VERBOSE:
        print(f"After SMOTE: {features_train_res.shape[0]} samples")
        print(f"Label distribution after SMOTE:")
        print(pd.Series(np.array(labels_train_res)).value_counts())

    # Convert to tensors
    data_tensors = {
        "features_train_tensor": torch.tensor(
            features_train_res, dtype=torch.float32),
        "labels_train_tensor": torch.tensor(
            labels_train_res, dtype=torch.float32),
        "features_val_tensor": torch.tensor(features_val, dtype=torch.float32),
        "labels_val_tensor": torch.tensor(labels_val, dtype=torch.float32),
        "features_test_tensor": torch.tensor(features_test, dtype=torch.float32),
        "labels_test_tensor": torch.tensor(labels_test, dtype=torch.float32),
    }
    
    
    
    PARAM_GRID = {
        'lr': [0.001, 0.0005],
        'weight_decay': [1e-5, 5e-6],
        'hidden_size': [256, 512],
        'dropout_rate': [0.3, 0.5]
    }

    keys, values = zip(*PARAM_GRID.items())
    param_combinations = [dict(zip(keys, v))
                          for v in itertools.product(*values)]

    best_overall_val_auc = -1.0
    best_params = {}

    print(f"\n--- Starting Grid Search ({len(param_combinations)} trials) ---")

    for i, params in enumerate(param_combinations):
        print(f"[Trial {i+1}/{len(param_combinations)}] Params: {params}")

        val_auc, model_state = run_trial(params, data_tensors, input_dim)

        print(f"  -> Trial Result: Best Val AUC = {val_auc:.4f}")

        if val_auc > best_overall_val_auc:
            best_overall_val_auc = val_auc
            best_params = params
            print(f"  *** New Best Trial Found! ***")

    # --- Final Output ---
    print(f"\n--- Grid Search Complete ---")
    print(f"Best Validation AUC: {best_overall_val_auc:.4f}")

    print("\n--- 📋 Best Hyperparameter Set ---")
    print("Copy these values into train_classifier.py:")
    print(f"\n{best_params}\n")
    print("-----------------------------------")
