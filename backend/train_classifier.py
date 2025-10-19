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
from utils import (
    DATA_DIR, FEATURES_OUTPUT_FILE, LABELS_OUTPUT_FILE, 
    SCALER_OUTPUT_FILE, MODEL_OUTPUT_FILE, DEVICE
)

# Binary neural network for real/fake classification.
# Differences from original DeepSonar:
# - Added dropout (0.3) to each hidden layer for regularisation (paper had no dropout).
# - Uses fixed hidden size (256) x4 instead of decreasing (512-256-128-64).
class BinaryClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(1)

if __name__ == "__main__":
    # Load data
    features_path = os.path.join(DATA_DIR, FEATURES_OUTPUT_FILE)
    labels_path = os.path.join(DATA_DIR, LABELS_OUTPUT_FILE)
    features = np.load(features_path)
    labels = np.load(labels_path)

    print("Label distribution:")
    print(pd.Series(labels).value_counts())

    # Split data: 60/20/20 with stratification
    features_train, features_temp, labels_train, labels_temp = train_test_split(
        features, labels, test_size=0.4, stratify=labels, random_state=4347)
    features_val, features_test, labels_val, labels_test = train_test_split(
        features_temp, labels_temp, test_size=0.5, stratify=labels_temp,
        random_state=4347)
    
    # Scale features (fits on train only)
    # Differences from original DeepSonar:
    # - Added scaling for better NN performance.
    feature_scaler = StandardScaler().fit(features_train)
    features_train = feature_scaler.transform(features_train)
    features_val = feature_scaler.transform(features_val)
    features_test = feature_scaler.transform(features_test)

    scaler_path = os.path.join(DATA_DIR, SCALER_OUTPUT_FILE)
    joblib.dump(feature_scaler, scaler_path)

    # To tensors
    features_train_tensor = torch.tensor(features_train, dtype=torch.float32)
    labels_train_tensor = torch.tensor(labels_train, dtype=torch.float32)
    features_val_tensor = torch.tensor(features_val, dtype=torch.float32)
    labels_val_tensor = torch.tensor(labels_val, dtype=torch.float32)
    features_test_tensor = torch.tensor(features_test, dtype=torch.float32)
    labels_test_tensor = torch.tensor(labels_test, dtype=torch.float32)
    
    # Setup model, loss, optimiser
    # Differences from original DeepSonar:
    # - Adam instead of SGD, added weight_decay.
    model = BinaryClassifier(input_dim=features.shape[1]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()  # Handles logits directly
    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
    
    
    for epoch in range(10000):
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
            val_loss = criterion(val_output, labels_val_tensor.to(DEVICE)).item()
            val_acc = accuracy_score(labels_val, val_pred)
            val_auc = roc_auc_score(labels_val, val_prob)
            val_f1 = f1_score(labels_val, val_pred)
        
        if (epoch + 1) % 1000 == 0:
            print(f"Epoch {epoch+1}: train_loss={loss.item():.6f}, "
                  f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f}, "
                  f"val_auc={val_auc:.3f}, val_f1={val_f1:.3f}")

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
    # Differences: Added EER metric (not in paper; my version had basic acc/auc only).
    fpr, tpr, thresholds = roc_curve(labels_test, test_prob)
    fnr = 1 - tpr
    eer_threshold_idx = np.nanargmin(np.absolute(fnr - fpr))
    eer = fpr[eer_threshold_idx]
    print(f"EER: {eer:.4f} at threshold={thresholds[eer_threshold_idx]:.4f}")

    model_path = os.path.join(DATA_DIR, MODEL_OUTPUT_FILE)
    torch.save(model.state_dict(), model_path)
    print("Classifier trained and saved.")
