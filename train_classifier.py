import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import pandas as pd

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class BinaryNN(nn.Module):
    def __init__(self, input_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(1)
    

if __name__ == "__main__":
    all_labels = np.load("labels.npy")
    print("Label distribution:")
    print(pd.Series(all_labels).value_counts())
    X = np.load("features.npy")
    y = np.load("labels.npy")

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=4347)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=4347)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)


    model = BinaryNN(input_dim=X.shape[1]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-6)

    for epoch in range(10000):
        model.train()
        optimizer.zero_grad()
        output = model(X_train.to(DEVICE))
        loss = criterion(output, y_train.to(DEVICE))
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_output = model(X_val.to(DEVICE))
            val_prob = torch.sigmoid(val_output).cpu().numpy()
            val_pred = (val_prob > 0.5).astype(int)
            val_loss = criterion(val_output, y_val.to(DEVICE)).item()
            acc = accuracy_score(y_val, val_pred)
            auc = roc_auc_score(y_val, val_prob) 
        
        if (epoch + 1) % 1000 == 0:
            print(f"Epoch {epoch+1}: loss={loss.item():.4f}, val_loss={val_loss:.4f}, val_acc={acc:.3f}, val_auc={auc:.3f}")

    torch.save(model.state_dict(), "trained_model.pth")
    print("Classifier trained and saved.")
