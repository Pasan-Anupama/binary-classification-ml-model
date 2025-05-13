"""
train.py
--------
Training loop for the SNN binary classification model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from training.utils import accuracy_score

def train(model, X_train, y_train, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    model.train()

    accuracy_history = []   # <--- Initialize here

    for epoch in range(1, config.epochs + 1):
        epoch_loss = 0.0
        epoch_acc = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            acc = accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
            epoch_acc += acc

        avg_loss = epoch_loss / len(dataloader)
        avg_acc = epoch_acc / len(dataloader)
        accuracy_history.append(avg_acc)   # Append accuracy for this epoch
        print(f"Epoch {epoch}/{config.epochs} - Loss: {avg_loss:.4f} - Accuracy: {avg_acc:.4f}")

    return accuracy_history

        
def evaluate(model, X_test, y_test):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        inputs = X_test.to(device)
        labels = y_test.to(device).unsqueeze(1)
        outputs = model(inputs)
        preds = (torch.sigmoid(outputs) > 0.5).float()
        acc = accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
        print(f"Number of test samples: {len(y_test)}")
        print(f"Test Accuracy: {acc:.4f}")
