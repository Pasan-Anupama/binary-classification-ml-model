import torch
import numpy as np
import os
from models.snn_model import SNNBinaryClassifier
from training.train import train, evaluate
from training.config import Config

def main():
    # Load data
    X_train = np.load('data/train_data.npy')
    y_train = np.load('data/train_labels.npy')
    X_test = np.load('data/test_data.npy')
    y_test = np.load('data/test_labels.npy')

    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    print(f"Testing labels shape: {y_test.shape}")

    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    input_size = X_train.shape[1]
    hidden_size = 64
    output_size = 1
    time_steps = 50

    model = SNNBinaryClassifier(input_size, hidden_size, output_size, time_steps)
    config = Config(lr=0.001, batch_size=32, epochs=20)

    print("Starting training...")
    accuracy_history = train(model, X_train, y_train, config)

    print("Evaluating model on test data...")
    evaluate(model, X_test, y_test)

    # Create directories if they don't exist
    os.makedirs('saved_models', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)

    # Save the trained model
    torch.save(model.state_dict(), 'saved_models/best_model.pth')
    print("Model saved to saved_models/best_model.pth")

    # Save accuracy history for plotting
    np.save('visualizations/accuracy_history.npy', np.array(accuracy_history))
    print("Training accuracy history saved to visualizations/accuracy_history.npy")

if __name__ == "__main__":
    main()
