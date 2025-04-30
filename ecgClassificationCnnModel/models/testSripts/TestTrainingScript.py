import sys
import os
# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.Train import train_model
import numpy as np

# Generate synthetic ECG-like data
def create_sample_data(n_samples=1000, timesteps=250):
    # Create normal beats (class 0)
    normal_beats = np.array([
        0.2 * np.exp(-50*(np.linspace(0,1,timesteps)-0.25)**2) +  # P-wave
        1.0 * np.exp(-50*(np.linspace(0,1,timesteps)-0.5)**2) -   # R-peak 
        0.3 * np.exp(-50*(np.linspace(0,1,timesteps)-0.7)**2)     # T-wave
        for _ in range(n_samples//2)
    ])
    
    # Create abnormal beats (class 1)
    abnormal_beats = np.array([
        0.5 * np.exp(-30*(np.linspace(0,1,timesteps)-0.3)**2) -  # Wide QRS
        0.8 * np.exp(-40*(np.linspace(0,1,timesteps)-0.6)**2)     # Inverted T
        for _ in range(n_samples//2)
    ])
    
    X = np.vstack([normal_beats, abnormal_beats])
    y = np.array([0]*(n_samples//2) + [1]*(n_samples//2))
    return X, y

# Run test
if __name__ == "__main__":
    print("Generating sample ECG data...")
    X, y = create_sample_data()
    
    print("Training model...")
    model, history = train_model(X, y)
    
    print("\nTraining completed!")
    print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.2f}")
    print("Model summary:")
    model.summary()