# This code contain s the code to train the model

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from CnnModel import build_cnn

def train_model(X,y):
     # Reshape for CNN (samples, timesteps, channels)
    X = np.expand_dims(X, axis=-1)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Build model
    model = build_cnn(input_shape=X_train.shape[1:])
    
    # Callbacks
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True)
    ]
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
        callbacks=callbacks
    )
    
    return model, history