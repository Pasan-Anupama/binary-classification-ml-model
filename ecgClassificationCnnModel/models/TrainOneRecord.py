# This code contain s the code to train the model

import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from .CnnModel import build_cnn

def train_model(X,y):
    # Reshape for CNN (samples, timesteps, channels)
    # X = np.expand_dims(X, axis=-1)
    
    # Split data (X -> class balanced set of segments from a record, y -> class balanced set of labels from a record
    # from these segmemts and labels, 80% is split as trainset (X_train, y_train) and 20% is split as testset(X_test, y_test)
    # This is done only for one record)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Build model -> No training happens here; this just sets up the layers and parameters of the CNN.
    model = build_cnn(input_shape=X_train.shape[1:])
    
    # Callbacks -> patience=5 means the training will stop if the monitored metric is not improved for 5 consecutivce epchos 
    # This reduces the over fitting and save computational resources
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    ]
    
    # Train -> trains the compiled CNN model on -> X_train is the input data (features/segments) and 
    # y_train are the corresponding labels (ground truth). 
    # batch_size=32 means the model processes 32 samples at a time before updating weights.
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
        callbacks=callbacks
    )
    
    return model, history