import tensorflow as tf
from .CnnModel import build_cnn

def train_model(X_train, y_train, X_val, y_val):
    model = build_cnn(input_shape=X_train.shape[1:])
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=callbacks
    )
    
    return model, history

