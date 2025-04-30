# This contains the code for designing the CNN model

from tensorflow.keras import layers, models

def build_cnn(input_shape, num_classes=2):
    """Builds CNN that adapts to any input length"""
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(64, 15, activation='relu', padding='same'), 
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Conv1D(128, 7, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Conv1D(256, 5, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
