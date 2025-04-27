#Designing the CNN model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def create_model(input_shape=(150,150,3)):
    """Define the CNN architecture"""
    model = Sequential([
        #Convolution layer 1 and pooling layer -> Block 1
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2,2),
        
        #Convolution layer 2 and pooling layer -> Block 2
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        
        #Fully connected layers
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy'] 
    )
    
    return model

if __name__ == "__main__" :
    model = create_model()
    model.summary()