#Contains the training script

from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from Model import create_model
import os

#Data generators
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    'data/validation',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

#create and train model
model = create_model()
history = model.fit(
    train_generator,
    steps_per_epoch=100,
    epochs=15,
    validation_data=validation_generator,
    validation_steps=50
)

#Save model
os.makedirs('saved_models', exist_ok=True)
model.save('saved_models/dog_cat_cnn.h5')
print("Model trained and saved !")