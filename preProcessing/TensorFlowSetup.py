#TensorFlow setup -> Use to develeop and deploy ML models

from tensorflow.keras.preprocessing.image import ImageDataGenerator 

#rescale pixel values
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

#Load data
train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(150,150),
    batch_size=32,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    'data/validation',
    target_size=(150,150),
    batch_size=32,
    class_mode='binary'
)
