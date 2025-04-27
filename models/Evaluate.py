#The testing script

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image 

def predict_image(model, img_path):
    """Predict classes for single image."""
    img = image.load_img(img_path, target_size=(150,150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)
    return "Dog 🐶" if prediction[0] > 0.5 else "Cat 😺"

if __name__ == "__main__":
    #Load model
    model = tf.keras.models.load_model('saved_models/dog_cat_cnn.h5')
    
    #Test on sample images 
    test_images = [
        'data/validation/dogs/test_dog.jpg',
        'data/validation/cats/8.jpg'
    ]
    
    for img_path in test_images:
        print(f"{img_path}: {predict_image(model, img_path)}")