#Verify the images are correctly loaded

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#plot a sample cat and dog
plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1)
img = mpimg.imread('data/train/cats/0.jpg')
plt.imshow(img)
plt.title("Cat")

plt.subplot(1, 2, 2)
img = mpimg.imread('data/train/dogs/0.jpg')
plt.imshow(img)
plt.title('Dog')
plt.show()