from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load the image
img = Image.open('opera_house.jpg')

# Print image dimensions and color channels
print(np.array(img).shape)

# Display the original image
plt.subplot(2, 2, 1)
plt.imshow(img)
plt.title("Original")

# Resize the image to 500 x 500 pixels
plt.subplot(2, 2, 2)
plt.imshow(img.resize((500, 500)))
plt.title("Resized")

# Flip the image horizontally
plt.subplot(2, 2, 3)
plt.imshow(img.transpose(Image.Transpose.FLIP_LEFT_RIGHT))
plt.title("Flipped")

# Rotate the image by 45 degrees
plt.subplot(2, 2, 4)
plt.imshow(img.rotate(45))
plt.title("Rotated")

# Display all four images
plt.show()
