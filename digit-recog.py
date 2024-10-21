import pickle
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt

with open('model_c','rb') as f:
    mp=pickle.load(f)

def preprocess_image(image_path):
    img = Image.open(image_path).convert('L')
    img = ImageOps.invert(img)
    img = img.resize((28, 28))
    img = np.array(img) / 255.0
    img = img.reshape(1, 28, 28, 1)
    return img

# Path to the handwritten digit image
image_path = 'digit9.jpg'
new_image = preprocess_image(image_path)

# Predict the digit
prediction = mp.predict(new_image)
predicted_digit = np.argmax(prediction)
print(f"Predicted Digit: {predicted_digit}")

# Display the test image and the predicted digit
plt.imshow(new_image.reshape(28, 28), cmap='gray')
plt.title(f"Predicted Digit: {predicted_digit}")
plt.show()