import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("mask_model.h5")

img = cv2.imread("dataset/images/maksssksksss0.png")  # Change this to test image
img = cv2.resize(img, (100, 100))
img = np.expand_dims(img / 255.0, axis=0)

prediction = model.predict(img)[0][0]
print("Mask Detected ✅" if prediction > 0.5 else "No Mask ❌")
