# test.py

import cv2
import numpy as np
from tensorflow.keras.models import load_model

MODEL_PATH = "mask_model_small.keras"
IMG_SIZE = (128, 128)
model = load_model(MODEL_PATH)

def predict(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)[0][0]
    label = "Mask" if prediction > 0.5 else "No Mask"
    confidence = max(prediction, 1 - prediction)
    print(f"Prediction: {label} (Confidence: {confidence:.2%})")

# Example: python test.py "path_to_image.jpg"
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        predict(sys.argv[1])
    else:
        print("Usage: python test.py <image_path>")
print("Class indices:", train_data.class_indices)
