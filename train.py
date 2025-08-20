# Import necessary libraries
import os  # For file and directory operations
import cv2  # OpenCV for image loading and processing
import numpy as np  # For array operations
import tensorflow as tf  # TensorFlow framework for deep learning
from tensorflow.keras.models import Sequential  # Sequential model API
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout  # CNN layers
from sklearn.model_selection import train_test_split  # For splitting dataset into training and validation

# ----------------------------
# Step 1: Prepare Data
# ----------------------------

data = []  # List to store all image data
labels = []  # List to store labels: 1 for mask, 0 for no mask

path = 'dataset/images'  # Directory where face mask images are stored

# Loop through each image in the dataset
for img in os.listdir(path):
    label = 1 if "mask" in img.lower() else 0  # Label the image based on filename (1 if 'mask' is in name)
    image = cv2.imread(os.path.join(path, img))  # Read the image using OpenCV
    image = cv2.resize(image, (100, 100))  # Resize image to 100x100 pixels (standard input size for CNN)
    data.append(image)  # Append image array to data list
    labels.append(label)  # Append corresponding label

# Convert lists to numpy arrays and normalize pixel values to [0, 1] by dividing by 255
data = np.array(data) / 255.0
labels = np.array(labels)

# Split the dataset into training and validation sets (80% train, 20% validation)
x_train, x_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)

# ----------------------------
# Step 2: Build CNN Model
# ----------------------------

model = Sequential([  # Start defining a sequential CNN model
    Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    # First convolutional layer with 32 filters of 3x3 size and ReLU activation
    MaxPooling2D(2, 2),  # Max pooling layer to reduce spatial size (downsampling)
    #Takes the output from the previous Conv2D layer.
    #Reduces its size by using a 2×2 window to pick the maximum value in each region.

    Conv2D(64, (3, 3), activation='relu'),  #deeper convolutional layer with 64 filters.(second)
    MaxPooling2D(2, 2),  # Second pooling layer
    # Why we do this:
#Continues reducing the data size while keeping important features.

#Makes model faster and more memory-efficient

    Flatten(),  # Flatten the 3D output from conv layers to 1D for Dense layers
    Dropout(0.3),  # Dropout to prevent overfitting (30% neurons randomly turned off during training)

    Dense(64, activation='relu'),  # Fully connected layer with 64 neurons and ReLU activation
    #What it does:
    #Each neuron gets input from all outputs of the previous layer.
    Dense(1, activation='sigmoid')
    # Output layer with 1 neuron and sigmoid activation for binary classification (mask/no mask)
])

# Compile the model
model.compile(
    optimizer='adam',  # Adam optimizer (adaptive learning rate)
    loss='binary_crossentropy',  # Loss function for binary classification
    metrics=['accuracy']  # Monitor accuracy during training
)

# ----------------------------
# Step 3: Train the Model
# ----------------------------

# Fit (train) the model using training data and validate on validation set
model.fit(
    x_train, y_train,  # Training data and labels
    validation_data=(x_val, y_val),  # Validation data and labels
    epochs=5,  # Train for 5 full passes over the dataset
    batch_size=32  # Number of samples processed before model update
)

# ----------------------------
# Step 4: Save the Model
# ----------------------------

model.save("mask_model.h5")  # Save the trained model to disk for later use (e.g., testing or deployment)


#Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),(size of input imag is 100*100 and apply 3 color(RGB)
#You’ll get 32 different feature maps

#Each map highlights different patterns like:
#Vertical edges
#Horizontal edges
#Corners
#Texture etc
