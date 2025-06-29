# train.py

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

DATA_DIR = r"C:\Users\Isha Kumari\PycharmProjects\PythonProject8\data"
IMG_SIZE = (128, 128)
BATCH_SIZE = 8
EPOCHS = 10
MODEL_PATH = "mask_model_small.keras"

# Preprocess data
datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

train_data = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

val_data = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# Build model
model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(*IMG_SIZE, 3)),
    MaxPooling2D(2, 2),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(train_data, validation_data=val_data, epochs=EPOCHS)

# Save in Keras format
model.save(MODEL_PATH)
print(f"✅ Model saved as: {MODEL_PATH}")
