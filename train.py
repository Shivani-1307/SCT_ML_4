import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Define dataset path and parameters
DATASET_PATH = 'leapGestRecog/' 
IMG_SIZE = 64  # Resize images to 64x64 for faster training

# Mapping gesture folder names to integer labels
GESTURES = {
    '01_palm': 0, '02_l': 1, '03_fist': 2, '04_fist_moved': 3, '05_thumb': 4,
    '06_index': 5, '07_ok': 6, '08_palm_moved': 7, '09_c': 8, '10_down': 9
}

def load_data():
    X = []
    y = []
    
    # Loop through all 10 subject directories (00 to 09)
    for subject in os.listdir(DATASET_PATH):
        subject_path = os.path.join(DATASET_PATH, subject)
        if not os.path.isdir(subject_path):
            continue
            
        # Loop through each gesture folder inside the subject directory
        for gesture_name, label in GESTURES.items():
            gesture_path = os.path.join(subject_path, gesture_name)
            if not os.path.isdir(gesture_path):
                continue
                
            for img_name in os.listdir(gesture_path):
                img_path = os.path.join(gesture_path, img_name)
                # Read as grayscale since the dataset contains infrared images
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                if img is not None:
                    # Resize and normalize image data
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    X.append(img)
                    y.append(label)
                    
    return np.array(X), np.array(y)

print("Loading dataset...")
X, y = load_data()

# Reshape data to include the channel dimension (grayscale = 1)
X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
X = X / 255.0  # Normalize pixels to [0, 1]

# Convert labels to one-hot encoding
y = to_categorical(y, num_classes=10)

# Split into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Data loaded successfully! Train shape: {X_train.shape}, Test shape: {X_test.shape}")

import tensorflow as tf
from tensorflow.keras import layers, models

def build_model():
    model = models.Sequential([
        # Convolutional Block 1
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        layers.MaxPooling2D((2, 2)),
        
        # Convolutional Block 2
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Convolutional Block 3
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Flattening and Dense Layers
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),  # Prevents overfitting
        layers.Dense(10, activation='softmax')  # 10 output classes
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model = build_model()
model.summary()

# Train the model
EPOCHS = 10
BATCH_SIZE = 64

from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=15,      # Randomly rotate images
    zoom_range=0.1,         # Randomly zoom in/out
    width_shift_range=0.1,  # Shift horizontally
    height_shift_range=0.1  # Shift vertically
)

# Then train using the generator instead of standard X_train
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=64),
    epochs=15,
    validation_data=(X_test, y_test)
)

history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test)
)

# Save the trained weights
model.save('hand_gesture_model.h5')
print("Model saved to 'hand_gesture_model.h5'")

