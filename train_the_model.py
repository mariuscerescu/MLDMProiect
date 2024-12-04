import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
import joblib

# Load the training data
print("Loading training data...")
train_data = pd.read_csv("dataset/train/emnist-letters-train.csv")

# Prepare the data
X = train_data.iloc[:, 1:].values  # All columns except the first one (label)
y = train_data.iloc[:, 0].values   # First column is the label

# Reshape the data for CNN
X = X.reshape(-1, 28, 28, 1)
X = X / 255.0  # Normalize pixel values

# Convert labels to one-hot encoding
y = to_categorical(y - 1)  # Subtract 1 because labels are 1-26, we need 0-25

# Split the data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(26, activation='softmax')  # 26 classes for letters A-Z
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("Training model...")
# Train the model
history = model.fit(X_train, y_train,
                    epochs=10,
                    batch_size=64,
                    validation_data=(X_val, y_val))

print("Saving model...")
# Save the model
model.save('letter_recognition_model.h5')

print("Training completed and model saved!")
