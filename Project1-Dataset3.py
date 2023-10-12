#Abhijeet Solanki T00221273
#Project: 1 Dataset:3
#CSC-6903-001
import numpy as np
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.initializers import he_normal
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt


# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Modify the dataset
train_images = (train_images + 765) / 4.0
test_images = (test_images + 765) / 4.0

# Normalize the images to [0, 1]
train_images = train_images / 1023.0
test_images = test_images / 1023.0

# Flatten the images
train_images = train_images.reshape(train_images.shape[0], 28*28)
test_images = test_images.reshape(test_images.shape[0], 28*28)

# Define a neural network model with batch normalization
model = Sequential([
    Dense(256, activation='relu', input_shape=(28*28,), kernel_initializer=he_normal()),
    BatchNormalization(),
    Dense(128, activation='relu', kernel_initializer=he_normal()),
    BatchNormalization(),
    Dense(64, activation='relu', kernel_initializer=he_normal()),
    BatchNormalization(),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001)

# Train the model for more epochs with callbacks
history = model.fit(train_images, train_labels, epochs=30, batch_size=32, validation_data=(test_images, test_labels), callbacks=[early_stopping, reduce_lr])

# Evaluate the model
train_accuracy = history.history['accuracy'][-1]
test_loss, test_accuracy = model.evaluate(test_images, test_labels)

# Calculate the ratio
params_count = model.count_params()
ratio = test_accuracy / np.power(params_count, 1/10)

# Print results
print('Mini-batch size:', 32)
print('Number of hidden layers:', 2)
print('Number of hidden units:', [128, 64])
print('Total number of parameters:', params_count)
print('Optimization method used:', 'Adam')
print('Learning rate:', 0.001)
print('Initializer:', 'he_normal')
print('Regularization:', 'None')
print('Number of training epochs:', 10)
print('Final training accuracy:', train_accuracy)
print('Final testing accuracy:', test_accuracy)
print('Ratio:', ratio)

# Plotting convergence curve of the loss function
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Testing Loss')
plt.title('Loss Convergence Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plotting training accuracy
plt.subplot(1, 3, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Testing Accuracy')
plt.title('Accuracy Convergence Curve')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

