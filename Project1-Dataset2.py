#Abhijeet Solanki T00221273
#Project: 1 Dataset:2
#CSC-6903-001
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# 1. Load the MNIST dataset
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 2. Preprocess the dataset
# Normalize the pixel values to be between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# Flatten the images
train_images = train_images.reshape((-1, 28*28))
test_images = test_images.reshape((-1, 28*28))

# Use only the first 1000 training images
train_images = train_images[:1000]
train_labels = train_labels[:1000]

# 3. Define the neural network architecture
model = keras.Sequential([
    keras.layers.Dense(60, activation='relu', input_shape=(28*28,), kernel_initializer='he_normal', kernel_regularizer=keras.regularizers.l1_l2(l1=0.00015, l2=0.00015)),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.15),
    keras.layers.Dense(30, activation='relu', kernel_initializer='he_normal', kernel_regularizer=keras.regularizers.l1_l2(l1=0.00015, l2=0.00015)),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.15),
    keras.layers.Dense(10, activation='relu', kernel_initializer='he_normal', kernel_regularizer=keras.regularizers.l1_l2(l1=0.00015, l2=0.00015)),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.15),
    keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.00025),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 4. Train the model on the modified dataset
history = model.fit(train_images, train_labels, epochs=50, batch_size=32, validation_data=(test_images, test_labels))

# 5. Evaluate the model
train_loss, train_acc = model.evaluate(train_images, train_labels, verbose=0)
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)

# Calculate the ratio
total_params = np.sum([np.prod(v.get_shape().as_list()) for v in model.trainable_variables])
ratio = test_acc / (total_params ** (1/10))

# Print results
print(f"Mini-batch size: 32")
print(f"Number of hidden layers: 3")
print(f"Listing of the number of hidden units: [60, 30, 10]")
print(f"Total number of parameters: {total_params}")
print(f"Optimization method used: Adam")
print(f"Learning rate: 0.00025")
print(f"Initializer: he_normal")
print(f"Regularization: L1 + L2 + Dropout")
print(f"Number of training epochs: {len(history.history['loss'])}")
print(f"Final training accuracy: {train_acc:.4f}")
print(f"Final testing accuracy: {test_acc:.4f}")
print(f"Ratio: {ratio:.4f}")

# 6. Plot the convergence curve
plt.figure(figsize=(12, 4))

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.tight_layout()
plt.show()