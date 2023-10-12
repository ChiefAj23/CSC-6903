#Abhijeet Solanki T00221273
#Project: 1 Dataset:4
#CSC-6903-001
import numpy as np
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import pandas as pd


#1. Load the MNIST dataset and modify it:
# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Modify the dataset

def concatenate_images(images, labels, is_train=True):
    if is_train:
        first = images[:40000]
        second = images[10000:50000]
        third = images[20000:60000]
    else:
        first = images[:8000]
        second = images[1000:9000]
        third = images[2000:10000]

    concatenated_images = np.hstack((first, second, third)).reshape(-1, 28, 84)

    if is_train:
        concatenated_labels = labels[:40000] * 100 + labels[10000:50000] * 10 + labels[20000:60000]
    else:
        concatenated_labels = labels[:8000] * 100 + labels[1000:9000] * 10 + labels[2000:10000]

    return concatenated_images, concatenated_labels

train_images_modified, train_labels_modified = concatenate_images(train_images, train_labels)
test_images_modified, test_labels_modified = concatenate_images(test_images, test_labels, is_train=False)

print("Shapes:", train_images_modified.shape, train_labels_modified.shape, test_images_modified.shape, test_labels_modified.shape)


# Normalize the data
train_images_modified = train_images_modified.astype('float32') / 255.0
test_images_modified = test_images_modified.astype('float32') / 255.0


#2. Define the fully connected neural network:
model = Sequential([
    Flatten(input_shape=(28, 84)),
    Dense(512, activation='relu', kernel_initializer='he_normal'),
    Dropout(0.2),
    Dense(256, activation='relu', kernel_initializer='he_normal'),
    Dropout(0.2),
    Dense(128, activation='relu', kernel_initializer='he_normal'),
    Dropout(0.2),
    Dense(64, activation='relu', kernel_initializer='he_normal'),
    Dropout(0.2),
    Dense(1000, activation='softmax')
])

#3. Train the neural network:
# Hyperparameters
batch_size = 128
epochs = 100
learning_rate = 0.001

# Compile the model
model.compile(optimizer=Adam(learning_rate=learning_rate),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Early stopping and learning rate reduction on plateau
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001)

# Train the model
history = model.fit(train_images_modified, train_labels_modified,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(test_images_modified, test_labels_modified),
                    callbacks=[early_stopping, reduce_lr])

#4. Evaluate the model:
train_accuracy = history.history['accuracy'][-1]
test_accuracy = history.history['val_accuracy'][-1]
total_parameters = np.sum([np.prod(v.get_shape().as_list()) for v in model.trainable_variables])
ratio = test_accuracy / (total_parameters ** (1/10))

# Create a dictionary with the results and hyperparameters
results = {
    'Mini-batch size': [batch_size],
    'Number of hidden layers': [len(model.layers) - 2],
    'Number of hidden units': [[256, 128, 64]],
    'Total number of parameters': [total_parameters],
    'Optimization method used': ['Adam'],
    'Learning rate': [learning_rate],
    'Initializer': ['Glorot Uniform'],
    'Regularization': ['Dropout (0.5)'],
    'Number of training epochs': [len(history.history['loss'])],
    'Final training accuracy': [train_accuracy],
    'Final testing accuracy': [test_accuracy],
    'Ratio': [ratio]
}
# Set pandas display options to show all columns
pd.set_option('display.max_columns', None)

# Convert the dictionary to a pandas DataFrame
df = pd.DataFrame(results)

# Display the DataFrame
print(df)

#5. Plot the convergence curve:
plt.figure(figsize=(12, 4))

# Plotting loss
plt.subplot(1, 3, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.legend()
plt.title('Loss Convergence')

#5. Plot the convergence curve:
plt.figure(figsize=(10, 4))

# Plotting loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.title('Loss Convergence')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()


# Plotting accuracy convergence
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Test Accuracy')
plt.title('Accuracy Convergence')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()



plt.tight_layout()
plt.show()