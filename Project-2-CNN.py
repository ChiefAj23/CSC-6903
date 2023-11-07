#Abhijeet Solanki T00221273
#Project: 2
#Model:CNN
#Due Date: 11/06/2023
#CSC-6903-001
import tensorflow as tf
import numpy as np
import keras.backend as K
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from keras import regularizers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# Constants
VOCAB_SIZE = 10000
MAX_LEN = 500
LEARNING_RATE = 0.0005
REGULARIZATION_FACTOR = 0.01

# Load the dataset and preprocess data
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=VOCAB_SIZE)
x_train = sequence.pad_sequences(x_train, maxlen=MAX_LEN)
x_test = sequence.pad_sequences(x_test, maxlen=MAX_LEN)

# Define the CNN model
cnn_model = Sequential([
    Embedding(VOCAB_SIZE, 128, input_length=MAX_LEN, embeddings_regularizer=regularizers.l2(REGULARIZATION_FACTOR/10)), # Added regularization to embedding
    Conv1D(128, 7, activation='relu', kernel_regularizer=regularizers.l2(REGULARIZATION_FACTOR)),
    tf.keras.layers.BatchNormalization(),  # Added batch normalization
    Dropout(0.5),  # Adjusted dropout rate from v6
    GlobalMaxPooling1D(),
    Dropout(0.5),  # Adjusted dropout rate from v6
    Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(REGULARIZATION_FACTOR))
])

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
cnn_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.001)

# Train the CNN model
cnn_history = cnn_model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test),
                            callbacks=[early_stopping, reduce_lr])



# Displaying model information and plotting history remain the same
def display_model_info(model, history, model_name, batch_size, epochs, optimizer, learning_rate):
    # Extracting details from the model
    if model_name == 'CNN':
        conv_layer = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Conv1D)][0]
        num_channels = conv_layer.filters
        kernel_size = conv_layer.kernel_size[0]
        details = f"Channels: {num_channels}, Kernel Size: {kernel_size}"
    elif model_name == 'LSTM':
        lstm_layer = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.LSTM)][0]
        hidden_units = lstm_layer.units
        details = f"Hidden Units: {hidden_units}"

    # Total parameters excluding embedding layer
    total_params = sum([l.count_params() for l in model.layers[1:]])

    # Final training and testing accuracy
    final_train_accuracy = history.history['accuracy'][-1]
    final_test_accuracy = history.history['val_accuracy'][-1]

    # Ratio of the test accuracy to the 10th root of the total number of parameters
    ratio = final_test_accuracy / (total_params ** (1/10))

    print(f"{'Model:':<40} {model_name}")
    print(f"{'Mini-batch size:':<40} {batch_size}")
    print(f"{'Number of hidden layers:':<40} {len(model.layers) - 2}")
    print(f"{'Details (Channels/Kernels/Units):':<40} {details}")
    print(f"{'Total parameters (excl. embedding):':<40} {total_params}")
    print(f"{'Optimization method used:':<40} {optimizer}")
    print(f"{'Learning rate:':<40} {learning_rate}")
    print(f"{'Initializer:':<40} {'Default (Glorot Uniform)'}")
    print(f"{'Regularization (if used):':<40} {'Kernel Regularizer'}")
    print(f"{'Number of training epochs:':<40} {epochs}")
    print(f"{'Final training accuracy:':<40} {final_train_accuracy:.4f}")
    print(f"{'Final testing accuracy:':<40} {final_test_accuracy:.4f}")
    print(f"{'Ratio (Accuracy/10th root of params):':<40} {ratio:.4f}")
    print("-" * 60)



def plot_history(history, title=''):
    plt.figure(figsize=(12, 5))

    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(title + ' Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(title + ' Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.tight_layout()
    plt.show()

# For CNN model
display_model_info(cnn_model, cnn_history, 'CNN', batch_size=32, epochs=10, optimizer='Adam', learning_rate=0.0005)
plot_history(cnn_history, title='CNN')
