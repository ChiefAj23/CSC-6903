#Abhijeet Solanki T00221273
#Project: 2
#Model:LSTM
#Due Date: 11/06/2021
#CSC-6903-001
import tensorflow as tf
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Embedding, Dense, Dropout
from keras.regularizers import l2
import matplotlib.pyplot as plt

# Define constants
VOCAB_SIZE = 10000
MAX_LEN = 500

# Load the dataset and preprocess data
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=VOCAB_SIZE)
x_train = sequence.pad_sequences(x_train, maxlen=MAX_LEN)
x_test = sequence.pad_sequences(x_test, maxlen=MAX_LEN)

# LSTM model with L2 regularization and Dropout
lstm_model = Sequential([
    Embedding(input_dim=VOCAB_SIZE, output_dim=128, input_length=MAX_LEN),
    LSTM(64, recurrent_dropout=0.5, dropout=0.5, kernel_regularizer=l2(0.001), recurrent_regularizer=l2(0.001)),
    Dropout(0.5),
    Dense(1, activation='sigmoid', kernel_regularizer=l2(0.001))
])

# Compile the model with a learning rate schedule
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=10000,
    decay_rate=0.9)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
lstm_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Use early stopping with patience and restore the best model weights
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model with validation data and early stopping
lstm_history = lstm_model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(x_test, y_test),
    callbacks=[early_stop]
)

# Evaluate the model
lstm_test_loss, lstm_test_accuracy = lstm_model.evaluate(x_test, y_test)

# Display Model Information
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
    print(f"{'Regularization (if used):':<40} {'l2'}")
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

# LSTM model
display_model_info(lstm_model, lstm_history, 'LSTM', batch_size=32, epochs=10, optimizer='Adam', learning_rate=0.001)
plot_history(lstm_history, title='LSTM')
