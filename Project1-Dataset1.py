#Abhijeet Solanki T00221273
#Project: 1 Dataset:1
#CSC-6903-001
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler

# 1 Data Loading and Preprocessing
(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
print("Data loaded successfully!")

train_images = train_images.astype("float32") / 255.0
test_images = test_images.astype("float32") / 255.0
print("Data Normalize successfully!")

train_images = train_images.reshape((-1, 28, 28, 1))
test_images = test_images.reshape((-1, 28, 28, 1))
print("Data reshaped to (28, 28, 1) successfully!")

train_images_1 = train_images[:50000]
train_labels_1 = train_labels[:50000]

# Data Augmentation
datagen = ImageDataGenerator(rotation_range=10, zoom_range=0.1, width_shift_range=0.1, height_shift_range=0.1)
datagen.fit(train_images_1.reshape(-1, 28, 28, 1))

# 2 Neural Networks Model
def create_model(hidden_layers, hidden_units):
    model = keras.Sequential()
    model.add(layers.InputLayer(input_shape=(28, 28, 1)))
    model.add(layers.Flatten())

    for _ in range(hidden_layers):
        model.add(layers.Dense(hidden_units, kernel_initializer="he_normal"))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.Dropout(0.1))  # Reduced dropout

    model.add(layers.Dense(10, activation='softmax'))
    return model
print("Model created successfully!")

# 3 Training and Evaluation
def train_and_evaluate(train_images, train_labels, test_images, test_labels,
                       hidden_layers, hidden_units,
                       batch_size, epochs, learning_rate,
                       optimizer, initializer, regularization=None,
                       callbacks=None):

    model = create_model(hidden_layers, hidden_units)

    if regularization:
        for layer in model.layers:
            if isinstance(layer, layers.Dense):
                layer.kernel_regularizer = keras.regularizers.l2(regularization)

    model.compile(optimizer=optimizer(learning_rate=learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Added early stopping
    early_stopping = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)

    # Train the model
    history = model.fit(datagen.flow(train_images, train_labels, batch_size=batch_size),
                        epochs=epochs,
                        validation_data=(test_images, test_labels),
                        callbacks=callbacks,
                        steps_per_epoch=train_images.shape[0] // batch_size)

    train_accuracy = history.history['accuracy'][-1]
    test_loss, test_accuracy = model.evaluate(test_images, test_labels)

    total_params = np.sum([np.prod(v.get_shape().as_list()) for v in model.trainable_variables])

    ratio = test_accuracy / (total_params ** (1/10))

    results = {
        "Mini-batch size": batch_size,
        "Number of hidden layers": hidden_layers,
        "Number of hidden units": hidden_units,
        "Total number of parameters": total_params,
        "Learning rate": learning_rate,
        "Initializer": initializer,
        "Regularization": regularization,
        "Number of training epochs": epochs,
        "Final training accuracy": train_accuracy,
        "Final testing accuracy": test_accuracy,
        "Ratio": ratio
    }

    return results, history
# Learning rate scheduler
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001, verbose=1)


# 4 Results and Visualization
def plot_results(history):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.tight_layout()
    plt.show()

# 5 Hyperparameter Tuning
params = {
    "hidden_layers": 3,
    "hidden_units": 60,
    "batch_size": 28,
    "epochs": 35,
    "learning_rate": 0.002,
    "optimizer": keras.optimizers.Adam,
    "initializer": "he_normal",
    "regularization": None  
}

# Adding early stopping with restore_best_weights
early_stopping = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)

results, history = train_and_evaluate(train_images_1, train_labels_1, test_images, test_labels, **params, callbacks=[early_stopping, reduce_lr])

for key, value in results.items():
    print(f"{key}: {value}")

plot_results(history)
