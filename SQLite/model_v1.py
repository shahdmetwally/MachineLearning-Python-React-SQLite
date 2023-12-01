from sklearn.model_selection import train_test_split
import sqlite3
import pickle
import numpy as np
import keras
from keras.utils import to_categorical


def load_dataset(database_path):
    # Connecting to the SQLite database
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    # Retrieving data from the database
    cursor.execute("SELECT target, name, image FROM faces")
    rows = cursor.fetchall()

    # Close the connection
    conn.close()

    # Splitting the dataset into features (X) and labels (y)
    images = []
    names = []
    targets = []

    for row in rows:
        target, name, image_bytes = row
        # Deserialize the image data
        image_data = pickle.loads(image_bytes)
        image_data_normalized = image_data
        images.append(image_data_normalized)
        names.append(name)
        targets.append(target)
    return images, targets, names


def split_dataset(images, names, targets, test_size=0.2, random_state=0):
    num_classes = len(np.unique(targets))
    X_train, X_test, y_train, y_test = train_test_split(
        images, names, test_size=test_size, random_state=random_state
    )
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # Print the shapes of the resulting sets
    print("Training set shape:", X_train.shape, y_train.shape)
    print("Testing set shape:", X_test.shape, y_test.shape)

    # Plot the first image in X_train
    plt.imshow(X_train[0])
    plt.title("First Image in X_train")
    plt.show()

    return X_train, X_test, y_train, y_test, num_classes, names


def preprocess_and_print_shapes(y_train, y_test):
    y_train_categorical = keras.utils.to_categorical(y_train, num_classes)
    y_test_categorical = keras.utils.to_categorical(y_test, num_classes)

    print("Training set shape:", X_train.shape, y_train.shape)
    print("Testing set shape:", X_test.shape, y_test.shape)

    return y_train_categorical, y_test_categorical


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


def train_cnn_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation="relu", input_shape=input_shape))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(256, (3, 3), activation="relu"))
    model.add(MaxPooling2D(2, 2))
    model.add(Flatten())
    model.add(Dense(1024, activation="relu"))
    model.add(Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return model


def get_accuracy(model):
    score = model.evaluate(X_test, y_test, verbose=0)
    print("Test accuracy:", score[1])
    return score


import matplotlib.pyplot as plt
import numpy as np


def visualize_predictions(
    model, X_test, y_test, num_rows=5, num_cols=5, figsize=(20, 20)
):
    # Make predictions on the test set
    predicted_probabilities = model.predict(X_test)
    predicted_classes = np.argmax(predicted_probabilities, axis=1)

    # Create subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.ravel()

    # Loop through the test set and visualize predictions
    for i in range(num_rows * num_cols):
        actual_name = np.argmax(y_test, axis=1)[i]
        predicted_name = predicted_classes[i]

        axes[i].imshow(X_test[i])
        axes[i].set_title(
            "Prediction Class = {}\nTrue Class = {}".format(predicted_name, actual_name)
        )
        axes[i].axis("off")

    plt.subplots_adjust(wspace=0.5)
    plt.show()


# In[7]:

"""'
images, targets, names = load_dataset('lfw_dataset.db')
X_train, X_test, y_train, y_test, num_classes, names = split_dataset(images, targets, names)

y_train, y_test = preprocess_and_print_shapes(y_train, y_test)

input_shape = (62, 47, 3)
cnn_model = train_cnn_model(input_shape, num_classes)

# Train the model
hist = cnn_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=10)
get_accuracy(cnn_model)

# Save the trained model
cnn_model.save('trained_model.h5')

# Assuming lfw_dataset.target_names contains class names
visualize_predictions(cnn_model, X_test, y_test)
"""
