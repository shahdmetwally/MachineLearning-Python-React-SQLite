from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import sqlite3
import numpy as np
import keras
import pandas as pd
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D, BatchNormalization
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import pickle
from keras.applications.vgg16 import preprocess_input
from PIL import Image
from keras.applications import vgg16, EfficientNetV2B0
from keras.callbacks import EarlyStopping, LearningRateScheduler
from keras import regularizers
import time
import os
import json
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix)
import numpy as np
from numpy import expand_dims
from keras.models import Model

class LoadDataset(BaseEstimator, TransformerMixin):
    def __init__(self, train_database_path):
        self.train_database_path = train_database_path
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Connect to the SQLite databases
        train_conn = sqlite3.connect(self.train_database_path)
        val_test_database_path='../monorepo/Model/Datasets/lfw_dataset.db'
        val_test_conn = sqlite3.connect(val_test_database_path)
        #Get tables
        train_table='faces' 
        val_table='faces_val'
        test_table='faces_test'

        # Query to select all records from the augmented dataset for training
        train_query = f"SELECT * FROM {train_table}"

        # Fetch records from the database into a Pandas DataFrame
        train_df = pd.read_sql_query(train_query, train_conn)
        train_df['image'] = train_df['image'].apply(lambda x: np.array(pickle.loads(x), dtype=np.uint8))

        # Query to select all records from the validation dataset
        val_query = f"SELECT * FROM {val_table}"

        # Fetch records from the database into a Pandas DataFrame
        val_df = pd.read_sql_query(val_query, val_test_conn)
        val_df['image'] = val_df['image'].apply(lambda x: np.array(pickle.loads(x), dtype=np.uint8))

        # Query to select all records from the test dataset
        test_query = f"SELECT * FROM {test_table}" 

        # Fetch records from the database into a Pandas DataFrame
        test_df = pd.read_sql_query(test_query, val_test_conn)
        test_df['image'] = test_df['image'].apply(lambda x: np.array(pickle.loads(x), dtype=np.uint8))


        # Close the database connections
        train_conn.close()
        val_test_conn.close()

        # Convert image bytes to numpy array
        num_classes = len(np.unique(train_df['target']))
        return train_df['image'], val_df['image'], test_df['image'], train_df['target'], val_df['target'], test_df['target'], num_classes, test_df    


class Preprocess(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_train, X_val, X_test, y_train, y_val, y_test, num_classes, df= X

        # One-hot encode the labels
        y_train_categorical = keras.utils.to_categorical(y_train, num_classes)
        y_val_categorical = keras.utils.to_categorical(y_val, num_classes)
        y_test_categorical = keras.utils.to_categorical(y_test, num_classes)

        # Resize images to 224x224
        X_train = np.array([np.array(Image.fromarray(img).resize((47, 62))) for img in X_train])
        X_val = np.array([np.array(Image.fromarray(img).resize((47, 62))) for img in X_val])
        X_test = np.array([np.array(Image.fromarray(img).resize((47, 62))) for img in X_test])

        # Normalize the images
        X_train = preprocess_input(X_train)
        X_val = preprocess_input(X_val)
        X_test = preprocess_input(X_test)

        return X_train, X_val, X_test, y_train_categorical, y_val_categorical, y_test_categorical, num_classes

class TrainModel(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X_train, X_val, X_test, y_train, y_val, y_test, num_classes = X
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(62,47,3)))
        model.add(MaxPooling2D(2, 2))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(2, 2))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D(2, 2))
        model.add(Conv2D(256, (3, 3), activation='relu'))
        model.add(MaxPooling2D(2, 2))
        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=10)
        print("Number of epochs:", len(history.history['loss']))
        return model, X_test, y_test


class PreprocessVGG16(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_train, X_val, X_test, y_train, y_val, y_test, num_classes, df= X

        # One-hot encode the labels
        y_train_categorical = keras.utils.to_categorical(y_train, num_classes)
        y_val_categorical = keras.utils.to_categorical(y_val, num_classes)
        y_test_categorical = keras.utils.to_categorical(y_test, num_classes)

        # Resize images to 224x224
        X_train = np.array([np.array(Image.fromarray(img).resize((224, 224))) for img in X_train])
        X_val = np.array([np.array(Image.fromarray(img).resize((224, 224))) for img in X_val])
        X_test = np.array([np.array(Image.fromarray(img).resize((224, 224))) for img in X_test])

        # Normalize the images
        X_train = preprocess_input(X_train)
        X_val = preprocess_input(X_val)
        X_test = preprocess_input(X_test)

        return X_train, X_val, X_test, y_train_categorical, y_val_categorical, y_test_categorical, num_classes
class TrainModelVGG16(BaseEstimator, TransformerMixin):    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_train, X_val, X_test, y_train, y_val, y_test, num_classes = X
        vgg=vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))
        for layer in vgg.layers:
            layer.trainable = False

        model = Sequential()
        model.add(vgg)
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l1_l2(0.1)))
        model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=10, callbacks=[early_stopping])
        print("Number of epochs:", len(history.history['loss']))
        return model, X_test, y_test
# import for efficient net preprocess_input()
from keras.applications.efficientnet import preprocess_input

class PreprocessEfficientNet(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_train, X_val, X_test, y_train, y_val, y_test, num_classes, df= X

        # One-hot encode the labels
        y_train_categorical = keras.utils.to_categorical(y_train, num_classes)
        y_val_categorical = keras.utils.to_categorical(y_val, num_classes)
        y_test_categorical = keras.utils.to_categorical(y_test, num_classes)

        # Resize images to 224x224
        X_train = np.array([np.array(Image.fromarray(img).resize((224, 224))) for img in X_train])
        X_val = np.array([np.array(Image.fromarray(img).resize((224, 224))) for img in X_val])
        X_test = np.array([np.array(Image.fromarray(img).resize((224, 224))) for img in X_test])

        # Normalize the images
        X_train = preprocess_input(X_train)
        X_val = preprocess_input(X_val)
        X_test = preprocess_input(X_test)

        return X_train, X_val, X_test, y_train_categorical, y_val_categorical, y_test_categorical, num_classes
    
class TrainModelEfficientNet(BaseEstimator, TransformerMixin):    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_train, X_val, X_test, y_train, y_val, y_test, num_classes = X
        efficientnet_model = EfficientNetV2B0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        for layer in efficientnet_model.layers[-20:]:
            layer.trainable = False

        model = Sequential()
        model.add(efficientnet_model)
        model.add(GlobalAveragePooling2D())
        # Add BatchNormalization after GlobalAveragePooling2D
        model.add(BatchNormalization())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l1_l2(0.08)))
        def schedule(epoch, lr):
            if epoch < 4:
                return 0.001
            else:
                return 0.001 * np.exp(0.1 * (4 - epoch))

        lr_scheduler = LearningRateScheduler(schedule)
        model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=12, batch_size=15, callbacks=[early_stopping, lr_scheduler])
        print("Number of epochs:", len(history.history['loss']))
        return model, X_test, y_test
    
class VisualizeFeatureMaps(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Assuming X is a tuple (image, label)
        model, X_test, y_test = X
        
        image = X_test[0]

        conv_layer_indices = [i for i, layer in enumerate(model.layers) if isinstance(layer, Conv2D)]

        #Create a list to store convolutional layers
        conv_layers = [layer for layer in model.layers if isinstance(layer, Conv2D)]

        #Create a new model containing only the convolutional layers
        model2 = Model(inputs=model.inputs, outputs=[layer.output for layer in conv_layers])

        #Expand the dimensions of the input to match the model's expected input shape
        image = expand_dims(image, axis=0)

        #Get the feature maps
        feature_maps = model2.predict(image)
        summed_feature_maps = [fmap_list.sum(axis=-1) for fmap_list in feature_maps]

        #Plot the feature maps
        fig = plt.figure(figsize=(15, 15))
        layer_index = 0

        for summed_fmap in summed_feature_maps:
            ax = plt.subplot(len(conv_layers), 1, layer_index + 1)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.imshow(summed_fmap[0, :, :], cmap='gray')

            layer_index += 1

        plt.show()
        return model, X_test, y_test

class EvaluateModel(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        model, X_test, y_test = X
        predicted_probabilities = model.predict(X_test)
        predicted_classes = np.argmax(predicted_probabilities, axis=1)

        if len(y_test.shape) > 1 and y_test.shape[1] > 1:
            y_test = np.argmax(y_test, axis=1)

        accuracy = accuracy_score(y_test, predicted_classes)
        precision = precision_score(y_test, predicted_classes, average='weighted')
        recall = recall_score(y_test, predicted_classes, average='weighted')
        f1 = f1_score(y_test, predicted_classes, average='weighted')
        #print(accuracy, precision, recall, f1)
        return accuracy, precision, recall, f1, model


pipeline_v1 = Pipeline([
    ('load_dataset', LoadDataset(train_database_path='../monorepo/Model/Datasets/lfw_augmented_dataset.db')),
    ('preprocess', Preprocess()),
    ('train_model', TrainModel()),
    ('visualize_feature_maps', VisualizeFeatureMaps()),
    ('evaluate_model', EvaluateModel())
])
# Run model_pipeline version 1 
#pipeline_v1 = pipeline_v1.transform(None)
#accuracy, precision, recall, f1, model  = pipeline_v1

# Transfer Learning Model (VGG16)
pipeline_v2 = Pipeline([
    ('load_dataset', LoadDataset(train_database_path='../monorepo/Model/Datasets/lfw_augmented_dataset.db')),
    ('preprocess', PreprocessVGG16()),
    ('train_model', TrainModelVGG16()),
    ('evaluate_model', EvaluateModel())
])
# Run model_pipeline version 2
#pipeline_v2 = pipeline_v2.transform(None)
#accuracy, precision, recall, f1, model  = pipeline_v2

# Transfer Learning Model (EfficientNet)
pipeline_v3 = Pipeline([
    ('load_dataset', LoadDataset(train_database_path='../monorepo/Model/Datasets/lfw_augmented_dataset.db')),
    ('preprocess', PreprocessEfficientNet()),
    ('train_model', TrainModelEfficientNet()),
    ('evaluate_model', EvaluateModel())
])
# Run model_pipeline version 3
#pipeline_v3 = pipeline_v3.transform(None)
#accuracy, precision, recall, f1, model  = pipeline_v3

''''
#save model
def save_trained_model(model, accuracy, precision, recall, f1):
    # Save the trained model
    save_path = "../monorepo/Model/model_registry/"
    timestamp = time.strftime("%Y%m%d%H%M%S")
    model.save(f'{save_path}model_version_{timestamp}.h5')
    # when the model is initally trained the way we load the model gives 0.0 for all evaluation metrics
    # therefore, we save the evaluation metrics in a json file
    
    evaluation_metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

    metrics_file_path = os.path.join(save_path, 'evaluation_metrics.json')
    with open(metrics_file_path, 'w') as metrics_file:
        json.dump(evaluation_metrics, metrics_file)

save_trained_model(model, accuracy, precision, recall, f1)
'''