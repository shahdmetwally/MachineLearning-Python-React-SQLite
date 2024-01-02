import io
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.applications.efficientnet import preprocess_input
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import pickle
import sqlite3
from keras.models import Model
from keras.layers import Dense, BatchNormalization, GlobalAveragePooling2D, Reshape
import json
from mtcnn.mtcnn import MTCNN
import cv2
from contextlib import redirect_stdout
from Model.model_pipeline import (
    LoadDataset,
    EvaluateModel,
    PreprocessEfficientNet,
)
from PIL import Image
import time
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    DateTime,
    Sequence,
    BLOB,
    inspect,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import scoped_session, sessionmaker
from datetime import datetime
import Model.server.model_registry as model_registry
import cv2
import imutils
import random

# Create database to save all of the model's predictions
DATABASE_URL_PREDICTION = "sqlite:///./Model/Datasets/prediction_history.db"
Base1 = declarative_base()
engine_prediction = create_engine(DATABASE_URL_PREDICTION)
SessionLocalPrediction = scoped_session(
    sessionmaker(autocommit=False, autoflush=False, bind=engine_prediction)
)


class Prediction(Base1):
    __tablename__ = "predictions"

    id = Column(Integer, Sequence("prediction_id_seq"), primary_key=True, index=True)
    score = Column(String)
    image = Column(BLOB)
    created_at = Column(DateTime, default=datetime.utcnow)


# Check if the table exists
if not inspect(engine_prediction).has_table("predictions"):
    # Create the table if it doesn't exist
    Base1.metadata.create_all(bind=engine_prediction)

# Create database for retraining to save the user's feedback
DATABASE_URL_FEEDBACK = "sqlite:///./Model/Datasets/retrain_dataset.db"
Base2 = declarative_base()
engine_feedback = create_engine(DATABASE_URL_FEEDBACK)
SessionLocalFeedback = scoped_session(
    sessionmaker(autocommit=False, autoflush=False, bind=engine_feedback)
)

class Feedback(Base2):
    __tablename__ = "faces"

    target = Column(Integer, Sequence("feedback_id_seq"), index=True)
    name = Column(String)
    image = Column(BLOB, primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow)

# Check if the table exists
if not inspect(engine_prediction).has_table("faces"):
    # Create the table if it doesn't exist
    Base2.metadata.create_all(bind=engine_feedback)


def preprocess_image(image):
    detector = MTCNN()

    with redirect_stdout(io.StringIO()):
        # image = img * 255
        # image = cv2.imread(image_path)
        imageRGB = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(imageRGB)
        # Only save faces that are bigger than the min_face_size
        min_face_size = 50
        faces = [
            face
            for face in faces
            if face["box"][2] > min_face_size and face["box"][3] > min_face_size
        ]

    if faces:
        # if there are multiple faces detected, consider the face with the largest area
        largest_face = max(faces, key=lambda x: x["box"][2] * x["box"][3])
        bounding_box = largest_face["box"]
        keypoints = largest_face["keypoints"]
        # If faces were detected, take the first result

        # Extract the coordinates of relevant facial keypoints
        left_eye = keypoints["left_eye"]
        right_eye = keypoints["right_eye"]
        # Calculate the angle of rotation for alignment
        eyes_center = (
            (left_eye[0] + right_eye[0]) / 2,
            (left_eye[1] + right_eye[1]) / 2,
        )
        angle_rad = np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])
        angle_deg = np.degrees(angle_rad)

        # Perform rotation and alignment
        M = cv2.getRotationMatrix2D(eyes_center, angle_deg, 1)
        aligned_face = cv2.warpAffine(
            imageRGB, M, (imageRGB.shape[1], imageRGB.shape[0])
        )

        # Extract the bounding box coordinates
        x, y, w, h = bounding_box

        # Crop the aligned face from the original image
        cropped_face = aligned_face[y : y + h, x : x + w]

        resized_face = cv2.resize(cropped_face, (224, 224))

        processed_image = resized_face
        processed_image = preprocess_input(processed_image)

    else:
        processed_image = None
        bounding_box = None

    return processed_image, bounding_box


def predict(image_data):
    try:
        # Load the image and resize it to match the model's expected input shape
        image = cv2.imread(image_data)
        img_array = img_to_array(image)
        processed_img, bounding_box = preprocess_image(img_array)

        if processed_img is not None:
            img_array = img_to_array(processed_img)

            # Add an extra dimension for the batch
            img_array = np.expand_dims(img_array, axis=0)

            # Load the latest trained model
            latest_model = model_registry.get_latest_model_version()
            loaded_model = load_model(latest_model)

            # Make predictions
            predictions = loaded_model.predict(img_array)

            # Get the predicted class
            predicted_class = int(np.argmax(predictions))
            max_confidence = np.max(predictions)

            threshold = 0.8
            if max_confidence >= threshold:
                loader = LoadDataset(
                    train_database_path="Model/Datasets/retrain_dataset.db"
                )
                data = None
                (
                    X_train,
                    X_val,
                    X_test,
                    y_train,
                    y_val,
                    y_test,
                    num_classes,
                    df,
                ) = loader.transform(data)
                predicted_name = df["name"][
                    np.where(df["target"].values == predicted_class)[0][0]
                ]
                # predicted_name += " with " + str(max_confidence) + " confidence"

            else:
                predicted_name = "Unknown"

        else:
            predicted_name = "No faces detected."

        return predicted_name, bounding_box
    except Exception as e:
        return {"error": str(e)}


# If retrain_dataset has 200 targets then initiate retraining (change threshold value if more than 10 makes more sense)
def trigger_retraining(datafile_path, threshold=200, **retrain_args):
    # Load and split the retraining dataset
    conn = sqlite3.connect(datafile_path)

    # Query to select all records from the faces table
    query = "SELECT * FROM faces"

    # Fetch records from the database into a Pandas DataFrame
    df = pd.read_sql_query(query, conn)
    # Close the database connection
    conn.close()
    
    # Set values for X_train and y_train
    y_train = df["target"].values

    num_entries = len(np.unique(y_train))

    if num_entries >= threshold:
        # Trigger retraining
        print(f"Triggering retraining with {num_entries} entries.")
        retrain(datafile_path, **retrain_args)
    else:
        print(f"Not enough entries ({num_entries}) to trigger retraining.")


def retrain(datafile_path):
    # Load the latest model
    latest_model = model_registry.get_latest_model_version()
    old_model = load_model(latest_model)

    # Get latest model's evaluation metrics
    metrics_file_path = "Model/model_registry/evaluation_metrics.json"
    with open(metrics_file_path, "r") as metrics_file:
        loaded_metrics = json.load(metrics_file)
    # Access the metrics
    accuracy = loaded_metrics["accuracy"]
    precision = loaded_metrics["precision"]
    recall = loaded_metrics["recall"]
    f1 = loaded_metrics["f1"]

    # Load and split the new dataset
    conn = sqlite3.connect(datafile_path)

    # Query to select all records from the faces table
    query = "SELECT * FROM faces"

    # Fetch records from the database into a Pandas DataFrame
    df = pd.read_sql_query(query, conn)
    df["image"] = df["image"].apply(lambda x: np.array(pickle.loads(x)))
    # Close the database connection
    conn.close()
    
    # Set values for X_train and y_train
    X_train, y_train = df["image"].values, df["target"].values

    # Get the number of unique classes
    unique_classes = np.unique(y_train)
    class_label_mapping = {label: idx for idx, label in enumerate(unique_classes)}
    y_train_mapped = np.array([class_label_mapping[label] for label in y_train])
    num_classes = len(np.unique(y_train_mapped))

    # Split the dataset into X_train/y_train and X_temp/y_temp (which will be split into validation and test set)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_train, y_train_mapped, test_size=0.4, random_state=42
    )

    # Split the temp set into validation and test sets
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    # Preprocess the data
    preprocess = PreprocessEfficientNet()
    data = X_train, X_val, X_test, y_train, y_val, y_test, num_classes, df
    X_train, X_val, X_test, y_train, y_val, y_test, num_classes = preprocess.transform(
        data
    )

    # Define functions for individual augmentations
    def apply_rotation(image):
        # Apply rotation to the image
        imagebyte = (image).astype('uint8')
        # Apply dynamic rotation to the original image
        rotation_angle = random.uniform(-15, -5) if random.choice([True, False]) else random.uniform(5, 15)
        rotated_image = imutils.rotate(imagebyte, angle=rotation_angle)
        return rotated_image
    
    def apply_edge_enhancement(image):
        # Convert the image to uint8 format
        imagebyte = (image).astype('uint8')

        # Apply Sobel filter for edge enhancement
        sobel_x = cv2.Sobel(imagebyte, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(imagebyte, cv2.CV_64F, 0, 1, ksize=3)

        # Combine the gradient images
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

        # Ensure the enhanced image is in the range [0, 255]
        enhanced_image = np.clip((imagebyte + 0.25 * gradient_magnitude), 0, 255).astype(np.uint8)

        return enhanced_image
    
    def apply_flip(image):
        # Apply horizontal flip to the image
        imagebyte = (image).astype('uint8')
        return np.fliplr(imagebyte)
    
    def apply_color_jittering(image):
        # Convert the image to uint8 format
        imagebyte = (image).astype('uint8')

        # Convert to HSV color space
        hsv_image = cv2.cvtColor(imagebyte, cv2.COLOR_RGB2HSV)

        # Apply color jittering to the HSV image
        # Adjust brightness
        brightness_factor = random.uniform(0.5, 1.5)
        hsv_image[..., 2] = cv2.multiply(hsv_image[..., 2], brightness_factor)

        # Adjust contrast
        contrast_factor = random.uniform(0.5, 1.5)
        hsv_image[..., 1] = cv2.multiply(hsv_image[..., 1], contrast_factor)

        # Adjust hue
        hue_factor = random.uniform(-10, 10)
        hsv_image[..., 0] = (hsv_image[..., 0] + hue_factor) % 180

        # Adjust saturation
        saturation_factor = random.uniform(0.5, 1.5)
        hsv_image[..., 1] = cv2.multiply(hsv_image[..., 1], saturation_factor)

        # Convert back to RGB
        color_jittered_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

        return color_jittered_image

    # Dictionary to store augmented images based on labels
    augmented_data = {}

    # Iterate over each image in X_train and apply augmentations
    for i in range(len(X_train)):
        image = X_train[i]
        label = y_train[i]

        # Convert label to a hashable type, e.g., tuple
        label_key = tuple(label)

        # Apply rotation
        rotated_image = apply_rotation(image)

        # Apply flip
        flipped_image = apply_flip(image)

        # Apply enhancement
        enhanced_image = apply_edge_enhancement(image)

        #Apply color jittering
        jittered_image = apply_color_jittering(image)

        # Store the augmented images in the dictionary
        if label_key not in augmented_data:
            augmented_data[label_key] = []

        augmented_data[label_key].extend([image, rotated_image, flipped_image, enhanced_image, jittered_image])

    # Make sure all arrays contain the same number of samples
    min_samples = min(len(samples) for samples in augmented_data.values())
    augmented_data = {key: np.array(samples[:min_samples]) for key, samples in augmented_data.items()}      
    
    # Initialize lists to store augmented images and labels
    augmented_X_train = []
    augmented_y_train = []

    # Iterate over each image in X_train and apply augmentations
    for i in range(len(X_train)):
        image = X_train[i]
        label = y_train[i]

        # Original image
        augmented_X_train.append(image)
        augmented_y_train.append(label)

        # Apply rotation
        rotated_image = apply_rotation(image)
        augmented_X_train.append(rotated_image)
        augmented_y_train.append(label)

        # Apply flip
        flipped_image = apply_flip(image)
        augmented_X_train.append(flipped_image)
        augmented_y_train.append(label)

        # Apply edge enhancement
        enhanced_image = apply_edge_enhancement(image)
        augmented_X_train.append(enhanced_image)
        augmented_y_train.append(label)

    # Convert lists to numpy arrays
    augmented_X_train = np.array(augmented_X_train)
    augmented_y_train = np.array(augmented_y_train)

    #Save X_test and y_test to have the X_train and y_train that are not augmented
    X_test = np.concatenate([X_train, X_test])
    y_test =  np.concatenate([y_train, y_test])

    #Split the new X_test and y_test images 50% for testing data and the validation data
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, test_size=0.5, random_state=42
    )
   
    # Unfreeze the layers of old model for retraining
    for layer in old_model.layers:
        layer.trainable = False

    # Get current time as a string to name the layers uniquely 
    current_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    # Add a new layers for retraining using the old model layers
    x = old_model.output
    x = Dense(128, activation="relu", name="new_dense_1" + current_time)(x)
    x = Reshape((1, 1, 128))(x)
    x = GlobalAveragePooling2D(name="new_pooling_" + current_time)(x)
    x = BatchNormalization(name="new_" + current_time)(x)
    shape = y_train.shape
    predictions = Dense(shape[1], activation="softmax", name="new_dense_2" + current_time)(x)

    # Create the new model
    new_model = Model(inputs=old_model.input, outputs=predictions)

    new_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    # Fit the model with the updated input shape
    new_model.fit(augmented_X_train, augmented_y_train, epochs=7, batch_size=5, validation_data=(X_val, y_val))

    # Save the retrained model
    timestamp = time.strftime("%Y%m%d%H%M%S")
    retrained_model_path = "Model/model_registry/"
    new_model.save(f"{retrained_model_path}model_version_{timestamp}.h5")

    # Load the retrained model
    latest_model = model_registry.get_latest_model_version()
    retrained_model = load_model(latest_model)

    if retrained_model is None:
        print("Error: Unable to load the retrained model.")

    # Evaluate retrained model
    evaluate = EvaluateModel()
    retrained_evaluate_data = retrained_model, X_test, y_test
    (
        retrained_accuracy,
        retrained_precision,
        retrained_recall,
        retrained_f1,
        m,
    ) = evaluate.transform(retrained_evaluate_data)

    print("old model performance: ", accuracy, precision, recall, f1)
    print(
        "retrained model performance: ",
        retrained_accuracy,
        retrained_precision,
        retrained_recall,
        retrained_f1,
    )

    # Compare retrained model's performance to the old model's performance
    if (
        retrained_accuracy > accuracy
        and retrained_precision > precision
        and retrained_recall > recall
        and retrained_f1 > f1
    ):
        print("retrained model is better")
        # Save retrained model's evaluation metrics
        evaluation_metrics = {
            "model": latest_model,
            "accuracy": retrained_accuracy,
            "precision": retrained_precision,
            "recall": retrained_recall,
            "f1": retrained_f1,
        }
        metrics_file_path = os.path.join(
            retrained_model_path, "evaluation_metrics.json"
        )
        with open(metrics_file_path, "w") as metrics_file:
            json.dump(evaluation_metrics, metrics_file)
    else:
        print("older model is better")
        # Don't save retrained model
        os.remove(latest_model)
    return (
        retrained_accuracy,
        retrained_precision,
        retrained_recall,
        retrained_f1,
        accuracy,
        precision,
        recall,
        f1,
    )

trigger_retraining('./Model/Datasets/retrain_dataset.db')