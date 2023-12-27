import io
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.utils import to_categorical
from keras.applications.efficientnet import preprocess_input
import os
import json
from mtcnn.mtcnn import MTCNN
import cv2
from contextlib import redirect_stdout
from Model.model_pipeline import LoadDataset, EvaluateModel
from PIL import Image
import time
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Sequence, BLOB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import scoped_session, sessionmaker
from datetime import datetime
import Model.server.model_registry as model_registry

DATABASE_URL_PREDICTION = "sqlite:///./prediction_history.db"
Base1 = declarative_base()
engine_prediction = create_engine(DATABASE_URL_PREDICTION)
SessionLocalPrediction = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine_prediction))

class Prediction(Base1):
    __tablename__ = "predictions"

    id = Column(Integer, Sequence("prediction_id_seq"), primary_key=True, index=True)
    score = Column(Integer)
    image = Column(BLOB)
    created_at = Column(DateTime, default=datetime.utcnow)

# Create the table in the database
Base1.metadata.create_all(bind=engine_prediction)

DATABASE_URL_FEEDBACK = "sqlite:///./retrain_dataset.db"
Base2 = declarative_base()
engine_feedback = create_engine(DATABASE_URL_FEEDBACK)
SessionLocalFeedback = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine_feedback))

class Feedback(Base2):
    __tablename__ = "faces"

    target= Column(Integer, Sequence("feedback_id_seq"), primary_key=True, index=True)
    name = Column(String)
    image = Column(BLOB)
    created_at = Column(DateTime, default=datetime.utcnow)

# Create the table in the database
Base2.metadata.create_all(bind=engine_feedback)

def preprocess_image(image):
    detector = MTCNN()

    with redirect_stdout(io.StringIO()):
        #image = img * 255
        #image = cv2.imread(image_path)
        imageRGB = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(imageRGB)
        #Only save faces that are bigger than the min_face_size
        min_face_size = 50
        faces = [face for face in faces if face['box'][2] > min_face_size and face['box'][3] > min_face_size]
        
    if faces:
        #if there are multiple faces detected, consider the face with the largest area
        largest_face = max(faces, key=lambda x: x['box'][2] * x['box'][3])
        bounding_box = largest_face['box']
        keypoints = largest_face['keypoints']
        # If faces were detected, take the first result

        # Extract the coordinates of relevant facial keypoints
        left_eye = keypoints['left_eye']
        right_eye = keypoints['right_eye']
        # Calculate the angle of rotation for alignment
        eyes_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
        angle_rad = np.arctan2(
        right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])
        angle_deg = np.degrees(angle_rad)

        # Perform rotation and alignment
        M = cv2.getRotationMatrix2D(eyes_center, angle_deg, 1)
        aligned_face = cv2.warpAffine(imageRGB, M, (imageRGB.shape[1], imageRGB.shape[0]))

        # Extract the bounding box coordinates
        x, y, w, h = bounding_box

        # Crop the aligned face from the original image
        cropped_face = aligned_face[y:y+h, x:x+w]

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
                loader = LoadDataset(train_database_path='lfw_augmented_dataset.db')
                data = None
                X_train, X_val, X_test, y_train, y_val, y_test, num_classes, df = loader.transform(data)
                predicted_name = df['name'][np.where(df['target'].values == predicted_class)[0][0]] 
                #predicted_name += " with " + str(max_confidence) + " confidence"
                
            else:
                predicted_name = "Unknown"

        else:
            predicted_name = "No faces detected."

        return predicted_name, bounding_box
    except Exception as e:
        return {"error": str(e)}

# If retrain_dataset has 10 false predictions then initiate retraining (change threshold value if more than 10 makes more sense)
def trigger_retraining(datafile_path, threshold=10, **retrain_args):
    loader = LoadDataset(train_database_path='lfw_augmented_dataset.db')
    data = None
    X_train, X_val, X_test, y_train, y_val, y_test, num_classes, df = loader.transform(data)

    num_entries = len(df)

    if num_entries >= threshold:
        # Trigger retraining
        print(f"Triggering retraining with {num_entries} entries.")
        retrain(datafile_path, **retrain_args)
    else:
        print(f"Not enough entries ({num_entries}) to trigger retraining.")


def retrain(datafile_path, test_size=0.2, random_state=42, epochs=10, batch_size=32):
    # Load the latest model
    latest_model = model_registry.get_latest_model_version()
    model = load_model(latest_model)

    # Load and split the new dataset
    loader = LoadDataset(train_database_path=datafile_path)
    data = None
    X_train, X_val, X_test, y_train, y_val, y_test, num_classes, df = loader.transform(data)
    print(df['image'])
    X_new, y_new = df['image'].values, df['target'].values

    # Resize images to (224, 224)
    X_new = [cv2.resize(img, (224, 224)) for img in X_new]

    X_new = np.array([np.array(img) for img in X_new])
    y_new = np.array(y_new)

    # Preprocess the labels
    y_new_categorical = to_categorical(y_new, num_classes=model.output_shape[1])
    print(X_new.shape)
    print(y_new_categorical.shape)

    # Get latest model's evaluation metrics
    metrics_file_path = 'Model/model_registry/evaluation_metrics.json'
    with open(metrics_file_path, 'r') as metrics_file:
        loaded_metrics = json.load(metrics_file)
    # Access the metrics
    accuracy = loaded_metrics['accuracy']
    precision = loaded_metrics['precision']
    recall = loaded_metrics['recall']
    f1 = loaded_metrics['f1']
    
    # Retrain the model on the new dataset
    model.fit(X_new, y_new_categorical, epochs=epochs, batch_size=batch_size, validation_split=test_size)

    # Save the retrained model
    timestamp = time.strftime("%Y%m%d%H%M%S")
    retrained_model_path = "Model/model_registry/"
    model.save(f'{retrained_model_path}model_version_{timestamp}.h5')

    # Load the retrained model
    latest_model = model_registry.get_latest_model_version()
    retrained_model = load_model(latest_model)

    if retrained_model is None:
        print("Error: Unable to load the retrained model.")
    
    # Evaluate retrained model
    evaluate = EvaluateModel()
    retrained_evaluate_data = retrained_model, X_new, y_new
    retrianed_accuracy, retrianed_precision, retrianed_recall, retrianed_f1, m = evaluate.transform(retrained_evaluate_data)
    
    print("old model performance: ", accuracy, precision, recall, f1)
    print("retrained model performance: ",retrianed_accuracy, retrianed_precision, retrianed_recall, retrianed_f1)

    # Compare retrained model's performance to the old model's performance
    if retrianed_accuracy > accuracy and retrianed_precision > precision and retrianed_recall > recall and retrianed_f1 > f1:
        print("retained model is better")
        # Save retrained model's evaluation metrics
        evaluation_metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
        }
        metrics_file_path = os.path.join(retrained_model_path, 'evaluation_metrics.json')
        with open(metrics_file_path, 'w') as metrics_file:
            json.dump(evaluation_metrics, metrics_file)
    else:
        print("older model is better")
        # Don't save retrained model
        os.remove(latest_model)

    return retrianed_accuracy, retrianed_precision, retrianed_recall, retrianed_f1, accuracy, precision, recall, f1
''''
# Example usage:
image_path = '/Users/shahhdhassann/monorepo/Arturo_Gatti_0002.jpg'
prediction_result = predict(image_path)
print(prediction_result)
'''

# trigger for retraining using retrain_dataset.db which contains the images and correct predictions of previously false predictions made by our model
#trigger_retraining('retrain_dataset.db')

retrain('retrain_dataset.db')