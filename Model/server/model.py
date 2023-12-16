import io
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.utils import to_categorical
import os
import json
from mtcnn.mtcnn import MTCNN
import cv2
from contextlib import redirect_stdout
from Model import model_v1
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
    image = Column(String)
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
        result = detector.detect_faces(imageRGB)

    if result:
        # If faces were detected, take the first result
        bounding_box = result[0]['box']
        keypoints = result[0]['keypoints']

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

        resized_face = cv2.resize(cropped_face, (47, 62))

        processed_image = resized_face

    else:
        processed_image = None

    return processed_image


def predict(image_data):
    try:
        # Load the image and resize it to match the model's expected input shape
        img = Image.open(image_data)
        img_array = img_to_array(img)
        processed_img = preprocess_image(img_array)
        
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

            df = model_v1.load_dataset('lfw_augmented_dataset.db')[0]
            predicted_name = df['name'][np.where(df['target'].values == predicted_class)[0][0]] 
            print(predicted_name)
        else:
            predicted_name = "No faces detected."

        return predicted_name
    except Exception as e:
        return {"error": str(e)}

def retrain(datafile_path, test_size=0.2, random_state=42, epochs=10, batch_size=32):
    # Load the latest model
    latest_model = model_registry.get_latest_model_version()
    model = load_model(latest_model)

    # Load and split the new dataset
    df = model_v1.load_dataset(datafile_path)[0]
    print(df['image'])
    X_new, y_new = df['image'].values, df['target'].values
    X_new = np.array([np.array(img) for img in X_new])
    #X_new = X_new.reshape(-1, 11, 3)
    y_new = np.array(y_new)

    # Preprocess the labels
    y_new_categorical = to_categorical(y_new, num_classes=model.output_shape[1])
    print(X_new.shape)
    print(y_new_categorical.shape)

    accuracy, precision, recall, f1 = model_v1.evaluate_model(model, X_new, y_new)

    # when the model is initally trained the way we load the model gives 0.0 for all evaluation metrics
    # therefore, we save the evaluation metrics in a json file and load the data from there
    if accuracy == 0 and precision == 0 and recall == 0 and f1 == 0:
        metrics_file_path = 'Model/model_registry/evaluation_metrics.json'
        with open(metrics_file_path, 'r') as metrics_file:
            loaded_metrics = json.load(metrics_file)
        # Access the metrics
        accuracy = loaded_metrics['accuracy']
        precision = loaded_metrics['precision']
        recall = loaded_metrics['recall']
        f1 = loaded_metrics['f1_score']
    
    # Retrain the model on the new dataset
    model.fit(X_new, y_new_categorical, epochs=epochs, batch_size=batch_size, validation_split=test_size)

    # Save the retrained model
    timestamp = time.strftime("%Y%m%d%H%M%S")
    temp_model_path = f'Model/model_registry/retrained_model_version_{timestamp}.h5'
    model.save(temp_model_path)
    # Load the retrained model
    retrained_model = load_model(temp_model_path)
    old_model = load_model(latest_model)
    if retrained_model is None:
        print("Error: Unable to load the retrained model.")
    retrianed_accuracy, retrianed_precision, retrianed_recall, retrianed_f1 = model_v1.evaluate_model(retrained_model, X_new, y_new)
    
    print(accuracy, precision, recall, f1)
    print(retrianed_accuracy, retrianed_precision, retrianed_recall, retrianed_f1)

    if retrianed_accuracy > accuracy and retrianed_precision > precision and retrianed_recall > recall and retrianed_f1 > f1:
        print("retained model is better")
        save_path = "Model/model_registry/"
        timestamp = time.strftime("%Y%m%d%H%M%S")
        retrained_model.save(f'{save_path}model_version_{timestamp}.h5')
        os.remove(temp_model_path)
    else:
        print("older model is better")
        os.remove(temp_model_path)

    return retrianed_accuracy, retrianed_precision, retrianed_recall, retrianed_f1, accuracy, precision, recall, f1

''''
conn = sqlite3.connect('new_dataset.db')
cursor = conn.cursor()
cursor.execute(
    CREATE TABLE IF NOT EXISTS faces (
        id INTEGER PRIMARY KEY,
        target INTEGER,
        name TEXT NOT NULL,
        image BLOB NOT NULL
       )
   )
cursor.execute('DELETE FROM faces')
# Assuming you have a directory with images
image_directory = 'example_images'

# Get all files in the directory
image_files = [f for f in os.listdir(image_directory) if f.endswith('.jpg')]

# Iterate over image files
for image_file in image_files:
    image_path = os.path.join(image_directory, image_file)
    
    # Load and preprocess the image
    img = load_img(image_path, target_size=(62, 47))
    img_array = img_to_array(img)

    # Convert the image data to bytes
    image_data = pickle.dumps(img_array)

    # Split the filename to extract target and label
    target_name, _ = os.path.splitext(image_file)
    name, target_str = target_name.split('_')

    # Convert the target to an integer
    target = int(target_str)

    # Execute the INSERT statement
    cursor.execute("INSERT INTO faces (target, name, image) VALUES (?, ?, ?)", (target, name, image_data))

# Commit the changes and close the connection
conn.commit()
conn.close()

# Example usage


# Example usage:
image_path = '/Users/shahhdhassann/monorepo/Arturo_Gatti_0002.jpg'
image = cv2.imread(image_path)
prediction_result = predict(image)
'''

#retrain('retrain_dataset.db')