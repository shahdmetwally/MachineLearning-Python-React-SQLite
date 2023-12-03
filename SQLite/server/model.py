import io
import numpy as np
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
from keras.utils import to_categorical
import sqlite3
import pickle
import os
from mtcnn.mtcnn import MTCNN
import cv2
from contextlib import redirect_stdout
from SQLite import model_v1
from PIL import Image
from pathlib import Path
import time

def get_latest_model_version():
    model_dir = 'SQLite/model_registry'  # Replace with the actual path to your model directory
    model_files = Path(model_dir).glob('model_version_*.h5')
    
    # Extract timestamps and sort the files based on the timestamp
    sorted_models = sorted(model_files, key=lambda f: os.path.getmtime(f), reverse=True)
    
    if sorted_models:
        latest_version = sorted_models[0]
        return str(latest_version)
    else:
        return None  # No model versions found

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

    return processed_image


def predict(image_data):
    try:
        # Load the image and resize it to match the model's expected input shape
        img = Image.open(image_data)
        img_array = img_to_array(img)
        processed_img = preprocess_image(img_array)
        #print(type(processed_img))
        img_array = img_to_array(processed_img)
        #img_array = cv2.resize(img_array, (47, 62))
        # Add an extra dimension for the batch
        img_array = np.expand_dims(img_array, axis=0)

        # Preprocess the image for the model
        img_array = preprocess_input(img_array)

        # Load the latest trained model
        latest_model = get_latest_model_version()
        loaded_model = load_model(latest_model)

        # Make predictions
        predictions = loaded_model.predict(img_array)

        # Get the predicted class
        predicted_class = int(np.argmax(predictions))

        df = model_v1.load_dataset('lfw_dataset.db')[0]
        predicted_name = df['name'][np.where(df['target'].values == predicted_class)[0][0]] 

        return predicted_name
    except Exception as e:
        return {"error": str(e)}

def retrain(datafile_path, test_size=0.2, random_state=42, epochs=10, batch_size=32):
    # Load the latest model
    latest_model = get_latest_model_version()
    model = load_model(latest_model)

    # Load and split the new dataset
    df = model_v1.load_dataset(datafile_path)[0]
    X_new, y_new = df['image'].values, df['target'].values
    X_new = np.array([np.array(img) for img in X_new])
    y_new = np.array(y_new)

    # Preprocess the labels
    y_new_categorical = to_categorical(y_new, num_classes=model.output_shape[1])

    accuracy, precision, recall, f1 = model_v1.evaluate_model(model, X_new, y_new)

    print(X_new.shape)
    print(y_new_categorical.shape)
    

    # Retrain the model on the new dataset
    model.fit(X_new, y_new_categorical, epochs=epochs, batch_size=batch_size, validation_split=test_size)

    # Save the retrained model
    model.save('retrained_model.h5')
    # Load the retrained model
    retrained_model = load_model('retrained_model.h5')
    old_model = load_model(latest_model)
    if retrained_model is None:
        print("Error: Unable to load the retrained model.")
    retrianed_accuracy, retrianed_precision, retrianed_recall, retrianed_f1 = model_v1.evaluate_model(retrained_model, X_new, y_new)

    if retrianed_accuracy > accuracy or retrianed_precision > precision or retrianed_recall > recall or retrianed_f1 > f1:
        print("retained model is better")
        save_path = "SQLite/model_registry"
        timestamp = time.strftime("%Y%m%d%H%M%S")
        retrained_model.save(f'{save_path}model_version_{timestamp}.h5')
        return retrained_model, retrianed_accuracy, retrianed_precision, retrianed_recall, retrianed_f1
    else:
        print("older model is better")
        return old_model, retrianed_accuracy, retrianed_precision, retrianed_recall, retrianed_f1



conn = sqlite3.connect('new_dataset.db')
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS faces (
        id INTEGER PRIMARY KEY,
        target INTEGER,
        name TEXT NOT NULL,
        image BLOB NOT NULL
       )
   ''')
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
retrain('new_dataset.db')
# Example usage:
image_path = '/Users/shahd.metwally/monorepo/SQLite/Arturo_Gatti_0002.jpg'
image = cv2.imread(image_path)
prediction_result = predict(image)
