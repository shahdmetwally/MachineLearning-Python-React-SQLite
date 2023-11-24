import io
import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
from keras.utils import to_categorical
import sqlite3
import pickle
import os
from mtcnn.mtcnn import MTCNN
import cv2
from contextlib import redirect_stdout
import io
from . import model_v1

def preprocess_image(image_path):
    detector = MTCNN()

    with redirect_stdout(io.StringIO()):
        #image = img * 255
        image = cv2.imread(image_path)
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


def predict(image_path):
    # Load the image and resize it to match the model's expected input shape
    img = preprocess_image(image_path)
    img_array = image.img_to_array(img)
    # Add an extra dimension for the batch
    img_array = np.expand_dims(img_array, axis=0)

    # Preprocess the image for the model
    img_array = preprocess_input(img_array)

    # Load the trained model
    loaded_model = tf.keras.models.load_model('SQLite/trained_model.h5')

    # Make predictions
    predictions = loaded_model.predict(img_array)
    # print(predictions)

    # Get the predicted class
    predicted_class = np.argmax(predictions)
    print(predicted_class)

    # Get the result
    # result = predictions.flatten()[0]
    # print(result)

    return predicted_class

def retrain(datafile_path, test_size=0.2, random_state=42, epochs=10, batch_size=32):
    # Load the existing model
    model = load_model('SQLite/trained_model.h5')

    # Load and split the new dataset
    X_new, y_new, names = model_v1.load_and_split_dataset(datafile_path)
    X_new = np.array(X_new)
    y_new = np.array(y_new)

    # Preprocess the labels
    y_new_categorical = to_categorical(y_new, num_classes=model.output_shape[1])

    print(X_new.shape)
    print(y_new_categorical.shape)

    # Retrain the model on the new dataset
    model.fit(X_new, y_new_categorical, epochs=epochs, batch_size=batch_size, validation_split=test_size)

    # Save the updated model
    model.save('updated_model.h5')

    return model

conn = sqlite3.connect('new_dataset.db')
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS faces (
        id INTEGER PRIMARY KEY,
        target INTEGER,
        name INTEGER NOT NULL,
        image BLOB NOT NULL
       )
   ''')
cursor.execute('DELETE FROM faces')
# Assuming you have a directory with images
image_directory = 'SQLite/example_images'

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
    target_label, _ = os.path.splitext(image_file)
    target, label = map(int, target_label.split('_'))

    # Execute the INSERT statement
    cursor.execute("INSERT INTO faces (target, name, image) VALUES (?, ?, ?)", (target, label, image_data))

    

# Commit the changes and close the connection
conn.commit()
conn.close()
# Example usage
retrain('new_dataset.db')


# Example usage:
image_path = '/Users/shahd.metwally/monorepo/SQLite/Arturo_Gatti_0002.jpg'
prediction_result = predict(image_path)
