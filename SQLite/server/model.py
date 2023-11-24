import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
import numpy as np
from mtcnn.mtcnn import MTCNN
import cv2
import numpy as np
from contextlib import redirect_stdout
import io

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


# Example usage:
image_path = '/Users/shahd.metwally/monorepo/SQLite/Arturo_Gatti_0002.jpg'
prediction_result = predict(image_path)
