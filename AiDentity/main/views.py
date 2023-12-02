from django.shortcuts import render
from .forms import FaceForm
from .models import Prediction

# from main.utils import load_trained_model

import os
from AiDentity.settings import BASE_DIR
from keras.applications.vgg16 import preprocess_input
from keras.models import load_model
from PIL import Image
import numpy as np
import pandas as pd
import io
from io import BytesIO
from mtcnn.mtcnn import MTCNN
import cv2
from contextlib import redirect_stdout
from model_v1 import load_dataset


model_file = os.path.join(BASE_DIR, "trained_model_v2_augmented.h5")
learning_model = load_model(model_file)


def preprocess_image(image):
    detector = MTCNN()

    # Read the image file from the InMemoryUploadedFile
    image_data = image.read()
    img = Image.open(BytesIO(image_data))

    with redirect_stdout(io.StringIO()):
        # image = img * 255
        # image = cv2.imread(image_path)
        imageRGB = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
        result = detector.detect_faces(imageRGB)

    if result:
        # If faces were detected, take the first result
        bounding_box = result[0]["box"]
        keypoints = result[0]["keypoints"]

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

        resized_face = cv2.resize(cropped_face, (47, 62))

        processed_image = resized_face

    return processed_image


def predict(img_array):
    try:
        # Add an extra dimension for the batch
        img_array = np.expand_dims(img_array, axis=0)

        # Preprocess the image for the model
        img_array = preprocess_input(img_array)

        # Load the trained model
        loaded_model = load_model("trained_model_v2_augmented.h5")

        # Make predictions
        predictions = loaded_model.predict(img_array)

        # Get the predicted class
        predicted_class = int(np.argmax(predictions))

        df = load_dataset("lfw_dataset.db")[0]

        # Print variable values for debugging
        print("predicted_class:", predicted_class)
        print("df columns:", df.columns)
        # print("df head:", df.head())

        predicted_name = df["name"][
            np.where(df["target"].values == predicted_class)[0][0]
        ]

        return predicted_name
    except Exception as e:
        return {"error": str(e)}


def home(request):
    # Initialize predictions outside the conditional block
    predictions = Prediction.objects.all()

    if request.method == "POST":
        form = FaceForm(request.POST, request.FILES)
        if form.is_valid():
            # Get the uploaded image
            image = form.cleaned_data["image"]

            # Perform preprocessing
            processed_image = preprocess_image(image)

            # Make predictions using your face recognition model
            predicted_name = predict(processed_image)

            # Save prediction to the database
            add_prediction_data = Prediction(image=image, name=predicted_name)
            add_prediction_data.save()

            # update prediction list
            predictions = Prediction.objects.all()

            context = {"form": form, "name": predicted_name, "predictions": predictions}

            # Render the prediction results
            return render(request, "home.html", context)
    else:
        form = FaceForm()
        return render(request, "home.html", {"form": form, "predictions": predictions})


def history(request):
    return render(request, "history.html")


def about(request):
    return render(request, "about.html")
