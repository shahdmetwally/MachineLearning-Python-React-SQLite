from fastapi import File, UploadFile, HTTPException, Form
from pydantic import BaseModel
import Model.server.model as model
from fastapi.responses import JSONResponse
from pathlib import Path
import io
from sqlalchemy.orm import Session
import pickle
import numpy as np
from keras.preprocessing.image import img_to_array
from PIL import Image
import base64
import os
import random
from fastapi import APIRouter
from .app import app

#All done by: Shahd

router = APIRouter()


class PredictionData(BaseModel):
    image: UploadFile


def save_prediction_to_db(db: Session, score: str, image: bytes):
    db_prediction = model.Prediction(score=score, image=image)
    db.add(db_prediction)
    db.commit()
    db.refresh(db_prediction)
    return db_prediction

def save_feedback_to_db(db: Session, name: str, image: str, target: int):
    # Read image file
    image_data = image.file.read()
    # Load and preprocess the image
    img = Image.open(io.BytesIO(image_data))
    img = img.resize((224, 224))
    img_array = (img_to_array(img) * 255 / 255.0).astype('uint8')

    # Convert the preprocessed image data to bytes using pickle
    preprocessed_image_bytes = pickle.dumps(img_array)

    db_feedback = model.Feedback(target=target, name=name, image=preprocessed_image_bytes)
    db.add(db_feedback)
    db.commit()
    db.refresh(db_feedback)
    return db_feedback


# user should be able to upload an image which should initiate a prediction
@app.post("/predict")
def predict(image: UploadFile = File(...)):
    try:
        # Ensure that the uploaded file is an image
        if not image.filename.lower().endswith((".png", ".jpg", ".jpeg")):
            raise HTTPException(
                status_code=400,
                detail="Invalid file format. Supported formats: ['.png', '.jpg', '.jpeg']",
            )

        # Save the uploaded image to a temporary file
        temp_image_path = Path("temp_image.jpg")
        with open(temp_image_path, "wb") as temp_file:
            temp_file.write(image.file.read())

        # Read image file as bytes
        with open(temp_image_path, "rb") as image_file:
            image_data = image_file.read()

        # Convert temp_image_path to str instead of Path object and make predictions
        temp_image_path = str(temp_image_path)
        prediction_result, bounding_box = model.predict(temp_image_path)

        # Storing prediction in database
        db = model.SessionLocalPrediction()
        db_prediction = save_prediction_to_db(db, prediction_result, image_data)
        db.close()
        os.remove(temp_image_path)
        return JSONResponse(
            content={
                "message": "Prediction successful",
                "score": prediction_result,
                "box": bounding_box,
            }
        )
    except Exception as e:
        # Return other exception details in the response
        return JSONResponse(content={"error": str(e)})


# user should be able to view the history of all previously predicitions
@app.get("/predictions")
def get_predictions():
    db = model.SessionLocalPrediction()
    predictions = db.query(model.Prediction).all()
    db.close()

    # Convert binary image data to base64
    for prediction in predictions:
        prediction.image = base64.b64encode(prediction.image).decode("utf-8")

    return predictions


# ask the user for feedback on the prediction's validity in order to use the false predictions for automated retraining
@app.post("/feedback")
def submit_feedback(
    image: UploadFile = File(...),
    is_correct: bool = Form(...),
    user_name: str = Form(""),
):
    try:
        feedback_entry = {
            "userName": user_name,
            "is_correct": is_correct,
            "image": image,
            "target": random.randint(158, 200)
        }

        db = model.SessionLocalFeedback()
        db_feedback = save_feedback_to_db(db, user_name, image, random.randint(158, 500))
        db.close()

        return {"message": "Feedback saved successfully"}
    except Exception as e:
        # Handle exceptions as needed
        raise HTTPException(status_code=500, detail=str(e))
