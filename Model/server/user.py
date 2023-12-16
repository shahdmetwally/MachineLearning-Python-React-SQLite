from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from pydantic import BaseModel
import Model.server.model as model
from fastapi.responses import JSONResponse, FileResponse
from pathlib import Path
import io
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import pickle
from keras.preprocessing.image import img_to_array
from PIL import Image

app = FastAPI()
origins = ["http://localhost:3000", " http://localhost:3000/"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionData(BaseModel):
    image: UploadFile


def save_prediction_to_db(db: Session, score: int, image: str):
    db_prediction = model.Prediction(score=score, image=image)
    db.add(db_prediction)
    db.commit()
    db.refresh(db_prediction)
    return db_prediction

def save_feedback_to_db(db: Session, name: str, image: str):
    # Read image file
    image_data = image.file.read()

    # Convert the image data to bytes using pickle
    image_bytes = pickle.dumps(image_data)

    # Load and preprocess the image
    img = Image.open(io.BytesIO(image_data))
    img = img.resize((62, 47))  # Resize the image to the target size
    img_array = img_to_array(img)

    # Convert the preprocessed image data to bytes using pickle
    preprocessed_image_bytes = pickle.dumps(img_array)

    db_feedback = model.Feedback(name=name, image=preprocessed_image_bytes)
    db.add(db_feedback)
    db.commit()
    db.refresh(db_feedback)
    return db_feedback

# user should be able to upload an image which should initiate a prediction 
@app.post('/predict')
def predict(image: UploadFile = File(...)):
    try:
       # Ensure that the uploaded file is an image
        if not image.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            raise HTTPException(status_code=400, detail="Invalid file format. Supported formats: ['.png', '.jpg', '.jpeg']")

        # Save the uploaded image to a temporary file
        temp_image_path = Path(f"Model/user_images/temp_{image.filename}")
        with open(temp_image_path, "wb") as temp_file:
            temp_file.write(image.file.read())
            #temp_file.close()
        prediction_result = model.predict(temp_image_path)
        image_str = temp_image_path.name
        # Storing result in database
        db = model.SessionLocalPrediction()
        db_prediction = save_prediction_to_db(db, prediction_result, image_str)
        db.close()

        return JSONResponse(content={"message": "Prediction successful", "score": prediction_result, "image": image_str})
    except Exception as e:
        # Return other exception details in the response
        return JSONResponse(content={"error": str(e)})

# user should be able to view the history of all previously predicitions
@app.get('/predictions')
def get_predictions():
    db = model.SessionLocalPrediction()
    predictions = db.query(model.Prediction).all()
    db.close()
    return predictions

#user should be able to view the image they uploaded
@app.get("/get_image/{image_name}")
async def get_image(image_name: str):
    image_dir = Path("Model/user_images/")
    image_path = image_dir / image_name

    if not image_path.is_file():
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(image_path, media_type="image/jpg")

# ask the user for feedback on the prediction's validity in order to use the false predictions for automated retraining
@app.post('/feedback')
def submit_feedback(
    image: UploadFile = File(...),
    is_correct: bool = Form(...),
    user_name: str = Form(''),
):
    try:
        feedback_entry = {
            "userName": user_name,
            "is_correct": is_correct,
            "image": image,
        }

        db = model.SessionLocalFeedback()
        db_feedback = save_feedback_to_db(db, user_name, image)
        db.close()

        return {"message": "Feedback saved successfully"}
    except Exception as e:
        # Handle exceptions as needed
        raise HTTPException(status_code=500, detail=str(e))