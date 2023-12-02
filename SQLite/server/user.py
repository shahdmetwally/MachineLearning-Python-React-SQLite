from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import SQLite.server.model as model
from fastapi.responses import JSONResponse
from pathlib import Path
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
origins = ["http://localhost:3000", " http://localhost:3000/"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictions_list = []

class PredictionData(BaseModel):
    image: UploadFile

# user should be able to upload an image which should initiate a prediction 
@app.post('/predict')
def predict(image: UploadFile = File(...)):
    try:
       # Ensure that the uploaded file is an image
        if not image.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            raise HTTPException(status_code=400, detail="Invalid file format. Supported formats: ['.png', '.jpg', '.jpeg']")

        # Save the uploaded image to a temporary file
        temp_image_path = Path(f"temp_{image.filename}")
        with open(temp_image_path, "wb") as temp_file:
            temp_file.write(image.file.read())
            #temp_file.close()
        prediction_result = model.predict(temp_image_path)
        #prediction_result_serializable = jsonable_encoder(prediction_result)
        # Removing the temporary file
        os.remove(temp_image_path)

        # Assuming predictions_list is a global list or stored elsewhere
        predictions_list.append({'score': prediction_result})

        return JSONResponse(content={"message": "Prediction successful", "score": prediction_result})
    except Exception as e:
        # Return other exception details in the response
        return JSONResponse(content={"error": str(e)})

# user should be able to view the history of all previously predicitions
@app.get('/predictions')
def get_predictions():
    return predictions_list