from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import SQLite.server.model as model
from fastapi.responses import JSONResponse
from pathlib import Path
import os
import logging

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('user')

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
        #temp_image_path = Path(f"temp_{image.filename}")
        #try:
            #temp_image_path.touch()
            #logger.info("Temp file created successfully")
        #except Exception as e:
            #logger.error(f"Error creating temp file: {e}")
        #with open(temp_image_path, "wb") as temp_file:
            #temp_file.write(image.file.read())
        #image_data = BytesIO(image.file.read())
        prediction_result = model.predict(image.file)
        #prediction_result_serializable = jsonable_encoder(prediction_result)
        # Removing the temporary file
        #os.remove(temp_image_path)

        # Assuming predictions_list is a global list or stored elsewhere
        predictions_list.append({'score': prediction_result})

        return JSONResponse(content={"message": "Prediction successful", "score": prediction_result})
    except HTTPException as e:
        # Return HTTPException details in the response
        return JSONResponse(content={"error": e.detail}, status_code=e.status_code)
    except Exception as e:
        # Return other exception details in the response
        return JSONResponse(content={"error": str(e)})

# user should be able to view the history of all previously predicitions
@app.get('/predictions')
def get_predictions():
    return predictions_list