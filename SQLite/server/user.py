from fastapi import APIRouter, HTTPException, UploadFile
from pydantic import BaseModel
import model

__all__ = ['get_router']

router = APIRouter(
    prefix='/user'
)

predictions_list = []

class PredictionData(BaseModel):
    image: bytes

class PredictionResponse(BaseModel):
    score: float

# user should be able to upload an image which should initate a predicition 
@router.post('/predict', response_model=PredictionResponse)
def predict(data: PredictionData):
    if not data.filename.endswith('.jpg'):
        raise HTTPException(status_code=400, detail="Supported file types: ['{}']".format('jpg'))
    
    image = data.image

    prediction_result = model.predict(image)

    predictions_list.append({'score': prediction_result})

    return {'score': prediction_result}

# user should be able to view the history of all previously predicitions
@router.get('/predictions', response_model=PredictionResponse)
def get_predictions():
    return predictions_list

def get_router():
    return router