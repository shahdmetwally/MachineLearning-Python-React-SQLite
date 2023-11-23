from fastapi import APIRouter, HTTPException, UploadFile
from pydantic import BaseModel
from PIL import Image
import model

__all__ = [
    'get_router'
]

router = APIRouter(
    prefix='/user'
)

class PredictionData(BaseModel):
    image: UploadFile

class PredictionResponse(BaseModel):
    score: float

def save_uploaded_file(file: UploadFile):
    with open('/Users/shahd.metwally/monorepo/Model/Arturo_Gatti_0002.jpg', 'wb') as f:
        f.write(file.file.read())

@router.post('/upload')
def upload(file: UploadFile):
    if not file.filename.endswith('.jpg'):
        raise HTTPException(status_code=400, detail="Supported file types: ['{}']".format('jpg'))
    
    save_uploaded_file(file)
    return {'filename': file.filename}

@router.post('/predict', response_model=PredictionResponse)
def predict(data: PredictionData):
    save_uploaded_file(data.image)

    # Use the predict function from your model
    prediction_result = model.predict('/Users/shahd.metwally/monorepo/Model/Arturo_Gatti_0002.jpg')

    # Return the prediction result
    return {'score': prediction_result}


def get_router():
    return router