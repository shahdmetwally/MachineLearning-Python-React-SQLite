from fastapi import APIRouter, UploadFile
from pathlib import Path
import shutil
import SQLite.server.model as model

__all__ = ['get_router']

router = APIRouter(
    prefix='/admin'
)

@router.post('/upload')
async def upload_and_retrain(db_file: UploadFile):
    try:
        # Save the uploaded SQLite database file to a temporary location
        temp_db_path = Path("temp_db.db")
        with temp_db_path.open("wb") as temp_db:
            shutil.copyfileobj(db_file.file, temp_db)

        # Assuming you have a method in your model module to retrain the model based on a database
        retrained_model = model.retrain(temp_db_path)

        # Clean up temporary files
        temp_db_path.unlink()

        return {"message": "Data uploaded and model retrained successfully", "retrained_model": retrained_model}
    except Exception as e:
        return {"error": str(e)}


def get_router():
    return router