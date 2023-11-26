from fastapi import FastAPI, UploadFile
from pathlib import Path
import SQLite.server.model as model
import logging
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('test_model')

app = FastAPI()

@app.post('/upload')
def upload_and_retrain(db_file: UploadFile):
    try:
        # Save the uploaded SQLite database file to a temporary location
        temp_db_path = Path("temp_db.db")
        try:
            temp_db_path.touch()
            logger.info("Temp file created successfully")
        except Exception as e:
            logger.error(f"Error creating temp file: {e}")

        with db_file.file as source_file, temp_db_path.open("wb") as temp_db:
            try:
                temp_db.write(source_file.read())
            except Exception as e:
                logger.error(f"Error copying file: {e}")

        # Assuming you have a method in your model module to retrain the model based on a database
        retrained_model = model.retrain(temp_db_path)

        # Convert retrained_model to a JSON-serializable format
        #retrained_model_serializable = jsonable_encoder(retrained_model)

        # Clean up temporary files
        temp_db_path.unlink()

        #return JSONResponse(content={"message": "Data uploaded and model retrained successfully", "retrained_model": retrained_model})
        return JSONResponse(content={"message": "Data uploaded and model retrained successfully"})
    except Exception as e:
        return {"error": str(e)}

#Then use the retrained model to get the evaluation metrics and GET them