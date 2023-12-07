from fastapi import FastAPI, UploadFile
from pathlib import Path
import Model.server.model as model
import os
from fastapi.responses import JSONResponse
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

#Admin should upload a new batch of data and retrain the model with it
#Then use the retrained model to get the evaluation metrics and GET them
@app.post('/retrain')
def upload_and_retrain(db_file: UploadFile):
    try:
        # Save the uploaded SQLite database file to a temporary location
        temp_db_path = Path("temp_db.db")

        with db_file.file as source_file, temp_db_path.open("wb") as temp_db:
            temp_db.write(source_file.read())

        # Assuming you have a method in your model module to retrain the model based on a database
        accuracy, precision, recall, f1 = model.retrain(temp_db_path)

        # Clean up temporary files
        os.remove(temp_db_path)

        return JSONResponse(content={"message": "Data uploaded and model retrained successfully", "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1})
    except Exception as e:
        return {"error": str(e)}


@app.get('/models')
def get_all_models():
    try:
        model_versions = model.get_all_models()
        active_model_path = model.get_latest_model_version()
        active_model_path = Path(active_model_path)
        active_model= active_model_path.stem

        return JSONResponse(content={"message": "Models were obtained successfully", "models": model_versions, "active_model": active_model})
    except Exception as e:
        return {"error": str(e)}