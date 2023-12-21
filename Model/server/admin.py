from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
from pathlib import Path
import Model.server.model as model
import Model.server.model_registry as model_registry

app = FastAPI()

origins = ["http://localhost:3000", " http://localhost:3000/"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Admin should upload a new batch of data and retrain the model with it
# Then use the retrained model to get the evaluation metrics and GET them
@app.post("/retrain")
def upload_and_retrain(db_file: UploadFile):
    try:
        # Save the uploaded SQLite database file to a temporary location
        temp_db_path = Path("temp_db.db")

        with db_file.file as source_file, temp_db_path.open("wb") as temp_db:
            temp_db.write(source_file.read())

        (
            retrained_accuracy,
            retrained_precision,
            retrained_recall,
            retrained_f1,
            old_accuracy,
            old_precision,
            old_recall,
            old_f1,
        ) = model.retrain(temp_db_path)

        # Clean up temporary files
        os.remove(temp_db_path)

        return JSONResponse(
            content={
                "retrained_accuracy": retrained_accuracy,
                "retrained_precision": retrained_precision,
                "retrained_recall": retrained_recall,
                "retrained_f1": retrained_f1,
                "old_accuracy": old_accuracy,
                "old_precision": old_precision,
                "old_recall": old_recall,
                "old_f1": old_f1,
            }
        )
    except Exception as e:
        return {"error": str(e)}


@app.get("/models")
def get_all_models():
    try:
        model_versions = model_registry.get_all_models()
        active_model_path = model_registry.get_latest_model_version()
        active_model_path = Path(active_model_path)
        active_model = active_model_path.stem

        return JSONResponse(
            content={
                "message": "Models were obtained successfully",
                "models": model_versions,
                "active_model": active_model,
            }
        )
    except Exception as e:
        return {"error": str(e)}


@app.put("/model")
def set_active_model(version: str):
    try:
        model_registry.set_active_model(version)

        return JSONResponse(
            content={
                "message": f"Model {version} has been set as the active model successfully"
            }
        )
    except Exception as e:
        return {"error": str(e)}
