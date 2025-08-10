from fastapi import FastAPI, HTTPException
import mlflow
import mlflow.sklearn
from pydantic import BaseModel
import pandas as pd
import logging
import sqlite3
import os

# ---------------------------
# Configuration
# ---------------------------
# MLFLOW_TRACKING_URI = "http://localhost:5000"

# Set MLflow tracking URI (default to localhost for local runs)
mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(mlflow_tracking_uri)

MODEL_NAME = "HousingModel"
DB_PATH = "logs/predictions.db"
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Logging setup
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "app.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# SQLite setup
conn = sqlite3.connect(DB_PATH)
conn.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        input TEXT NOT NULL,
        output FLOAT NOT NULL
    )
""")
conn.close()

# ---------------------------
# MLflow Setup
# ---------------------------
# mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
try:
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/latest")
    logging.info(f"Successfully loaded latest version of '{MODEL_NAME}'")
except Exception as e:
    logging.error(f"Failed to load model '{MODEL_NAME}': {str(e)}")
    raise RuntimeError(f"Model loading failed: {str(e)}")

# ---------------------------
# FastAPI App
# ---------------------------
app = FastAPI(title="Housing Price Prediction API")

class HousingInput(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

@app.post("/predict")
def predict(input_data: HousingInput):
    """Predict housing prices from input features."""
    try:
        df = pd.DataFrame([input_data.dict()])
        prediction = float(model.predict(df)[0])

        # Log prediction to DB
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "INSERT INTO predictions (input, output) VALUES (?, ?)",
                (str(input_data.dict()), prediction)
            )
            conn.commit()

        logging.info(f"Prediction made: Input={input_data.dict()}, Output={prediction}")
        return {"prediction": prediction}

    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/metrics")
def get_metrics():
    """Retrieve total number of predictions made."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            count = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        return {"total_predictions": count}

    except Exception as e:
        logging.error(f"Metrics retrieval error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Metrics retrieval failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
