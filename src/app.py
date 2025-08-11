from fastapi import FastAPI, HTTPException
import mlflow
import mlflow.sklearn
from pydantic import BaseModel, Field, field_validator
import pandas as pd
import logging
import sqlite3
import os
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

# ---------------------------
# Configuration
# ---------------------------
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
try:
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/latest")
    logging.info(f"Successfully loaded latest version of '{MODEL_NAME}'")
except Exception as e:
    logging.error(f"Failed to load model '{MODEL_NAME}': {str(e)}")
    raise RuntimeError(f"Model loading failed: {str(e)}")

# ---------------------------
# Prometheus Metrics
# ---------------------------
PREDICTIONS_COUNTER = Counter("total_predictions", "Total number of predictions made")

# ---------------------------
# FastAPI App
# ---------------------------
app = FastAPI(title="Housing Price Prediction API")

class HousingInput(BaseModel):
    MedInc: float = Field(..., ge=0.0, le=15.0, description="Median income in tens of thousands")
    HouseAge: float = Field(..., ge=1.0, le=52.0, description="Median house age in years")
    AveRooms: float = Field(..., ge=1.0, le=140.0, description="Average number of rooms per dwelling")
    AveBedrms: float = Field(..., ge=0.5, le=35.0, description="Average number of bedrooms per dwelling")
    Population: float = Field(..., ge=3.0, le=35000.0, description="Block group population")
    AveOccup: float = Field(..., ge=0.5, le=500.0, description="Average number of household members")
    Latitude: float = Field(..., ge=32.5, le=42.0, description="Latitude in degrees")
    Longitude: float = Field(..., ge=-124.5, le=-114.0, description="Longitude in degrees")

    @field_validator("AveBedrms")
    @classmethod
    def validate_bedrooms_vs_rooms(cls, v, values):
        if "AveRooms" in values and v > values["AveRooms"]:
            raise ValueError("AveBedrms cannot exceed AveRooms")
        return v

    @field_validator("Latitude", "Longitude")
    @classmethod
    def validate_location(cls, v, info):
        if info.field_name == "Latitude" and not (32.5 <= v <= 42.0):
            raise ValueError("Latitude must be within California's range (32.5 to 42.0)")
        if info.field_name == "Longitude" and not (-124.5 <= v <= -114.0):
            raise ValueError("Longitude must be within California's range (-124.5 to -114.0)")
        return v

@app.post("/predict")
def predict(input_data: HousingInput):
    """Predict housing prices from input features."""
    try:
        df = pd.DataFrame([input_data.dict()])
        prediction = float(model.predict(df)[0])

        # Increment Prometheus counter
        PREDICTIONS_COUNTER.inc()

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
def metrics():
    """Prometheus metrics endpoint."""
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
