from fastapi import FastAPI
import mlflow
import mlflow.sklearn
from pydantic import BaseModel
import pandas as pd
import logging
import sqlite3
import os

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# # Configure logging
# os.makedirs('logs', exist_ok=True)
# logging.basicConfig(filename='logs/app.log', level=logging.INFO)

# # SQLite setup
# conn = sqlite3.connect('logs/predictions.db')
# conn.execute('CREATE TABLE IF NOT EXISTS predictions (input TEXT, output FLOAT)')
# conn.close()

app = FastAPI()

# Load the latest version of the registered model
try:
    model = mlflow.sklearn.load_model("models:/HousingModel/latest")  # Use 'latest' tag
    logging.info("Loaded latest version of HousingModel")
except Exception as e:
    logging.error(f"Failed to load model: {str(e)}")
    raise

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
def predict(input: HousingInput):
    try:
        data = pd.DataFrame([input.dict()])
        prediction = model.predict(data)[0]
        # Log prediction
        logging.info(f"Input: {input.dict()}, Prediction: {prediction}")
        conn = sqlite3.connect('logs/predictions.db')
        conn.execute('INSERT INTO predictions (input, output) VALUES (?, ?)', 
                    (str(input.dict()), float(prediction)))
        conn.commit()
        conn.close()
        return {"prediction": prediction}
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return {"error": str(e)}

@app.get("/metrics")
def metrics():
    try:
        conn = sqlite3.connect('logs/predictions.db')
        count = conn.execute('SELECT COUNT(*) FROM predictions').fetchone()[0]
        conn.close()
        return {"total_predictions": count}
    except Exception as e:
        logging.error(f"Metrics error: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)