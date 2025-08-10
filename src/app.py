from fastapi import FastAPI
import mlflow.sklearn
from pydantic import BaseModel  # For input validation (bonus)
import pandas as pd

app = FastAPI()


mlflow.set_tracking_uri("http://localhost:5000")

# Load registered model
model = mlflow.sklearn.load_model("models:/HousingModel/1")  # Version 1

class HousingInput(BaseModel):  # Bonus: Validation
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
    data = pd.DataFrame([input.dict()])
    prediction = model.predict(data)[0]
    return {"prediction": prediction}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)