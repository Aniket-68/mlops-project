import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
import os

mlflow.set_tracking_uri("http://localhost:5000")

# Load data
train = pd.read_csv('data/train.csv')
X = train.drop('target', axis=1)
y = train['target']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(model, model_name):
    with mlflow.start_run(run_name=model_name) as run:
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        mse = mean_squared_error(y_val, preds)
        mlflow.log_param("model_type", model_name)
        mlflow.log_metric("mse", mse)
        mlflow.sklearn.log_model(model, "model")
        print(f"{model_name} MSE: {mse}")
        return mse, run.info.run_id  # return run_id


# Train two models
lr_mse, lr_run_id = train_model(LinearRegression(), "LinearRegression")
dt_mse, dt_run_id = train_model(DecisionTreeRegressor(max_depth=5), "DecisionTree")


# Pick best model
# Select best (lowest MSE) and register
if lr_mse < dt_mse:
    best_run_id = lr_run_id
    best_model = LinearRegression()
    best_model.fit(X_train, y_train)
else:
    best_run_id = dt_run_id
    best_model = DecisionTreeRegressor(max_depth=5)
    best_model.fit(X_train, y_train)

# Register best model
with mlflow.start_run(run_name="BestModel") as best_run:
    mlflow.sklearn.log_model(best_model, "best_model")
    mlflow.register_model(f"runs:/{best_run_id}/model", "HousingModel")
