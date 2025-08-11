# import mlflow
# import mlflow.sklearn
# from sklearn.linear_model import LinearRegression
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.metrics import mean_squared_error
# from sklearn.model_selection import train_test_split
# import pandas as pd
# from mlflow.tracking import MlflowClient
# import os
# # MLflow tracking DB (Postgres on Neon)
# # mlflow.set_tracking_uri(
# #     "postgresql://neondb_owner:npg_KIfYBkW6e3Ey@ep-frosty-sunset-a1rsfxmx-pooler.ap-southeast-1.aws.neon.tech/neondb?sslmode=require"
# # )
# # Set MLflow tracking URI (default to localhost for local runs)
# # mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
# # mlflow.set_tracking_uri(mlflow_tracking_uri)
# mlflow.set_tracking_uri("http://localhost:5000")


# MODEL_NAME = "HousingModel"
# client = MlflowClient()

# # Load data
# train = pd.read_csv('data/train.csv')
# X = train.drop('target', axis=1)
# y = train['target']
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# def train_model(model, model_name):
#     with mlflow.start_run(run_name=model_name) as run:
#         model.fit(X_train, y_train)
#         preds = model.predict(X_val)
#         mse = mean_squared_error(y_val, preds)
#         mlflow.log_param("model_type", model_name)
#         mlflow.log_metric("mse", mse)
#         mlflow.sklearn.log_model(model, "model")
#         print(f"{model_name} MSE: {mse}")
#         return mse, run.info.run_id

# # Train two models
# lr_mse, lr_run_id = train_model(LinearRegression(), "LinearRegression")
# dt_mse, dt_run_id = train_model(DecisionTreeRegressor(max_depth=5), "DecisionTree")

# # Pick best
# if lr_mse < dt_mse:
#     best_run_id = lr_run_id
#     best_model = LinearRegression()
#     best_model.fit(X_train, y_train)

# else:
#     best_run_id = dt_run_id
#     best_model = DecisionTreeRegressor(max_depth=5)
#     best_model.fit(X_train, y_train)


# # Register best model
# with mlflow.start_run(run_name="BestModel") as best_run:
#     mlflow.sklearn.log_model(best_model, "best_model")

# result=mlflow.register_model(f"runs:/{best_run_id}/model", MODEL_NAME)

# # Promote to Production (always overwrites)
# # client.transition_model_version_stage(
# #     name=MODEL_NAME,
# #     version=result.version,
# #     stage="Production",
# #     archive_existing_versions=True
# # )

# print(f"Model {MODEL_NAME} version {result.version} promoted to Production")



import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
from mlflow.tracking import MlflowClient
import os
import logging

# ---------------------------
# Configuration
# ---------------------------
MODEL_NAME = "HousingModel"
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Logging setup
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "train.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# MLflow tracking
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
client = MlflowClient()

# ---------------------------
# Load data
# ---------------------------
try:
    train = pd.read_csv('data/train.csv')
    logging.info("Successfully loaded training data from 'data/train.csv'")
except Exception as e:
    logging.error(f"Failed to load training data: {str(e)}")
    raise

X = train.drop('target', axis=1)
y = train['target']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------
# Training function
# ---------------------------
def train_model(model, model_name):
    logging.info(f"Starting training for model: {model_name}")
    with mlflow.start_run(run_name=model_name) as run:
        try:
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            mse = mean_squared_error(y_val, preds)

            mlflow.log_param("model_type", model_name)
            mlflow.log_metric("mse", mse)
            mlflow.sklearn.log_model(model, "model")

            logging.info(f"{model_name} training completed. MSE={mse:.4f}")
            return mse, run.info.run_id
        except Exception as e:
            logging.error(f"Error during training of {model_name}: {str(e)}")
            raise

# ---------------------------
# Train models
# ---------------------------
lr_mse, lr_run_id = train_model(LinearRegression(), "LinearRegression")
dt_mse, dt_run_id = train_model(DecisionTreeRegressor(max_depth=5), "DecisionTree")

# ---------------------------
# Pick best
# ---------------------------
if lr_mse < dt_mse:
    best_run_id = lr_run_id
    best_model = LinearRegression()
    best_model.fit(X_train, y_train)
    logging.info("Selected LinearRegression as the best model.")
else:
    best_run_id = dt_run_id
    best_model = DecisionTreeRegressor(max_depth=5)
    best_model.fit(X_train, y_train)
    logging.info("Selected DecisionTree as the best model.")

# ---------------------------
# Register best model
# ---------------------------
with mlflow.start_run(run_name="BestModel") as best_run:
    mlflow.sklearn.log_model(best_model, "best_model")

try:
    result = mlflow.register_model(f"runs:/{best_run_id}/model", MODEL_NAME)
    logging.info(f"Model {MODEL_NAME} version {result.version} registered successfully.")
except Exception as e:
    logging.error(f"Failed to register model {MODEL_NAME}: {str(e)}")
    raise

# Promote to Production (optional)
# client.transition_model_version_stage(
#     name=MODEL_NAME,
#     version=result.version,
#     stage="Production",
#     archive_existing_versions=True
# )

logging.info(f"Training pipeline completed. Model {MODEL_NAME} version {result.version} is ready.")