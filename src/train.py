import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
from mlflow.tracking import MlflowClient

# MLflow tracking DB (Postgres on Neon)
mlflow.set_tracking_uri(
    "postgresql://neondb_owner:npg_KIfYBkW6e3Ey@ep-frosty-sunset-a1rsfxmx-pooler.ap-southeast-1.aws.neon.tech/neondb?sslmode=require"
)

MODEL_NAME = "HousingModel"
client = MlflowClient()

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
        return mse, run.info.run_id

# Train two models
lr_mse, lr_run_id = train_model(LinearRegression(), "LinearRegression")
dt_mse, dt_run_id = train_model(DecisionTreeRegressor(max_depth=5), "DecisionTree")

# Pick best
if lr_mse < dt_mse:
    best_run_id = lr_run_id
else:
    best_run_id = dt_run_id

# Register best model
result = mlflow.register_model(f"runs:/{best_run_id}/model", MODEL_NAME)

# Promote to Production (always overwrites)
client.transition_model_version_stage(
    name=MODEL_NAME,
    version=result.version,
    stage="Production",
    archive_existing_versions=True
)

print(f"✅ Model {MODEL_NAME} version {result.version} promoted to Production")
