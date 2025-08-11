FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY logs/ ./logs/
COPY mlruns/ ./mlruns/  

# Set MLflow URI to access host's server
ENV MLFLOW_TRACKING_URI=http://host.docker.internal:5000

EXPOSE 8000

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]

CMD ["/bin/bash", "-c", "python retrain.py && uvicorn src.app:app --host 0.0.0.0 --port 8000"]