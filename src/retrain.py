import subprocess
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/retrain.log"),
        logging.StreamHandler()
    ]
)

def check_data_changes():
    """Check if data files or their dependencies have changed."""
    try:
        # Run dvc status for the preprocess stage 
        status = subprocess.run(
            ["dvc", "status", "preprocess"],
            capture_output=True,
            text=True,
            check=True
        )
        logging.info(f"DVC status output:\n{status.stdout}")
        # Check for changes in the preprocess stage
        return "changed" in status.stdout.lower() or "modified" in status.stdout.lower()
    except subprocess.CalledProcessError as e:
        logging.error(f"DVC status check failed: {e.stderr}")
        raise

def retrain_if_changed():
    """Run training if data or dependencies have changed."""
    try:
        if check_data_changes():
            logging.info("Data or dependencies changed, re-running training...")
            subprocess.run(["python", "src/train.py"], check=True)
            logging.info("Training completed successfully.")
            subprocess.run(["dvc", "commit"], check=True)
            logging.info("Training outputs committed to DVC.")
        else:
            logging.info("No data or dependency changes detected.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Training or DVC commit failed: {e.stderr}")
        raise

if __name__ == "__main__":
    retrain_if_changed()