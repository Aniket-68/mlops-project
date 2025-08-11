import subprocess
import os
from dvc.api import DVCFileSystem

def check_data_changes():
    # Initialize DVC filesystem
    fs = DVCFileSystem(".")
    # Check if data files have changed
    status = subprocess.run(["dvc", "status", "data/train.csv", "data/test.csv"], capture_output=True, text=True)
    return "changed" in status.stdout

def retrain_if_changed():
    if check_data_changes():
        print("Data changed, re-running training...")
        subprocess.run(["python", "src/train.py"], check=True)
        print("Training completed.")
    else:
        print("No data changes detected.")

if __name__ == "__main__":
    retrain_if_changed()