# import pandas as pd
# from sklearn.datasets import fetch_california_housing
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# import os

# # Fetch dataset
# housing = fetch_california_housing()
# df = pd.DataFrame(housing.data, columns=housing.feature_names)
# df['target'] = housing.target

# # Basic preprocessing: Scale features
# scaler = StandardScaler()
# df[housing.feature_names] = scaler.fit_transform(df[housing.feature_names])

# # Split into train/test
# train, test = train_test_split(df, test_size=0.2, random_state=42)

# # Save to data folder
# os.makedirs('data', exist_ok=True)
# train.to_csv('data/train.csv', index=False)
# test.to_csv('data/test.csv', index=False)
# print("Data preprocessed and saved.")

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/preprocess.log"),  # Log to file
        logging.StreamHandler()  # Log to stdout for Docker
    ]
)

def clean_data(df, feature_names):
    """Clean the dataset by handling missing values and outliers."""
    logging.info("Starting data cleaning...")

    # Check for missing values
    if df.isnull().any().any():
        logging.warning(f"Found {df.isnull().sum().sum()} missing values.")
        # Impute missing numerical values with median
        df[feature_names] = df[feature_names].fillna(df[feature_names].median())
        logging.info("Missing values imputed with median.")
    else:
        logging.info("No missing values found.")

    # Remove outliers using IQR method for numerical features
    for col in feature_names:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        initial_rows = df.shape[0]
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        logging.info(f"Removed {initial_rows - df.shape[0]} outliers from {col}.")
    
    # Check for negative or invalid values in features that should be non-negative
    for col in feature_names:
        if (df[col] < 0).any():
            logging.warning(f"Negative values found in {col}. Replacing with median.")
            df.loc[df[col] < 0, col] = df[col].median()
    
    # Ensure target is non-negative (housing prices)
    if (df['target'] < 0).any():
        logging.warning("Negative target values found. Replacing with median.")
        df.loc[df['target'] < 0, 'target'] = df['target'].median()

    logging.info("Data cleaning completed.")
    return df

def preprocess_data():
    """Preprocess and save the California Housing dataset."""
    try:
        logging.info("Fetching California Housing dataset...")
        # Fetch dataset
        housing = fetch_california_housing()
        df = pd.DataFrame(housing.data, columns=housing.feature_names)
        df['target'] = housing.target
        logging.info(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns.")

        # Clean the data
        df = clean_data(df, housing.feature_names)

        # Scale features
        logging.info("Scaling features...")
        scaler = StandardScaler()
        df[housing.feature_names] = scaler.fit_transform(df[housing.feature_names])
        logging.info("Features scaled successfully.")

        # Split into train/test
        logging.info("Splitting data into train and test sets...")
        train, test = train_test_split(df, test_size=0.2, random_state=42)
        logging.info(f"Train set: {train.shape[0]} rows, Test set: {test.shape[0]} rows.")

        # Save to data folder
        os.makedirs('data', exist_ok=True)
        train.to_csv('data/train.csv', index=False)
        test.to_csv('data/test.csv', index=False)
        logging.info("Data preprocessed and saved to data/train.csv and data/test.csv.")

    except Exception as e:
        logging.error(f"Preprocessing failedS: {str(e)}")
        raise

if __name__ == "__main__":
    preprocess_data()