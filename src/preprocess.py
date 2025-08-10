import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# Fetch dataset
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['target'] = housing.target

# Basic preprocessing: Scale features
scaler = StandardScaler()
df[housing.feature_names] = scaler.fit_transform(df[housing.feature_names])

# Split into train/test
train, test = train_test_split(df, test_size=0.2, random_state=42)

# Save to data folder
os.makedirs('data', exist_ok=True)
train.to_csv('data/train.csv', index=False)
test.to_csv('data/test.csv', index=False)
print("Data preprocessed and saved.")