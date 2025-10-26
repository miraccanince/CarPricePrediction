#!/usr/bin/env python3
# Script to train the car price prediction model

#!/usr/bin/env python3
# Script to prepare data for car price prediction

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.features import select_features, encode_features, transform_target, scale_features
from src.data import load_data, preprocess_data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib


def main():
    # Load processed data
    df = load_data('data/processed/processed_data.csv')
    print(f"Loaded processed data: {df.shape}")

    # Feature selection
    df = select_features(df)
    print(f"After feature selection: {df.shape}")

    # Encode categorical features
    df = encode_features(df)
    print(f"After encoding features: {df.shape}")

    # Transform target variable
    df = transform_target(df)
    print(f"After transforming target: {df.shape}")

    # Scale features
    df, scaler = scale_features(df)
    print(f"After scaling features: {df.shape}")

    # Split data
    X = df.drop('price', axis=1)
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Model training completed.")

    # Save model and scaler
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/linear_regression_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    print("Model and scaler saved to models/ directory.")

if __name__ == "__main__":
    main()
