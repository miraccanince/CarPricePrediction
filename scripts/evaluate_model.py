#!/usr/bin/env python3
# Script to evaluate the car price prediction model

import sys
import os
import json
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.evaluate import evaluate_model
from src.data import load_data
from src.features import select_features, encode_features, transform_target, scale_features
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    # Load processed data
    df = load_data('data/processed/processed_data.csv')
    df = select_features(df)
    df = encode_features(df)
    df = transform_target(df)
    df, scaler = scale_features(df)

    # Split data
    X = df.drop('price', axis=1)
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Load model
    model = joblib.load('models/linear_regression_model.pkl')

    # Evaluate
    rmse, r2, y_pred = evaluate_model(model, X_test, y_test)
    print(f"RMSE: {rmse:.4f}, R2: {r2:.4f}")

    # Save metrics
    os.makedirs('reports', exist_ok=True)
    with open('reports/metrics.json', 'w') as f:
        import json
        json.dump({'rmse': rmse, 'r2': r2}, f)
    print("Metrics saved to reports/metrics.json")

if __name__ == "__main__":
    main()
