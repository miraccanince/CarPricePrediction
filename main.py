# main.py
import os
import json
import joblib
from src.data import load_data, preprocess_data
from src.features import select_features, encode_features, transform_target, scale_features
from src.train import train_model
from src.evaluate import evaluate_model, interpret_model

def main():
    # Paths
    raw_data_path = 'data/raw/CarPrice_Assignment.csv'
    processed_data_path = 'data/processed/processed_data.csv'
    feature_engineered_path = 'data/processed/feature_engineered.csv'
    model_path = 'models/linear_regression_model.pkl'
    scaler_path = 'models/scaler.pkl'
    metrics_path = 'reports/metrics.json'

    # Ensure directories exist
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('reports', exist_ok=True)

    # Step 1: Load and preprocess data
    print("Loading and preprocessing data...")
    df = load_data(raw_data_path)
    df = preprocess_data(df)
    df.to_csv(processed_data_path, index=False)

    # Step 2: Feature engineering
    print("Performing feature engineering...")
    df = select_features(df)
    df = encode_features(df)
    df = transform_target(df)
    df, scaler = scale_features(df)
    df.to_csv(feature_engineered_path, index=False)
    joblib.dump(scaler, scaler_path)

    # Step 3: Prepare for training
    X = df.drop('price', axis=1)
    y = df['price']

    # Step 4: Train model
    print("Training model...")
    model, X_train, X_test, y_train, y_test = train_model(X, y)
    joblib.dump(model, model_path)

    # Step 5: Evaluate model
    print("Evaluating model...")
    rmse, r2, y_pred = evaluate_model(model, X_test, y_test)

    # Step 6: Interpret model
    feature_names = X.columns.tolist()
    interpretation_path = 'reports/model_interpretation.txt'
    interpret_model(model, feature_names, save_path=interpretation_path)

    # Save metrics
    metrics = {'rmse': rmse, 'r2': r2}
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"Model trained and evaluated. RMSE: {rmse:.4f}, R2: {r2:.4f}")

if __name__ == "__main__":
    main()