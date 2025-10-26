# tests/test_evaluate.py
import pandas as pd
from src.data import load_data
from src.features import select_features, encode_features, transform_target, scale_features
from src.train import train_model
from src.evaluate import evaluate_model

def test_evaluate_model():
    df = load_data('data/raw/CarPrice_Assignment.csv')
    df = select_features(df)
    df = encode_features(df)
    df = transform_target(df)
    df, _ = scale_features(df)
    X = df.drop('price', axis=1)
    y = df['price']
    model, _, X_test, _, y_test = train_model(X, y)
    rmse, r2, y_pred = evaluate_model(model, X_test, y_test)
    assert rmse > 0
    assert 0 <= r2 <= 1
    assert len(y_pred) == len(y_test)