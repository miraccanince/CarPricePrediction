# tests/test_train.py
import pandas as pd
from sklearn.linear_model import LinearRegression
from src.data import load_data
from src.features import select_features, encode_features, transform_target, scale_features
from src.train import train_model

def test_train_model():
    df = load_data('data/raw/CarPrice_Assignment.csv')
    df = select_features(df)
    df = encode_features(df)
    df = transform_target(df)
    df, _ = scale_features(df)
    X = df.drop('price', axis=1)
    y = df['price']
    model, X_train, X_test, y_train, y_test = train_model(X, y)
    assert isinstance(model, LinearRegression)
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0