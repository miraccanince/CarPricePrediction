# tests/test_features.py
import pandas as pd
import numpy as np
from src.data import load_data
from src.features import select_features, encode_features, transform_target, scale_features

def test_select_features():
    df = load_data('data/raw/CarPrice_Assignment.csv')
    selected = select_features(df)
    expected_cols = ['symboling', 'curbweight', 'horsepower', 'compressionratio', 'citympg', 'fueltype', 'carbody', 'drivewheel', 'price']
    assert list(selected.columns) == expected_cols

def test_encode_features():
    df = load_data('data/raw/CarPrice_Assignment.csv')
    df = select_features(df)
    encoded = encode_features(df)
    # Should have more columns due to dummies
    assert encoded.shape[1] > df.shape[1]

def test_transform_target():
    original_df = load_data('data/raw/CarPrice_Assignment.csv')
    df = select_features(original_df)
    transformed = transform_target(df)
    # Price should be log transformed
    assert (transformed['price'] == np.log1p(original_df['price'])).all()

def test_scale_features():
    df = load_data('data/raw/CarPrice_Assignment.csv')
    df = select_features(df)
    df = encode_features(df)
    scaled, scaler = scale_features(df)
    num_cols = ['symboling', 'curbweight', 'horsepower', 'compressionratio', 'citympg']
    # Check if scaled
    assert scaled[num_cols].std().all() > 0  # Not all zero