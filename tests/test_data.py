# tests/test_data.py
import pandas as pd
from src.data import load_data, preprocess_data

def test_load_data():
    df = load_data('data/raw/CarPrice_Assignment.csv')
    assert isinstance(df, pd.DataFrame)
    assert not df.empty

def test_preprocess_data():
    df = load_data('data/raw/CarPrice_Assignment.csv')
    processed = preprocess_data(df)
    assert processed.shape[0] <= df.shape[0]  # Should not have more rows after dropna
    assert processed.isnull().sum().sum() == 0  # No nulls