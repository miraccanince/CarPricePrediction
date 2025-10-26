# src/data.py
import pandas as pd

def load_data(filepath):
    return pd.read_csv(filepath)

def preprocess_data(df):
    df = df.dropna() 
    return df