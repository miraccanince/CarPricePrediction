# src/features.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def select_features(df):
    selected = ['symboling', 'curbweight', 'horsepower', 'compressionratio', 'citympg', 'fueltype', 'carbody', 'drivewheel', 'price']
    return df[selected].copy()

def encode_features(df):
    df = pd.get_dummies(df, drop_first=True)
    return df

def transform_target(df):
    df['price'] = np.log1p(df['price'])
    return df

def scale_features(df):
    scaler = StandardScaler()
    num_cols = ['symboling', 'curbweight', 'horsepower', 'compressionratio', 'citympg']
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df, scaler