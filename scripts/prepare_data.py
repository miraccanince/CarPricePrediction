#!/usr/bin/env python3
# Script to prepare data for car price prediction

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data import load_data, preprocess_data

def main():
    # Load raw data
    df = load_data('data/raw/CarPrice_Assignment.csv')
    print(f"Loaded data: {df.shape}")

    # Preprocess
    df = preprocess_data(df)
    print(f"After preprocessing: {df.shape}")

    # Save processed data
    os.makedirs('data/processed', exist_ok=True)
    df.to_csv('data/processed/processed_data.csv', index=False)
    print("Processed data saved to data/processed/processed_data.csv")

if __name__ == "__main__":
    main()
