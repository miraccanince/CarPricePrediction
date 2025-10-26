# src/evaluate.py
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    return rmse, r2, y_pred

def interpret_model(model, feature_names, save_path=None):
    """
    Interpret the linear regression model by showing feature coefficients.
    Since features are scaled and target is log-transformed, coefficients show
    the impact on log(price) for a one-standard-deviation change in the feature.
    """
    coefficients = model.coef_
    intercept = model.intercept_

    interpretation = "\n=== Model Interpretation ===\n"
    interpretation += f"Intercept: {intercept:.4f}\n"
    interpretation += "\nFeature Coefficients (impact on log(price)):\n"
    for name, coef in zip(feature_names, coefficients):
        interpretation += f"{name}: {coef:.4f}\n"

    # Most important features
    coef_df = pd.DataFrame({'feature': feature_names, 'coefficient': coefficients})
    coef_df['abs_coef'] = coef_df['coefficient'].abs()
    top_features = coef_df.nlargest(5, 'abs_coef')

    interpretation += "\nTop 5 Most Important Features:\n"
    for _, row in top_features.iterrows():
        interpretation += f"{row['feature']}: {row['coefficient']:.4f}\n"

    print(interpretation)

    if save_path:
        with open(save_path, 'w') as f:
            f.write(interpretation)
        print(f"\nInterpretation saved to {save_path}")

    return coef_df