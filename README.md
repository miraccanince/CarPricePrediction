# Car Price Prediction Project

[![CI/CD Pipeline](https://github.com/miraccanince/machineLearning/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/miraccanince/machineLearning/actions/workflows/ci-cd.yml)

A comprehensive machine learning project for predicting car prices using linear regression. This project demonstrates a complete ML pipeline from data exploration to model deployment, built with modular, testable code.

## Features

- **Data Exploration**: Exploratory data analysis with visualizations (histograms, correlations, boxplots)
- **Feature Engineering**: Feature selection, categorical encoding, target transformation (log), and feature scaling
- **Model Training**: Linear regression with train/test split
- **Model Evaluation**: RMSE and R² metrics
- **Modular Code**: Clean separation of concerns with reusable functions
- **Automated Testing**: Unit tests for all core functions
- **Build Automation**: Makefile for common tasks

## Model Performance

- **RMSE**: 0.2030
- **R² Score**: 0.8482 (84.8% variance explained)

## Model Interpretation

The model provides interpretable coefficients showing how each feature impacts car prices:

- **Positive coefficients**: Features that increase price (e.g., higher curb weight, horsepower)
- **Negative coefficients**: Features that decrease price
- **Scaled features**: Coefficients represent impact per standard deviation change
- **Log-transformed target**: Effects are on the logarithm of price

See `reports/model_interpretation.txt` for detailed feature importance.

## Project Structure

```
CarPricePrediction/
├── data/
│   ├── raw/           # Raw dataset (CarPrice_Assignment.csv)
│   └── processed/     # Processed and feature-engineered data
├── notebooks/         # Jupyter notebooks for analysis
│   ├── 01_eda.ipynb              # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb  # Feature engineering
│   └── 03_modeling.ipynb         # Model training and evaluation
├── src/               # Core source code (modular functions)
│   ├── __init__.py
│   ├── data.py        # Data loading and preprocessing
│   ├── features.py    # Feature selection, encoding, transformation, scaling
│   ├── train.py       # Model training utilities
│   └── evaluate.py    # Model evaluation metrics
├── scripts/           # Executable scripts
│   ├── prepare_data.py    # Data preparation pipeline
│   ├── train_model.py     # Model training pipeline
│   └── evaluate_model.py  # Model evaluation pipeline
├── models/            # Saved models and scalers
├── reports/           # Evaluation reports and metrics
│   ├── metrics.json          # Model performance metrics
│   └── model_interpretation.txt  # Feature importance analysis
├── tests/             # Unit tests
├── main.py            # Main entry point for full pipeline
├── Makefile           # Build automation
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

## Installation

1. **Clone the repository** (if applicable) or navigate to the project directory

2. **Create virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   # Or use: make install
   ```

## Usage

### Quick Start (Full Pipeline)
Run the complete ML pipeline with a single command:
```bash
python main.py
```

### Step-by-Step Execution
Use the Makefile for individual steps:
```bash
make data       # Prepare and process data
make train      # Train the model
make evaluate   # Evaluate the model
```

Or run scripts directly:
```bash
python scripts/prepare_data.py
python scripts/train_model.py
python scripts/evaluate_model.py
```

### Interactive Analysis
Explore the data and model using Jupyter notebooks:
```bash
jupyter notebook notebooks/
```

### Testing
Run unit tests to validate the code:
```bash
python -m pytest tests/
# Or: make test
```

## Dependencies

Key packages (see `requirements.txt` for full list):
- pandas, numpy: Data manipulation
- scikit-learn: Machine learning
- matplotlib, seaborn: Visualization
- pytest: Testing
- joblib: Model serialization

## Data

The project uses the "Car Price Assignment" dataset with features like:
- Car specifications (engine, horsepower, curb weight)
- Categorical features (fuel type, car body, drive wheel)
- Target: Car price

## Methodology

1. **Data Loading & Preprocessing**: Load CSV, handle missing values
2. **Feature Selection**: Select relevant features for modeling
3. **Feature Engineering**:
   - One-hot encoding for categorical variables
   - Log transformation of target (price)
   - Standard scaling of numerical features
4. **Model Training**: Linear regression with 80/20 train/test split
5. **Evaluation**: RMSE and R² on test set

## CI/CD Pipeline

This project uses GitHub Actions for automated testing and validation:

- **Triggers**: Runs on every push and pull request to `main` branch
- **Environment**: Ubuntu with Python 3.9
- **Steps**:
  1. **Setup**: Install Python and dependencies
  2. **Testing**: Run unit tests with pytest
  3. **Pipeline**: Execute full ML pipeline (`main.py`)
  4. **Validation**: Check model performance meets thresholds
  5. **Artifacts**: Save model and reports for download

**Performance Guards**: Pipeline fails if R² < 0.8 or RMSE > 0.25, ensuring model quality.

**Learning CI/CD**: The workflow demonstrates:
- Automated testing on code changes
- ML pipeline validation
- Artifact storage for model deployment
- Quality gates for ML models

## License

This project is for educational purposes.