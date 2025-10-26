#!/usr/bin/env python3
"""Check model performance metrics and fail if below thresholds."""
import json
import sys

def main():
    with open('reports/metrics.json', 'r') as f:
        metrics = json.load(f)
    
    rmse = metrics['rmse']
    r2 = metrics['r2']
    
    print(f'Model Performance - RMSE: {rmse:.4f}, R²: {r2:.4f}')
    
    # Fail if performance drops significantly
    if r2 <= 0.8:
        print(f'❌ R² too low: {r2:.4f} (threshold: > 0.8)', file=sys.stderr)
        sys.exit(1)
    
    if rmse >= 0.25:
        print(f'❌ RMSE too high: {rmse:.4f} (threshold: < 0.25)', file=sys.stderr)
        sys.exit(1)
    
    print('✅ Model performance meets requirements')

if __name__ == '__main__':
    main()
