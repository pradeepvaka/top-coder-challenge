import json
from pathlib import Path
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np

DATA_FILE = Path('public_cases.json')

def load_data(path=DATA_FILE):
    with open(path) as f:
        data = json.load(f)
    X, y = [], []
    for case in data:
        inp = case['input']
        X.append([
            inp['trip_duration_days'],
            inp['miles_traveled'],
            inp['total_receipts_amount'],
        ])
        y.append(case['expected_output'])
    return np.array(X), np.array(y)

def train_model(X, y):
    poly = PolynomialFeatures(degree=3, include_bias=True)
    X_poly = poly.fit_transform(X)
    model = LinearRegression(fit_intercept=True)
    model.fit(X_poly, y)
    return model, poly

def main():
    X, y = load_data()
    model, poly = train_model(X, y)

    coef_list = model.coef_.tolist()
    intercept = model.intercept_

    print("Intercept:", intercept)
    print("Coefficients:")
    for name, coef in zip(poly.get_feature_names_out(['d','m','r']), coef_list):
        print(f"  {name}: {coef}")

    # also show arrays that can be copied into run.sh
    print("\nPython list format for run.sh:")
    print("coefs = [" + ",".join(map(str, coef_list)) + "]")
    print("intercept =", intercept)

if __name__ == '__main__':
    main()
