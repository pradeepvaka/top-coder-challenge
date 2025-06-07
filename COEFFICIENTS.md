# Generating Polynomial Regression Coefficients

This repository includes a simple baseline model for the reimbursement system. The coefficients used in `run.sh` were obtained by training a polynomial regression model on the provided `public_cases.json` data. The training code lives in `compute_coefficients.py`.

## Steps

1. Install the required Python dependencies. The script only relies on `scikit-learn` and `numpy`:
   ```bash
   pip install scikit-learn numpy
   ```
2. Run the training script:
   ```bash
   python3 compute_coefficients.py
   ```
3. The script prints the intercept and coefficient list. These numbers were copied directly into `run.sh`.

The polynomial features include all terms up to degree three (including interaction terms). The resulting model achieves an average error around $99 when evaluated with `./eval.sh`.
