import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency

# Load Previous & New Data (Ensure files exist)
previous_data = pd.read_csv("data/train_features.csv")
new_data = pd.read_csv("data/test_features.csv")

# Drop columns with NaN values in either dataset
previous_data.dropna(axis=1, inplace=True)
new_data.dropna(axis=1, inplace=True)

# Ensure both datasets have the same columns
common_columns = previous_data.columns.intersection(new_data.columns)
previous_data = previous_data[common_columns]
new_data = new_data[common_columns]

# Remove constant (zero variance) features
constant_features = previous_data.columns[previous_data.nunique() == 1]
previous_data.drop(columns=constant_features, inplace=True)
new_data.drop(columns=constant_features, inplace=True)

# Function to detect drift using Kolmogorov-Smirnov (KS) test
def detect_drift(previous, new):
    drift_results = {}
    for col in previous.columns:
        if previous[col].dtype in [np.float64, np.int64]:  # Numeric features
            stat, p_value = ks_2samp(previous[col], new[col])
            drift_results[col] = p_value
        else:  # Categorical features
            contingency_table = pd.crosstab(previous[col], new[col])
            _, p_value, _, _ = chi2_contingency(contingency_table)
            drift_results[col] = p_value

    return drift_results

# Run drift detection
drift_p_values = detect_drift(previous_data, new_data)

# Log drift results
drift_log = "reports/drift_detection1.log"
with open(drift_log, "w") as f:
    for feature, p_value in drift_p_values.items():
        f.write(f"{feature}: {'Drift detected' if p_value < 0.05 else 'No drift'} (p={p_value:.4f})\n")

print("Drift detection completed. Results saved to reports/drift_detection.log")
