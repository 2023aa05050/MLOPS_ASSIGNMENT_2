import pandas as pd
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load Data
train_data = pd.read_csv("data/train_features.csv")
test_data = pd.read_csv("data/test_features.csv")

# Extract features and target
X_train, y_train = train_data.iloc[:, :-1], train_data.iloc[:, -1]
X_test, y_test = test_data.iloc[:, :-1], test_data.iloc[:, -1]

# Ensure the target variable is categorical
y_train = y_train.astype(int)
y_test = y_test.astype(int)

# Split training data for validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Initialize TPOT AutoML
tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2, random_state=42)

# Fit the model
tpot.fit(X_train, y_train)

# Evaluate Model
y_pred = tpot.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)

# Save results
with open("reports/tpot_results.txt", "w") as f:
    f.write(f"Best Model Pipeline: {tpot.fitted_pipeline_}\n")
    f.write(f"Validation Accuracy: {accuracy:.4f}\n")

# Save best model
joblib.dump(tpot.fitted_pipeline_, "models/best_tpot_model.pkl")

print("Model selection completed using TPOT. Results saved.")
