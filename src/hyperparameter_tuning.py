import optuna
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load Data
train_data = pd.read_csv("data/train_features.csv")
X, y = train_data.iloc[:, :-1], train_data.iloc[:, -1]

# Ensure target is categorical
y = y.astype(int)

# Split data into train-validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the optimization objective
def objective(trial):
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
    }
    
    # Train XGBoost model
    model = xgb.XGBClassifier(**params, eval_metric="logloss", random_state=42)
    model.fit(X_train, y_train)
    
    # Validate model
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    
    return accuracy  # Optuna maximizes this metric

# Run Optuna optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

# Best model parameters
best_params = study.best_params
print("Best Hyperparameters:", best_params)

# Train final model with best parameters
final_model = xgb.XGBClassifier(**best_params, eval_metric="logloss", random_state=42)
final_model.fit(X_train, y_train)

# Save final tuned model
joblib.dump(final_model, "models/final_model.pkl")

# Save tuning results
with open("reports/hyperparameter_tuning.log", "w") as f:
    f.write(f"Best Hyperparameters: {best_params}\n")
    f.write(f"Best Accuracy: {study.best_value:.4f}\n")

print("Hyperparameter tuning completed. Best model saved.")
