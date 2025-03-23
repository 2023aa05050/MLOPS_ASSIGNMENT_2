import mlflow
import mlflow.sklearn
import xgboost as xgb
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.pipeline import Pipeline  # Import Pipeline

# Load Data
train_data = pd.read_csv("data/train_features.csv")
X, y = train_data.iloc[:, :-1], train_data.iloc[:, -1]

# Convert `y` to integer labels
y = y.astype(int)  # Ensure `y` is an integer

# Split data into training & validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=35)

# Load best TPOT pipeline
tpot_pipeline = joblib.load("models/best_tpot_model.pkl")  

# Extract the actual classifier from the TPOT pipeline
if isinstance(tpot_pipeline, Pipeline):
    best_model = tpot_pipeline.steps[-1][1]  # Get the final model from pipeline
    best_params = best_model.get_params()  # Extract hyperparameters
else:
    raise ValueError("Loaded TPOT model is not a valid scikit-learn pipeline.")

# Start MLflow tracking
mlflow.set_experiment("MLOps_Model_Tracking")

with mlflow.start_run():
    # Train model with extracted parameters
    model = xgb.XGBClassifier(**best_params, eval_metric="logloss", random_state=35)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_val)

    # Compute Metrics
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, average="macro")
    recall = recall_score(y_val, y_pred, average="macro")

    # Log Metrics
    mlflow.log_params(best_params)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)

    # Log Model
    mlflow.sklearn.log_model(model, "xgboost_model")

    # Save Model
    joblib.dump(model, "models/final_model_m4.pkl")

    print(f"Model trained and logged in MLflow. Accuracy: {accuracy:.4f}")
