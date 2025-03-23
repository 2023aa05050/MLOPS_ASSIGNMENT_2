import shap
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from sklearn.preprocessing import MinMaxScaler

# Load data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Reshape images
sample_size = 3000  # Adjust as needed
x_train, y_train = x_train[:sample_size], y_train[:sample_size]
x_test, y_test = x_test[:sample_size], y_test[:sample_size]

x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

# Normalize pixel values
scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Train a simple XGBoost model
xgb_model = xgb.XGBClassifier(n_estimators=50, max_depth=3, eval_metric='mlogloss', use_label_encoder=False)
xgb_model.fit(x_train_scaled, y_train)

# Explainability with SHAP
explainer = shap.Explainer(xgb_model)
shap_values = explainer(x_train_scaled[:100])  # Take a subset for efficiency

# Convert SHAP values to NumPy
shap_values_array = shap_values.values

# Handle Multiclass Case: Take the SHAP values for one class (e.g., class 0)
if shap_values_array.ndim == 3:  # If shape is (num_samples, num_features, num_classes)
    shap_values_array = shap_values_array[:, :, 0]  # Pick one class

# Ensure feature matrix is a DataFrame
feature_df = pd.DataFrame(x_train_scaled[:100])

# SHAP Summary Plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, feature_df, show=False)
plt.savefig("reports/shap_summary_plot.png", bbox_inches="tight")
plt.close()

# Get most important feature index
mean_abs_shap = np.abs(shap_values_array).mean(axis=0)
top_feature = int(np.argmax(mean_abs_shap))

# SHAP Dependence Plot (Fixed)
plt.figure(figsize=(8, 5))
shap.dependence_plot(
    top_feature,
    shap_values_array,  # Use processed SHAP values
    feature_df,  # Ensure feature matrix matches SHAP values
    show=False
)
plt.savefig("reports/shap_dependence_plot.png", bbox_inches="tight")
plt.close()


# Save feature importance insights
with open("reports/feature_analysis.txt", "w") as f:
    f.write("Feature Engineering & Explainability Analysis\n")
    f.write("------------------------------------------------\n")
    f.write(f"Top important feature: {top_feature}\n")
    f.write("This feature significantly impacts the model's predictions.\n")
    f.write("Consider refining this feature further if needed.\n")

print("Explainability analysis completed! Reports saved.")
