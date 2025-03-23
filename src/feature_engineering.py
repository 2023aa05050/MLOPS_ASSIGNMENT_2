import numpy as np
import pandas as pd
from tensorflow.keras.datasets import fashion_mnist
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Load Fashion MNIST dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
sample_size = 3000  # Adjust this as needed
x_train, y_train = x_train[:sample_size], y_train[:sample_size]
x_test, y_test = x_test[:sample_size], y_test[:sample_size]

# Reshape data for scaling (convert 2D images into 1D features)
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

# Normalize pixel values (0-255) to range [0,1]
scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Convert to DataFrame for easier analysis
df_train = pd.DataFrame(x_train_scaled)
df_test = pd.DataFrame(x_test_scaled)

# Save processed data
df_train.to_csv("C:/Users/selva/Downloads/MLOPS_ASSIGNMENT_2/data/train_features.csv", index=False)
df_test.to_csv("C:/Users/selva/Downloads/MLOPS_ASSIGNMENT_2/data/test_features.csv", index=False)

print("✅ Feature engineering completed! Processed data saved.")
