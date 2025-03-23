import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.datasets import fashion_mnist
from ydata_profiling import ProfileReport
import sweetviz as sv

# Load Fashion MNIST dataset
(X_train, y_train), (_, _) = fashion_mnist.load_data()

# Reduce sample size for faster execution
sample_size = 300  
indices = np.random.choice(X_train.shape[0], sample_size, replace=False)

X_sample = X_train[indices]
y_sample = y_train[indices]

# Flatten images into tabular format
X_sample_flat = X_sample.reshape(sample_size, -1)

# Keep only the first 50 pixel columns (Reduce profiling complexity)
X_sample_flat = X_sample_flat[:, :30]

# Create DataFrame
sample_df = pd.DataFrame(X_sample_flat)
sample_df['label'] = y_sample

# ✅ Generate YData Profiling Report (Optimized)
profile = ProfileReport(
    sample_df, 
    title="Fashion MNIST EDA Report (Optimized)", 
    minimal=True,   # 🚀 Speed optimization
    progress_bar=False
)
profile.to_file("C:/Users/selva/Downloads/MLOPS_ASSIGNMENT_2/reports/fashion_mnist_eda_sample.html")

# ✅ Generate Sweetviz Report (Optimized)

sweetviz_report = sv.analyze(sample_df, pairwise_analysis="off")

sweetviz_report.show_html("C:/Users/selva/Downloads/MLOPS_ASSIGNMENT_2/reports/fashion_mnist_sweetviz_sample.html")

print("✅ EDA reports saved in reports/ folder.")
