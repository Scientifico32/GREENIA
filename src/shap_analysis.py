import shap
import numpy as np
import torch
import pandas as pd

# Function to reshape 2D inputs back to 3D for the model
def model_predict_3d(flattened_data):
    reshaped_data = flattened_data.reshape(-1, seq_len, feature_dim // seq_len)
    return model(torch.tensor(reshaped_data, dtype=torch.float32)).detach().numpy()

# Flatten the 3D test data to 2D for SHAP
X_test_flat = X_test_tensor.numpy().reshape(-1, seq_len * (feature_dim // seq_len))

# Initialize the SHAP KernelExplainer with the flat data and the wrapper function
explainer = shap.KernelExplainer(model_predict_3d, X_test_flat[:30])  # Background data

# Calculate SHAP values with the flattened test data
shap_values = explainer.shap_values(X_test_flat)  # Evaluate on a subset

# Ensure that we are working with the correct SHAP values format
if isinstance(shap_values, list):
    shap_values_array = shap_values[0]
else:
    shap_values_array = shap_values

# Get the raw values and average them
vals = np.abs(shap_values_array).mean(0)

# Feature names corresponding to the original data
feature_names = merged_df.drop(columns=['Energy (MWh)', 'Date']).columns

# Truncate the `vals` to match the length of `feature_names`
vals_truncated = vals[:len(feature_names)]

# Create the SHAP values DataFrame with aligned lengths
shap_table = pd.DataFrame({
    'Features': feature_names,
    'Shap value': list(vals_truncated)
})

# Sort the table by SHAP value
shap_table = shap_table.sort_values(by=["Shap value"], ascending=False)

# Round the SHAP values for better readability
shap_table = shap_table.round(3)

# Print or use shap_table as needed
print(shap_table)

# Optionally, plot SHAP values for better visualization
shap.summary_plot(shap_values, X_test_flat, feature_names=feature_names)

