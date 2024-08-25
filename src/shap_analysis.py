import shap
import torch
import numpy as np
import pickle
import os

from model_training import model, X_test_tensor, scaler_X, scaler_y
from data_preprocessing import merged_df

def model_predict_3d(flattened_data):
    reshaped_data = flattened_data.reshape(-1, 10, 50 // 10)
    return model(torch.tensor(reshaped_data, dtype=torch.float32)).detach().numpy()

X_test_flat = X_test_tensor.numpy().reshape(-1, 10 * (50 // 10))
explainer = shap.KernelExplainer(model_predict_3d, X_test_flat[:30])
shap_values = explainer.shap_values(X_test_flat)

vals = np.abs(shap_values).mean(0)
feature_names = merged_df.drop(columns=['Energy (MWh)', 'Date']).columns
vals_truncated = vals[:len(feature_names)]

shap_table = pd.DataFrame({
    'Features': feature_names,
    'Shap value': list(vals_truncated)
})
shap_table = shap_table.sort_values(by=["Shap value"], ascending=False).round(3)

save_dir = '/content/drive/MyDrive/TimeMixWithAttention'
explainer_save_path = os.path.join(save_dir, 'shap_explainer.pkl')
shap_values_save_path = os.path.join(save_dir, 'shap_values.pkl')

with open(explainer_save_path, 'wb') as f:
    pickle.dump(explainer, f)
with open(shap_values_save_path, 'wb') as f:
    pickle.dump(shap_values, f)

print(f"SHAP explainer saved to {explainer_save_path}")
print(f"SHAP values saved to {shap_values_save_path}")
print(shap_table)
