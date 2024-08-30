import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def data_preprocessing(merged_df):
    # Feature and target extraction
    X = merged_df.drop(columns=['Energy (MWh)', 'Date']).values
    y = merged_df['Energy (MWh)'].values

    # Scaling the features and the target variable
    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)

    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    # Pad X to 50 features for divisibility by sequence length of 10
    X_padded = np.pad(X_scaled, ((0, 0), (0, 3)), 'constant')

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_padded, y_scaled, test_size=0.2, random_state=42)

    # Convert the data to PyTorch tensors for the GRU and LSTM models
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).view(-1, 10, 5)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).view(-1, 10, 5)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

    # Flatten the 3D tensor to 2D for XGBoost
    X_train_xgb = X_train_tensor.view(X_train_tensor.shape[0], -1).numpy()
    X_val_xgb = X_val_tensor.view(X_val_tensor.shape[0], -1).numpy()
    y_train_xgb = y_train_tensor.numpy()
    y_val_xgb = y_val_tensor.numpy()

    return (X_train_tensor, X_val_tensor, y_train_tensor, y_val_tensor,
            X_train_xgb, X_val_xgb, y_train_xgb, y_val_xgb,
            scaler_y)

# Usage
X_train_tensor, X_val_tensor, y_train_tensor, y_val_tensor, X_train_xgb, X_val_xgb, y_train_xgb, y_val_xgb, scaler_y = data_preprocessing(merged_df)

