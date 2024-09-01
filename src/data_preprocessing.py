import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def preprocess_data(merged_df, seq_len=10):
    # Drop the 'Energy (MWh)' and 'Date' column and convert the rest to a NumPy array
    X = merged_df.drop(columns=['Energy (MWh)', 'Date']).values
    y = merged_df['Energy (MWh)'].values

    # Scale the features and the target variable
    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)

    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    # Pad X to 50 features for divisibility by sequence length of 10
    feature_dim = 50  # Adjust this as needed
    X_padded = np.pad(X_scaled, ((0, 0), (0, feature_dim - X_scaled.shape[1])), 'constant')

    # Determine the split index (e.g., 80% for training and 20% for testing)
    split_index = int(0.8 * len(X_padded))

    # Split the data chronologically
    X_train, X_test = X_padded[:split_index], X_padded[split_index:]
    y_train, y_test = y_scaled[:split_index], y_scaled[split_index:]

    # Convert the data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).view(-1, seq_len, feature_dim // seq_len)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).view(-1, seq_len, feature_dim // seq_len)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, scaler_y

# Example usage:
# X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, scaler_y = preprocess_data(merged_df)


