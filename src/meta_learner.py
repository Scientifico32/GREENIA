import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Custom Activation Function: Swish
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# Define the Neural Network Meta-Learner
class MetaLearnerNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MetaLearnerNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.custom_activation = Swish()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.output_activation = nn.Identity()  # No activation at the output layer for regression tasks

    def forward(self, x):
        x = self.fc1(x)
        x = self.custom_activation(x)
        x = self.fc2(x)
        x = self.custom_activation(x)
        x = self.fc3(x)
        return self.output_activation(x)

def create_lagged_features(data, lags):
    X, y = [], []
    for i in range(lags, len(data)):
        X.append(data[i-lags:i])
        y.append(data[i])
    return np.array(X), np.array(y)

# Step 1: Use predictions from the ensemble method as features for the neural network
def prepare_meta_data(ensemble_pred_orig, lags=7):
    X_meta, y_meta = create_lagged_features(ensemble_pred_orig, lags)

    # Split the data into training and test sets
    X_train_meta, X_test_meta, y_train_meta, y_test_meta = train_test_split(X_meta, y_meta, test_size=0.2, random_state=42)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_meta, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_meta, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_meta, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_meta, dtype=torch.float32)

    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor

def train_meta_learner(X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, input_dim, hidden_dim=128, output_dim=1, num_epochs=1000, patience=100):
    # Initialize the Meta-Learner Neural Network
    meta_learner_nn = MetaLearnerNN(input_dim, hidden_dim, output_dim)

    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(meta_learner_nn.parameters(), lr=0.01)

    # Training loop
    min_val_loss = float('inf')
    early_stopping_counter = 0

    for epoch in range(num_epochs):
        meta_learner_nn.train()
        optimizer.zero_grad()
        outputs = meta_learner_nn(X_train_tensor).squeeze()
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        # Validation
        meta_learner_nn.eval()
        with torch.no_grad():
            val_outputs = meta_learner_nn(X_test_tensor).squeeze()
            val_loss = criterion(val_outputs, y_test_tensor)

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

    return meta_learner_nn

# Step 3: Forecast 7 Days Ahead Using the Trained Meta-Learner NN
def sequential_forecast_nn(model, initial_input, steps):
    predictions = []
    current_input = initial_input

    for _ in range(steps):
        with torch.no_grad():
            pred = model(current_input.unsqueeze(0)).squeeze().item()
        predictions.append(pred)
        current_input = torch.cat((current_input[1:], torch.tensor([pred], dtype=torch.float32)))

    return np.array(predictions)

def meta_learner_forecast(meta_learner_nn, X_train_tensor, y_val, scaler_y, steps=7):
    initial_input = X_train_tensor[-1]  # Start from the last known data point in the training set
    forecast_7_days = sequential_forecast_nn(meta_learner_nn, initial_input, steps)

    # If you have actual future values, compare them
    actual_future_values_7_days = y_val[-7:]
    actual_future_values_7_days_orig = scaler_y.inverse_transform(actual_future_values_7_days.reshape(-1, 1)).flatten()

    # Plot the forecasted values against the actual values
    plt.figure(figsize=(10, 6))
    plt.plot(range(steps), actual_future_values_7_days_orig, label='Actual Values (Last 7 Days)', marker='o', color='green')
    plt.plot(range(steps), forecast_7_days, label='7 Days Ahead Forecast', marker='o', color='blue')
    plt.title('7 Days Ahead Forecast vs Actual Values')
    plt.xlabel('Days')
    plt.ylabel('Energy Production')
    plt.legend()
    plt.show()
