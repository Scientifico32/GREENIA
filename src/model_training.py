import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from model_definition import GRUWithAttention  # Assuming your model is saved in model_definition.py

# Define the training function
def train_model(model, X_train_tensor, y_train_tensor, learning_rate=0.0008, num_epochs=2000):
    # Initialize the loss function and optimizer
    criterion = nn.MSELoss()  # Mean Squared Error Loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        optimizer.zero_grad()

        # Forward pass through the model
        outputs = model(X_train_tensor)
        loss = criterion(outputs.squeeze(), y_train_tensor)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Log the training loss
        if (epoch + 1) % 100 == 0:  # Log every 100 epochs
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Define the evaluation function
def evaluate_model(model, X_test_tensor, y_test_tensor, scaler_y):
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        # Make predictions
        test_outputs = model(X_test_tensor).squeeze()
        test_loss = nn.MSELoss()(test_outputs, y_test_tensor)
        print(f'Test Loss: {test_loss.item():.4f}')

        # Inverse transform to original scale
        y_test_orig = scaler_y.inverse_transform(y_test_tensor.numpy().reshape(-1, 1)).flatten()
        test_outputs_orig = scaler_y.inverse_transform(test_outputs.numpy().reshape(-1, 1)).flatten()

        # Calculate evaluation metrics: MAE, RMSE, MAPE
        mae = mean_absolute_error(y_test_orig, test_outputs_orig)
        rmse = np.sqrt(mean_squared_error(y_test_orig, test_outputs_orig))
        mape = mean_absolute_percentage_error(y_test_orig, test_outputs_orig)

        print(f"MAE: {mae}")
        print(f"RMSE: {rmse}")
        print(f"MAPE: {mape * 100}%")

        # Plot actual vs forecasted values
        plt.figure(figsize=(10, 6))
        plt.plot(y_test_orig, label='Actual Values', marker='o', color='green')
        plt.plot(test_outputs_orig, label='Forecasted Values', marker='o', color='blue')
        plt.title('Forecast vs Actual Values')
        plt.xlabel('Samples')
        plt.ylabel('Energy Production')
        plt.legend()
        plt.show()

# Example usage
# Assuming X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, scaler_y are defined elsewhere
# model = GRUWithAttention(input_dim=50, hidden_dim=256, output_dim=1, seq_len=10)
# train_model(model, X_train_tensor, y_train_tensor)
# evaluate_model(model, X_test_tensor, y_test_tensor, scaler_y)


