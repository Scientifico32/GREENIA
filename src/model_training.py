import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

def train_model(model, X_train_tensor, y_train_tensor, learning_rate=0.0008, num_epochs=1000):
    # Initialize the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        outputs = model(X_train_tensor)
        loss = criterion(outputs.squeeze(), y_train_tensor)

        # Backward and optimize
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

def evaluate_model(model, X_test_tensor, y_test_tensor, scaler_y):
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor).squeeze()
        test_loss = nn.MSELoss()(test_outputs, y_test_tensor)
        print(f'Test Loss: {test_loss.item():.4f}')

        # Inverse transform to original scale
        y_test_orig = scaler_y.inverse_transform(y_test_tensor.numpy().reshape(-1, 1)).flatten()
        test_outputs_orig = scaler_y.inverse_transform(test_outputs.numpy().reshape(-1, 1)).flatten()

        # Calculate evaluation metrics
        mae = mean_absolute_error(y_test_orig, test_outputs_orig)
        rmse = np.sqrt(mean_squared_error(y_test_orig, test_outputs_orig))
        mape = mean_absolute_percentage_error(y_test_orig, test_outputs_orig)

        print(f"MAE: {mae}")
        print(f"RMSE: {rmse}")
        print(f"MAPE: {mape}%")

        # Plot the forecasted values vs the actual values
        plt.figure(figsize=(10, 6))
        plt.plot(y_test_orig, label='Actual Values', marker='o', color='green')
        plt.plot(test_outputs_orig, label='Forecasted Values', marker='o', color='blue')
        plt.title('Forecast vs Actual Values')
        plt.xlabel('Samples')
        plt.ylabel('Energy Production')
        plt.legend()
        plt.show()

# Example usage:
# train_model(model, X_train_tensor, y_train_tensor)
# evaluate_model(model, X_test_tensor, y_test_tensor, scaler_y)


