import numpy as np
import torch
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

def forecast_xgb_gru(model, X_train_tensor, X_initial, n_days, n_estimators=100, max_depth=3, learning_rate=0.1):
    model.eval()
    
    # Use the training data to train the XGBoost model
    with torch.no_grad():
        gru_train_outputs = model(X_train_tensor).squeeze().numpy()

    X_train_reshaped = X_train_tensor.view(X_train_tensor.shape[0], -1).numpy()  # Flatten features

    # Train XGBoost model on the GRU model's outputs
    xgb_model = xgb.XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=17)
    xgb_model.fit(X_train_reshaped, gru_train_outputs)

    # Forecast future values
    forecast = []
    current_input = X_initial[-1].view(1, -1).numpy()  # Start with the last input features

    for _ in range(n_days):
        next_pred = xgb_model.predict(current_input)  # Prediction as 1D array
        forecast.append(next_pred.item())

        # Prepare the next input for prediction
        current_input = np.roll(current_input, -1, axis=1)
        current_input[:, -1] = next_pred  # Update the last feature with the prediction
    
    return np.array(forecast)

def plot_forecast_vs_actual(forecasted_values, actual_values, n_days):
    plt.figure(figsize=(10, 6))
    plt.plot(range(n_days), actual_values, label='Actual Values', marker='o', color='green')
    plt.plot(range(n_days), forecasted_values, label='Forecasted Values', marker='o', color='blue')
    plt.title(f'{n_days}-Day Forecast vs Actual Values')
    plt.xlabel('Day')
    plt.ylabel('Energy Production')
    plt.legend()
    plt.show()

# Example usage:
# X_initial = X_test_tensor[-1:]  # Last sequence of the test set
# n_days = 30
# forecasted_values = forecast_xgb_gru(model, X_train_tensor, X_initial, n_days)

# Inverse transform the forecasts to the original scale
# forecasted_values_orig = scaler_y.inverse_transform(forecasted_values.reshape(-1, 1)).flatten()

# Get the actual values for the next n_days from the test set
# actual_values_orig = scaler_y.inverse_transform(y_test_tensor[-n_days:].reshape(-1, 1)).flatten()

# Plot the forecasted values vs the actual values
# plot_forecast_vs_actual(forecasted_values_orig, actual_values_orig, n_days)

# Print the forecasted and actual values
# print(f"Forecasted Energy Production for the next {n_days} days: {forecasted_values_orig}")
# print(f"Actual Energy Production for the next {n_days} days: {actual_values_orig}")
