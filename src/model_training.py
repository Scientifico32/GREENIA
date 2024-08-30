import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import xgboost as xgb

# Helper function to train the GRU and LSTM models
def train_model(model, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, num_epochs=1000, learning_rate=0.1, patience=100, min_epochs=300):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20, min_lr=1e-6)

    best_val_loss = float('inf')
    early_stopping_counter = 0

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        outputs = model(X_train_tensor)
        loss = criterion(outputs.squeeze(), y_train_tensor)

        # Backward and optimize
        loss.backward()
        optimizer.step()

        # Validation step
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor).squeeze()
            val_loss = criterion(val_outputs, y_val_tensor)

        # Step the scheduler
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if (epoch + 1) >= min_epochs and early_stopping_counter >= patience:
            print(f'Early stopping at epoch {epoch + 1}')
            break

        if (epoch + 1) % 1 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')

    return model

# Training XGBoost model
def train_xgb(X_train_xgb, y_train_xgb, X_val_xgb, y_val_xgb):
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        learning_rate=0.01,
        n_estimators=1000,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=20
    )

    xgb_model.fit(
        X_train_xgb, y_train_xgb,
        eval_set=[(X_val_xgb, y_val_xgb)],
        verbose=True
    )
    return xgb_model

# Model 1: Train XGBoost
xgb_model = train_xgb(X_train_xgb, y_train_xgb, X_val_xgb, y_val_xgb)
pred1 = torch.tensor(xgb_model.predict(X_val_xgb))

# Model 2: Train Simple GRU
model2 = SimpleGRU(input_dim=5, hidden_dim=128, output_dim=1, seq_len=10)
model2 = train_model(model2, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor)

# Model 3: Train Simple LSTM
model3 = SimpleLSTM(input_dim=5, hidden_dim=128, output_dim=1, seq_len=10)
model3 = train_model(model3, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor)

# Model Evaluation and Ensemble
def evaluate_models(model2, model3, pred1, X_val_tensor, y_val_tensor, scaler_y):
    # Switch to evaluation mode for GRU and LSTM models
    model2.eval()
    model3.eval()

    with torch.no_grad():
        pred2 = model2(X_val_tensor).squeeze()
        pred3 = model3(X_val_tensor).squeeze()

    # Simple Averaging of the predictions from XGBoost, GRU, and LSTM
    ensemble_pred = (pred1 + pred2 + pred3) / 3

    # Inverse transform the ensemble predictions to the original scale
    ensemble_pred_orig = scaler_y.inverse_transform(ensemble_pred.numpy().reshape(-1, 1)).flatten()

    # Get the actual values
    actual_values_orig = scaler_y.inverse_transform(y_val_tensor.numpy().reshape(-1, 1)).flatten()

    # Calculate evaluation metrics
    mae = mean_absolute_error(actual_values_orig, ensemble_pred_orig)
    rmse = np.sqrt(mean_squared_error(actual_values_orig, ensemble_pred_orig))
    mape = mean_absolute_percentage_error(actual_values_orig, ensemble_pred_orig)

    print(f"Ensemble MAE: {mae}")
    print(f"Ensemble RMSE: {rmse}")
    print(f"Ensemble MAPE: {mape}%")

    # Plot the ensemble forecasted values vs the actual values
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(actual_values_orig)), actual_values_orig, label='Actual Values', marker='o', color='green')
    plt.plot(range(len(ensemble_pred_orig)), ensemble_pred_orig, label='Ensemble Forecasted Values', marker='o', color='blue')
    plt.title('Ensemble Forecast vs Actual Values')
    plt.xlabel('Samples')
    plt.ylabel('Energy Production')
    plt.legend()
    plt.show()

# Usage
evaluate_models(model2, model3, pred1, X_val_tensor, y_val_tensor, scaler_y)

