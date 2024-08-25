import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import os
import pickle
from model_definition import TimeMixWithAttention

# Assuming merged_df has been processed and saved
merged_df = pd.read_csv('/content/drive/MyDrive/TimeMixWithAttention/processed_data/merged_df.csv')

# Prepare data for training
X = merged_df.drop(columns=['Energy (MWh)', 'Date']).values
y = merged_df['Energy (MWh)'].values

scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

X_padded = np.pad(X_scaled, ((0, 0), (0, 3)), 'constant')
X_train, X_test, y_train, y_test = train_test_split(X_padded, y_scaled, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).view(-1, 10, 50 // 10)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).view(-1, 10, 50 // 10)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

input_dim = X_train_tensor.shape[2]
model = TimeMixWithAttention(input_dim, hidden_dim=256, output_dim=1, seq_len=10)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0008)

training_losses = []

for epoch in range(2000):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs.squeeze(), y_train_tensor)
    loss.backward()
    optimizer.step()
    training_losses.append(loss.item())
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/2000], Loss: {loss.item():.4f}')

model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor).squeeze()
    test_loss = criterion(test_outputs, y_test_tensor)
    print(f'Test Loss: {test_loss.item():.4f}')

    y_test_orig = scaler_y.inverse_transform(y_test_tensor.numpy().reshape(-1, 1)).flatten()
    test_outputs_orig = scaler_y.inverse_transform(test_outputs.numpy().reshape(-1, 1)).flatten()

    mae = mean_absolute_error(y_test_orig, test_outputs_orig)
    rmse = np.sqrt(mean_squared_error(y_test_orig, test_outputs_orig))
    mape = mean_absolute_percentage_error(y_test_orig, test_outputs_orig)

    print(f"MAE: {mae}")
    print(f"RMSE: {rmse}")
    print(f"MAPE: {mape}%")

    plt.figure(figsize=(10, 6))
    plt.plot(y_test_orig, label='Actual Values', marker='o', color='green')
    plt.plot(test_outputs_orig, label='Forecasted Values', marker='o', color='blue')
    plt.title('Forecast vs Actual Values')
    plt.xlabel('Samples')
    plt.ylabel('Energy Production')
    plt.legend()
    plt.show()

# Save model, scalers, and training log
save_dir = '/content/drive/MyDrive/TimeMixWithAttention'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

model_save_path = os.path.join(save_dir, 'timemix_with_attention_model.pth')
torch.save(model.state_dict(), model_save_path)

scaler_X_path = os.path.join(save_dir, 'scaler_X.pkl')
scaler_y_path = os.path.join(save_dir, 'scaler_y.pkl')
with open(scaler_X_path, 'wb') as f:
    pickle.dump(scaler_X, f)
with open(scaler_y_path, 'wb') as f:
    pickle.dump(scaler_y, f)

log_save_path = os.path.join(save_dir, 'training_log.csv')
with open(log_save_path, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=["epoch", "loss"])
    writer.writeheader()
    for epoch, loss in enumerate(training_losses):
        writer.writerow({'epoch': epoch, 'loss': loss})