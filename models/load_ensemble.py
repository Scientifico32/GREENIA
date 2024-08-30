import joblib
# Load models
xgb_model = joblib.load(os.path.join(save_dir, 'xgboost_model.pkl'))

model2 = SimpleGRU(input_dim=5, hidden_dim=128, output_dim=1, seq_len=10)
model2.load_state_dict(torch.load(os.path.join(save_dir, 'gru_model.pth')))
model2.eval()

model3 = SimpleLSTM(input_dim=5, hidden_dim=128, output_dim=1, seq_len=10)
model3.load_state_dict(torch.load(os.path.join(save_dir, 'lstm_model.pth')))
model3.eval()

# Load scalers
scaler_X = joblib.load(os.path.join(save_dir, 'scaler_X.pkl'))
scaler_y = joblib.load(os.path.join(save_dir, 'scaler_y.pkl'))

# Load ensemble predictions and actual values (if needed)
ensemble_pred = torch.tensor(np.load(os.path.join(save_dir, 'ensemble_predictions.npy')))
actual_values = torch.tensor(np.load(os.path.join(save_dir, 'actual_values.npy')))
