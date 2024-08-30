import torch.nn as nn

# Define the GRU model
class SimpleGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, seq_len):
        super(SimpleGRU, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h, _ = self.gru(x)
        out = self.fc(h[:, -1, :])
        return out.view(-1, 1)

# Define the LSTM model
class SimpleLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, seq_len):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h, _ = self.lstm(x)
        out = self.fc(h[:, -1, :])
        return out.view(-1, 1)
