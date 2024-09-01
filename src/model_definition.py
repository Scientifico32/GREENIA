import torch
import torch.nn as nn

# Define the Swish activation function
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# Define the TimeMix model
class TimeMix(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, seq_len, num_layers=2):
        super(TimeMix, self).__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # GRU layer
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        
        # Fully connected layers with Swish activation
        self.fc_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.fc_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.fc_layers.append(Swish())  # Apply Swish activation after each FC layer
        
        # Output layer
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        
        # Time mixing weights
        self.mix_weights = nn.Parameter(torch.ones(seq_len, hidden_dim))
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        h, _ = self.gru(x)
        mix_weights = self.softmax(self.mix_weights)
        h = h * mix_weights  # Time mixing
        
        # Pass through fully connected layers with Swish activation
        h = h[:, -1, :]  # Take the last timestep
        for layer in self.fc_layers:
            h = layer(h)  # Apply FC layers followed by Swish activation
        
        # Final output layer
        out = self.fc_out(h)
        return out

# Example usage:
# model = TimeMix(input_dim=50, hidden_dim=256, output_dim=1, seq_len=10)
