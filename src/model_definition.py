import torch
import torch.nn as nn

# Define the Swish activation function
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# Define the Attention layer
class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super(AttentionLayer, self).__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        return attn_output

# Define the TimeMix model with GRU and Attention mechanism
class GRUWithAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, seq_len, num_heads=4, num_layers=2):
        super(GRUWithAttention, self).__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # GRU layer
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)

        # Attention layer
        self.attention = AttentionLayer(hidden_dim, num_heads)

        # Fully connected layers with Swish activation
        self.fc_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.fc_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.fc_layers.append(Swish())  # Apply Swish activation after each FC layer

        # Output layer
        self.fc_out = nn.Linear(hidden_dim, output_dim)

        # Time mixing weights
        self.time_mixing_weights = nn.Parameter(torch.ones(seq_len, hidden_dim))
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        # Process through GRU layer
        h, _ = self.gru(x)

        # Apply attention mechanism
        h_att = self.attention(h)

        # Apply time-mixing weights
        mix_weights = self.softmax(self.time_mixing_weights)
        h_mixed = h_att * mix_weights

        # Pass through fully connected layers with Swish activation
        h_mixed = h_mixed[:, -1, :]  # Take the last timestep
        for layer in self.fc_layers:
            h_mixed = layer(h_mixed)  # Apply FC layers followed by Swish activation

        # Final output layer
        out = self.fc_out(h_mixed)
        return out

# Example usage:
# model = GRUWithAttention(input_dim=50, hidden_dim=256, output_dim=1, seq_len=10)

