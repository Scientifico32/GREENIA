import torch
import torch.nn as nn

class TimeMixWithAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, seq_len, attention_dim=64):
        super(TimeMixWithAttention, self).__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

        # LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

        # Attention layer
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

        # Time-mixing weights
        self.mix_weights = nn.Parameter(torch.ones(seq_len, hidden_dim))
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        h, _ = self.lstm(x)
        attn_output, _ = self.attention(h, h, h)
        mix_weights = self.softmax(self.mix_weights)
        h = attn_output * mix_weights
        out = self.fc(h[:, -1, :])
        return out
