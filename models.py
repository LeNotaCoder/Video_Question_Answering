import torch
import torch.nn as nn
import torch.nn.functional as F

class Final_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(Final_Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM to process sequences of frame embeddings
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        # Projection to CLIP's 512-dim embedding space
        self.proj = nn.Linear(hidden_dim, 512)

    def forward(self, x):
        # x: (batch_size, num_frames, input_dim)
        _, (h_n, _) = self.lstm(x)          # h_n: (num_layers, batch_size, hidden_dim)
        context_vector = h_n[-1]            # Take last layer hidden state: (batch_size, hidden_dim)
        
        clip_vector = self.proj(context_vector)  # (batch_size, 512)
        clip_vector = F.normalize(clip_vector, dim=-1)  # L2 normalize to match CLIP behavior
        
        return clip_vector
