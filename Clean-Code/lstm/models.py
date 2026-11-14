"""LSTM autoencoder: encoder -> latent vector -> repeated -> decoder."""

import torch.nn as nn


class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, latent_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.encoder_fc = nn.Linear(hidden_size, latent_dim)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.encoder_fc(h_n[-1])  # last layer's hidden state


class LSTMDecoder(nn.Module):
    def __init__(self, hidden_size, input_size, num_layers, latent_dim):
        super().__init__()
        self.lstm = nn.LSTM(latent_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, z, seq_len):
        out, _ = self.lstm(z.unsqueeze(1).repeat(1, seq_len, 1))
        return self.fc(out)


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, latent_dim):
        super().__init__()
        self.encoder = LSTMEncoder(input_size, hidden_size, num_layers, latent_dim)
        self.decoder = LSTMDecoder(hidden_size, input_size, num_layers, latent_dim)
        self.hparams = {
            "Model": "LSTM",
            "Input Size": input_size,
            "Hidden Size": hidden_size,
            "Latent Dim": latent_dim,
            "Num Layers": num_layers,
        }

    def forward(self, x):
        return self.decoder(self.encoder(x), x.size(1))
