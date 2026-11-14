"""
Recurrent autoencoder with a selectable recurrent cell and encoder summary.

    cell:          "LSTM" or "GRU"
    bidirectional: False / True (applies to encoder and decoder)
    attention:     None               -> last hidden state of the top layer (baseline)
                   "temporal"         -> softmax-weighted sum of all encoder outputs over time
                   "feature_temporal" -> softmax feature weights on the input, then temporal attention

The decoder is the baseline decoder: the latent vector is repeated over time,
passed through the recurrent layers, and mapped back to the input features.

With cell="LSTM", bidirectional=False this reproduces:
    attention=None               -> lstm.models.LSTMAutoencoder
    attention="temporal"         -> TemporalAttentionLSTMAutoencoder (attention_models.py)
    attention="feature_temporal" -> FeatureTemporalAttentionLSTMAutoencoder (STSM attention_models.py)
and loads their saved .pth files.
"""

import torch
import torch.nn as nn

CELLS = {"LSTM": nn.LSTM, "GRU": nn.GRU}
ATTENTIONS = (None, "temporal", "feature_temporal")


def model_name(cell="LSTM", bidirectional=False, attention=None):
    """Short name used in folder and file tags, e.g. 'LSTM', 'BiGRU', 'LSTM-TA', 'BiLSTM-FTA'."""
    suffix = {None: "", "temporal": "-TA", "feature_temporal": "-FTA"}[attention]
    return f"{'Bi' if bidirectional else ''}{cell}{suffix}"


class RecurrentEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, latent_dim, cell, bidirectional, attention):
        super().__init__()
        if attention not in ATTENTIONS:
            raise ValueError(f"attention must be one of {ATTENTIONS}")
        out_size = hidden_size * (2 if bidirectional else 1)
        self.bidirectional = bidirectional

        self.feature_attention = nn.Linear(input_size, input_size) if attention == "feature_temporal" else None
        self.rnn = CELLS[cell](input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        self.temporal_attention = nn.Linear(out_size, 1) if attention else None
        self.encoder_fc = nn.Linear(out_size, latent_dim)

    def forward(self, x, return_attention=False):
        weights = {}
        if self.feature_attention is not None:
            weights["feature"] = torch.softmax(self.feature_attention(x), dim=2)  # over features
            x = x * weights["feature"]

        outputs, hidden = self.rnn(x)  # outputs: [B, T, H * directions]

        if self.temporal_attention is not None:
            weights["temporal"] = torch.softmax(self.temporal_attention(outputs), dim=1)  # over time
            summary = (weights["temporal"] * outputs).sum(dim=1)
        else:
            h_n = hidden[0] if isinstance(hidden, tuple) else hidden  # LSTM returns (h_n, c_n)
            summary = torch.cat([h_n[-2], h_n[-1]], dim=1) if self.bidirectional else h_n[-1]  # top layer

        z = self.encoder_fc(summary)
        return (z, weights) if return_attention else z


class RecurrentDecoder(nn.Module):
    def __init__(self, hidden_size, input_size, num_layers, latent_dim, cell, bidirectional):
        super().__init__()
        self.rnn = CELLS[cell](latent_dim, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1), input_size)

    def forward(self, z, seq_len):
        out, _ = self.rnn(z.unsqueeze(1).repeat(1, seq_len, 1))
        return self.fc(out)


class RecurrentAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, latent_dim,
                 cell="LSTM", bidirectional=False, attention=None):
        super().__init__()
        self.encoder = RecurrentEncoder(input_size, hidden_size, num_layers, latent_dim, cell, bidirectional, attention)
        self.decoder = RecurrentDecoder(hidden_size, input_size, num_layers, latent_dim, cell, bidirectional)
        self.hparams = {
            "Model": model_name(cell, bidirectional, attention),
            "Input Size": input_size,
            "Hidden Size": hidden_size,
            "Latent Dim": latent_dim,
            "Num Layers": num_layers,
        }

    def forward(self, x, return_attention=False):
        z, weights = self.encoder(x, return_attention=True)
        decoded = self.decoder(z, x.size(1))
        return (decoded, weights) if return_attention else decoded

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        """Also accepts checkpoints of the original LSTM models (layer name 'lstm' instead of 'rnn')."""
        state_dict = {k.replace(".lstm.", ".rnn."): v for k, v in state_dict.items()}
        return super().load_state_dict(state_dict, strict=strict, **kwargs)
