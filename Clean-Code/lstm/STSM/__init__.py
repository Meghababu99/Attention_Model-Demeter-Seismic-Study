"""STSM extension: LSTM / GRU, uni- or bidirectional, with optional attention in the encoder."""

from .models import RecurrentAutoencoder, model_name

__all__ = ["RecurrentAutoencoder", "model_name"]
