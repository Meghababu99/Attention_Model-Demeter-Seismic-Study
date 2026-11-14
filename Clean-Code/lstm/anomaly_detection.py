"""Reconstruction-error based anomaly detection (aggregate over all features)."""

import numpy as np
import torch


class AnomalyDetector:
    def __init__(self, model):
        self.model = model.eval()
        self.device = next(model.parameters()).device
        self.criterion = torch.nn.MSELoss()

    def compute_reconstruction_errors_agg(self, dataloader):
        """Mean MSE of the two half-orbits, one value per batch (use batch_size=1)."""
        errors = []
        with torch.no_grad():
            for x1, x2 in dataloader:
                x1, x2 = x1.to(self.device), x2.to(self.device)
                loss = (self.criterion(self.model(x1), x1) + self.criterion(self.model(x2), x2)) / 2
                errors.append(loss.item())
        return np.array(errors)

    def detect_anomalies_agg(self, errors, threshold):
        """Indices whose error is >= threshold."""
        return np.flatnonzero(errors >= threshold).tolist()
