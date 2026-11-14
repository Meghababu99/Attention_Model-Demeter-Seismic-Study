"""Lean LSTM-autoencoder package for the DEMETER rolling-window training pipeline."""

from .dataset import HalfOrbitPairDataset
from .scaling import scale_datasets
from .models import LSTMAutoencoder
from .anomaly_detection import AnomalyDetector
from .SeismicCriteria import SeismicCriteria
from .Threshold_anomaly import SeismicAnalysis
from .training_modes import train_lstm_ae_mode
from .pipeline import (
    generate_windows,
    load_eq_catalog,
    prepare_window,
    train_window,
    process_split,
    run_seismic_analysis,
    plot_error_distributions,
    evaluate_window,
)

__all__ = [
    "HalfOrbitPairDataset",
    "scale_datasets",
    "LSTMAutoencoder",
    "AnomalyDetector",
    "SeismicCriteria",
    "SeismicAnalysis",
    "train_lstm_ae_mode",
    "generate_windows",
    "load_eq_catalog",
    "prepare_window",
    "train_window",
    "process_split",
    "run_seismic_analysis",
    "plot_error_distributions",
    "evaluate_window",
]
