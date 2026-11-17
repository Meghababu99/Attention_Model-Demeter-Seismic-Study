"""
Rolling-window pipeline steps shared by the training and the pretrained-evaluation notebooks.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import DataLoader

from .anomaly_detection import AnomalyDetector
from .dataset import HalfOrbitPairDataset
from .scaling import scale_datasets
from .Threshold_anomaly import SeismicAnalysis
from .training_modes import train_lstm_ae_mode

SPLITS = ("Train", "Val")


def generate_windows(start_date, end_date, train_months, val_months, stride):
    """Rolling train/validation windows, shifted by `stride` months."""
    windows = []
    current, final = pd.to_datetime(start_date), pd.to_datetime(end_date)
    while current + pd.DateOffset(months=train_months + val_months) < final:
        train_end = current + pd.DateOffset(months=train_months)
        windows.append({
            "train_start": current,
            "train_end": train_end,
            "val_start": train_end,
            "val_end": train_end + pd.DateOffset(months=val_months),
        })
        current += pd.DateOffset(months=stride)
    return windows


def load_eq_catalog(eq_path):
    """Earthquake catalogue with 'Time' as timezone-free datetime64 in whole seconds (needed by SeismicCriteria.is_eq)."""
    eq = pd.read_csv(eq_path)
    eq["Time"] = pd.to_datetime(pd.to_datetime(eq["Time"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S"))
    return eq


def prepare_window(bg_window_path, label_paths, w, min_data_points):
    """
    Split one background-corrected window into train / val and build its datasets.

    Args:
        bg_window_path: Background_data-window_{i}.pkl
        label_paths: {"Train": train label csv, "Val": val label csv}
        w: window dict from generate_windows

    Returns:
        splits:   {"Train": df, "Val": df}
        datasets: {"Train": all pairs, "Val": all pairs}
        scaled:   {"Train_normal", "Val_normal", "Train", "Val"}, scaler fitted on normal training pairs
    """
    dfw = pd.read_pickle(bg_window_path)
    dfw = dfw.loc[:, ~dfw.columns.str.startswith("Q3")]
    dfw.index = pd.to_datetime(dfw.index)

    splits = {
        "Train": dfw[(dfw.index >= w["train_start"]) & (dfw.index < w["train_end"])],
        "Val": dfw[(dfw.index >= w["val_start"]) & (dfw.index < w["val_end"])],
    }
    datasets = {s: HalfOrbitPairDataset(splits[s], min_data_points=min_data_points) for s in SPLITS}
    normal = {
        s: HalfOrbitPairDataset(splits[s], pd.read_csv(label_paths[s]), min_data_points, use_label_0_only=True)
        for s in SPLITS
    }

    scaled = dict(zip(
        ("Train_normal", "Val_normal", "Train", "Val"),
        scale_datasets(normal["Train"], normal["Train"], normal["Val"], datasets["Train"], datasets["Val"]),
    ))
    print("Pairs: " + ", ".join(f"{k}={len(v)}" for k, v in scaled.items()))
    return splits, datasets, scaled


def train_window(model, scaled, tag, model_dir, batch_size, prev_model_path=None, save_all_epochs=True, **train_kwargs):
    """
    Train on the normal training pairs of one window, with early stopping on the normal
    validation pairs. Optionally start from `prev_model_path` (warm start).

    Saves in `model_dir`: best_model_{tag}.pth, check_{tag}/ (every epoch), loss_curve_model_{tag}.png,
    loss_history_{tag}.csv. `train_kwargs` go to train_lstm_ae_mode (num_epochs, lr, patience, ...).
    """
    model_dir = Path(model_dir)
    if prev_model_path is not None:
        model.load_state_dict(torch.load(prev_model_path, map_location="cpu"))
        print(f"[Warm start] Loaded {Path(prev_model_path).name}")

    model, train_losses, val_losses = train_lstm_ae_mode(
        model,
        DataLoader(scaled["Train_normal"], batch_size=batch_size, shuffle=True),
        DataLoader(scaled["Val"], batch_size=batch_size, shuffle=False),
        best_model_path=str(model_dir / f"best_model_{tag}.pth"),
        save_all_epochs=save_all_epochs,
        all_epochs_dir=str(model_dir / f"check_{tag}"),
        plot_loss=True,
        loss_plot_path=str(model_dir / f"loss_curve_model_{tag}.png"),
        **train_kwargs,
    )
    pd.DataFrame({
        "epoch": np.arange(1, len(train_losses) + 1),
        "train_loss": train_losses,
        "val_loss": val_losses,
    }).to_csv(model_dir / f"loss_history_{tag}.csv", index=False)
    return model


def process_split(model, dataloader, threshold_agg=None, threshold_percentile=None):
    """Aggregate reconstruction errors and anomalies. Threshold = given value or percentile of these errors."""
    detector = AnomalyDetector(model)
    
    errors_agg = detector.compute_reconstruction_errors_agg(dataloader)
    if threshold_agg is None:
        threshold_agg = np.percentile(errors_agg, threshold_percentile)
    return {
        "errors_agg": errors_agg,
        "threshold_agg": threshold_agg,
        "anomalies_agg": detector.detect_anomalies_agg(errors_agg, threshold_agg),
    }


def run_seismic_analysis(split_df, dataset, eq, seismic_criteria, anomalous_indices):
    """Percentage of anomalous sequences that are seismic, its binomial error, and matched earthquakes."""
    sa = SeismicAnalysis(split_df, eq, dataset.create_half_orbit_sequences, seismic_criteria.is_eq)
    seismic_indices, matched_eqs = sa.agg_analysis(anomalous_indices)

    n, k = len(anomalous_indices), len(seismic_indices)
    if n > 0:
        if k == n:
            p = (k - 1) / n
        elif k == 0:
            p = 1 / n
        else:
            p = k / n
        value, error = p * 100, np.sqrt(p * (1 - p) / n) * 100
    else:
        value, error = 0, 0

    return {"value": value, "error": error, "total_eq": len(matched_eqs), "seismic_indices": seismic_indices}


def plot_error_distributions(i, m, train, val, pc, output_dir):
    """KDE of log aggregate errors for train and validation with their pc-th percentile lines."""
    plt.figure(figsize=(13, 7))
    for label, res, color in ((f"Train_w{i}", train, "tab:blue"), (f"Validation_w{i}", val, "tab:orange")):
        log_err = np.log(res["errors_agg"])
        thr = np.percentile(log_err, pc)
        sns.histplot(log_err, bins=50, stat="density", element="step", fill=False, color=color, label=label)
        plt.axvline(thr, color=color, ls="--", lw=1.8)
        plt.text(thr, plt.ylim()[1] * 0.80, f"{thr:.2f}", rotation=90, color=color,
                 va="bottom", ha="right", fontsize=10)

    plt.xlabel("Error (log)")
    plt.ylabel("Density")
    plt.title(f"Reconstruction error distributions with {pc}th percentile thresholds")
    plt.legend(title="Datasets", loc="upper right")
    plt.tight_layout()
    plt.savefig(Path(output_dir) / f"Aggregate_Errordistribution-pc{pc}-HP_m{m}_w{i}.png", dpi=300, bbox_inches="tight")
    plt.close()

def evaluate_window(i, w, model, splits, datasets, scaled, eq, seismic_criteria, pc, m, output_dir, match_a_rng=False):
    """
    Evaluate a trained model on all train and validation pairs of window i.
    The anomaly threshold is the pc-th percentile of the training errors.

    Returns:
        Two result rows (Val, Train).
    """

    train = process_split(model, DataLoader(scaled["Train"], batch_size=1),
                          threshold_percentile=pc, match_a_rng=match_a_rng)
    val = process_split(model, DataLoader(scaled["Val"], batch_size=1),
                        threshold_agg=train["threshold_agg"], match_a_rng=match_a_rng)

    # train = process_split(model, DataLoader(scaled["Train"], batch_size=1), threshold_percentile=pc)
    # val = process_split(model, DataLoader(scaled["Val"], batch_size=1), threshold_agg=train["threshold_agg"])
    plot_error_distributions(i, m, train, val, pc, output_dir)

    rows = []
    for split, res in (("Val", val), ("Train", train)):
        sa = run_seismic_analysis(splits[split], datasets[split], eq, seismic_criteria, res["anomalies_agg"])
        rows.append({
            "window": i,
            "split": split,
            **w,
            "anomalies": len(res["anomalies_agg"]),
            "agg_value": sa["value"],
            "agg_error": sa["error"],
            "agg_total_eq": sa["total_eq"],
            "seismic_indices": sa["seismic_indices"],
            "threshold_agg": train["threshold_agg"],
        })
    return rows
