"""
Analysis of the rolling-window result files: random baseline (RST), Z-factor,
per-window plots, and comparison of a model with the baseline LSTM-AE.

Result files have one Val and one Train row per window with the columns
`split`, `anomalies`, `agg_value`, `agg_error`, `agg_total_eq`, `seismic_indices`
(and `window` in the Clean-Code results).
"""

import ast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

plt.rcParams["font.family"] = "DejaVu Serif"

RUNS = list("ABCDE") + [f"In-{c}" for c in "ABCDE"]


# ------------------------------------------------------------------ loading
def rst_levels(rst_path, tw):
    """Median and 0.3 / 99.7 percentiles (%) of the random seismic ratio (RST) for time window tw."""
    rst = pd.read_pickle(rst_path)[tw]
    return {
        "q2": np.percentile(rst, 50) * 100,
        "low": np.percentile(rst, 0.3) * 100,
        "high": np.percentile(rst, 99.7) * 100,
    }


def read_results(path, first_window=0):
    """
    Result CSV with lower-case `split` and a `window` column. If the file has no `window` column
    (rows alternate Val/Train per window), windows are numbered from `first_window`.
    """
    df = pd.read_csv(path)
    df["split"] = df["split"].str.strip().str.lower()
    if "window" not in df:
        df["window"] = np.arange(len(df)) // 2 + first_window
    return df


def split(df, name):
    """Rows of one split ('train' or 'val'), sorted by window."""
    return df[df["split"] == name].sort_values("window").reset_index(drop=True)


# ------------------------------------------------------------------ Z-factor
def z_windows(run):
    """
    Windows used for the Z-factor: the 2005-2009 validation periods (windows 0-15).
    Window 0 is left out for weight-updating runs (A-E), whose later windows start from it.
    """
    return range(0, 16) if run.startswith("In-") else range(1, 16)


def z_factor(files, rst, tw, windows=None):
    """
    Z-factor and average seismic ratio of each run.

    files: {run: result DataFrame}
    windows: windows to use for every run; None uses z_windows(run) (2005-2009).
    For each split, over these windows:
        z_i       = (ratio_i - RST median) / error_i
        z         = mean(z_i), with its 32-68 % range
        avg       = mean(ratio_i),   err = sqrt(sum(error_i^2)) / n
    """
    rows = []
    for run, df in files.items():
        row = {"Naming": run, "RST_Median": rst["q2"], "RST_99.7": rst["high"]}
        for name, tag in (("val", "va"), ("train", "tr")):
            d = split(df, name)
            d = d[d["window"].isin(z_windows(run) if windows is None else windows)]
            z_i = (d["agg_value"] - rst["q2"]) / d["agg_error"]
            row[f"z_{tag}_all"] = z_i.mean()
            row[f"z_{tag}_err32_all"], row[f"z_{tag}_err68_all"] = np.percentile(z_i, [32, 68])
            row[f"avg_{name}_all"] = d["agg_value"].mean()
            row[f"err_{name}_all"] = np.sqrt((d["agg_error"] ** 2).sum()) / len(d)
        rows.append(row)
    return pd.DataFrame(rows).assign(mode="all", tw=tw)


# ------------------------------------------------------------------ comparison with the baseline
def seismic_count(indices):
    """Number of distinct seismic sequences in a stored `seismic_indices` list."""
    return len(set(ast.literal_eval(indices))) if pd.notna(indices) else np.nan


def paired_validation(base, model):
    """Validation rows of the baseline and the model, paired by window (windows of the baseline only)."""
    cols = ["window", "anomalies", "agg_value", "agg_error", "agg_total_eq", "seismic_indices"]
    b, m = split(base, "val")[cols], split(model, "val")[cols]
    paired = b.merge(m, on="window", suffixes=("_base", "_model"), validate="one_to_one")
    if len(paired) != len(b):
        raise ValueError("The model results do not contain all validation windows of the baseline.")
    for tag in ("base", "model"):
        paired[f"seismic_count_{tag}"] = paired[f"seismic_indices_{tag}"].map(seismic_count)
    return paired


def compare_with_baseline(pairs):
    """
    Per-window and per-run comparison of the validation results.

    pairs: {run: paired_validation(...)}
    Returns (per_window, summary).
    """
    per_window = pd.concat(
        [p.assign(Run=run) for run, p in pairs.items()], ignore_index=True
    )
    per_window["Anomalies_reduced"] = per_window["anomalies_base"] - per_window["anomalies_model"]
    per_window["Ratio_gain_pp"] = per_window["agg_value_model"] - per_window["agg_value_base"]
    per_window["Seismic_gain"] = per_window["seismic_count_model"] - per_window["seismic_count_base"]
    per_window["Fewer_anomalies_higher_ratio"] = (
        per_window["Anomalies_reduced"].gt(0) & per_window["Ratio_gain_pp"].gt(0)
    )

    g = per_window.groupby("Run", sort=False)
    selected = per_window[per_window["Fewer_anomalies_higher_ratio"]].groupby("Run", sort=False)
    summary = pd.DataFrame({
        "Windows": g.size(),
        "Anomalies_baseline": g["anomalies_base"].sum(),
        "Anomalies_model": g["anomalies_model"].sum(),
        "Mean_ratio_baseline": g["agg_value_base"].mean(),
        "Mean_ratio_model": g["agg_value_model"].mean(),
        "Mean_ratio_gain_pp": g["Ratio_gain_pp"].mean(),
        "Median_ratio_gain_pp": g["Ratio_gain_pp"].median(),
        "Earthquakes_baseline": g["agg_total_eq_base"].sum(),
        "Earthquakes_model": g["agg_total_eq_model"].sum(),
        "Seismic_sequences_baseline": g["seismic_count_base"].sum(),
        "Seismic_sequences_model": g["seismic_count_model"].sum(),
        "Windows_more_seismic_sequences": g["Seismic_gain"].apply(lambda s: s.gt(0).sum()),
        "Windows_fewer_anomalies_higher_ratio": g["Fewer_anomalies_higher_ratio"].sum(),
        "Mean_anomalies_reduced_in_those": selected["Anomalies_reduced"].mean(),
        "Mean_ratio_gain_pp_in_those": selected["Ratio_gain_pp"].mean(),
    })
    summary["Earthquakes_difference"] = summary["Earthquakes_model"] - summary["Earthquakes_baseline"]
    summary["Fewer_anomalies_higher_ratio_percent"] = (
        100 * summary["Windows_fewer_anomalies_higher_ratio"] / summary["Windows"]
    )
    return per_window, summary.reset_index(names="Run")


def compare_z(z_base, z_model):
    """Average ratios and Z-factors of the baseline and the model, with their differences."""
    cols = {"avg_train_all": "train_ratio", "avg_val_all": "val_ratio", "z_tr_all": "train_z", "z_va_all": "val_z"}
    b = z_base.set_index("Naming")[list(cols)].rename(columns=lambda c: cols[c] + "_baseline")
    m = z_model.set_index("Naming")[list(cols)].rename(columns=lambda c: cols[c] + "_model")
    out = b.join(m, how="inner")
    for c in cols.values():
        out[f"delta_{c}"] = out[f"{c}_model"] - out[f"{c}_baseline"]
    return out.reset_index(names="Run")


# ------------------------------------------------------------------ plots
def _save(fig, path):
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_ratio_per_window(df, rst, title, path=None):
    """Seismic ratio of each window (train and validation) with the RST median and 0.3-99.7 % band."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, label in (("train", "Train"), ("val", "Validation")):
        d = split(df, name)
        ax.errorbar(d["window"], d["agg_value"], yerr=d["agg_error"], fmt="o", capsize=3, label=label)
    ax.axhline(rst["q2"], color="red", linewidth=1.5, label="Q2 RST")
    ax.axhline(rst["low"], color="red", linestyle="--", linewidth=1)
    ax.axhline(rst["high"], color="red", linestyle="--", linewidth=1, label="0.3–99.7% RST")
    windows = split(df, "train")["window"]
    ax.set_xticks(windows, [f"w{w}" for w in windows], rotation=50)
    ax.set_xlabel("Window", fontsize=18)
    ax.set_ylabel("Seismic Anomaly ratio (%)", fontsize=16)
    ax.set_title(title, fontsize=18)
    ax.set_ylim(20, 130)
    ax.legend(fontsize=12, loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, path)


def plot_anomaly_bars(df, title, path=None):
    """Anomalies per window, split into seismic (hatched) and non-seismic, for train and validation."""
    fig, ax = plt.subplots(figsize=(12, 5))
    width = 0.35
    for name, color, shift in (("train", "tab:blue", -width / 2), ("val", "#FF8C00", width / 2)):
        d = split(df, name)
        x = np.arange(len(d)) + shift
        seismic = d["anomalies"] * d["agg_value"] / 100
        ax.bar(x, seismic, width, color=color, edgecolor="k", hatch="//")
        ax.bar(x, d["anomalies"] - seismic, width, bottom=seismic, color=color, edgecolor="k")
    windows = split(df, "train")["window"]
    ax.set_xticks(np.arange(len(windows)), [f"w{w}" for w in windows])
    ax.set_xlabel("Window", fontsize=18)
    ax.set_ylabel("Anomaly Count", fontsize=18)
    ax.set_title(title, fontsize=18)
    ax.legend(handles=[
        Patch(facecolor="tab:blue", edgecolor="k", hatch="//", label="Train - Seismic"),
        Patch(facecolor="tab:blue", edgecolor="k", label="Train - Non-seismic"),
        Patch(facecolor="#FF8C00", edgecolor="k", hatch="//", label="Val - Seismic"),
        Patch(facecolor="#FF8C00", edgecolor="k", label="Val - Non-seismic"),
    ], ncol=2, frameon=False, loc="upper left", fontsize=11)
    _save(fig, path)


def plot_z_factor(z, title, n=15, path=None):
    """Validation Z-factor of each run with its 32-68 % range and the Z = 0, 2/sqrt(n), 3/sqrt(n) levels (n windows)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(z))
    ax.vlines(x, z["z_va_err32_all"], z["z_va_err68_all"], color="darkorange", linewidth=2, label="Model Z_f 32-68%")
    ax.scatter(x, z["z_va_all"], color="darkorange", s=35, zorder=3, label="Model Z_f")
    ax.axhline(0, color="red", linewidth=1.5, label="Z score = 0")
    ax.axhline(2 / np.sqrt(n), color="red", linestyle="--", linewidth=1, label="Z score = 2")
    ax.axhline(3 / np.sqrt(n), color="red", linestyle=":", linewidth=1, label="Z score = 3")
    ax.set_xticks(x, z["Naming"])
    ax.set_xlabel("Model", fontsize=16)
    ax.set_ylabel("Z-Factor", fontsize=16)
    ax.set_title(title, fontsize=17, fontweight="bold")
    ax.set_ylim(-2, 3)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(fontsize=10, frameon=False, loc="best")
    _save(fig, path)


def plot_average_ratio(z, rst, title, path=None):
    """Average seismic ratio (train and validation) of each run with the RST median and band."""
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(z))
    ax.errorbar(x - 0.175, z["avg_train_all"], yerr=z["err_train_all"], fmt="o", capsize=5, color="tab:blue", label="Training")
    ax.errorbar(x + 0.175, z["avg_val_all"], yerr=z["err_val_all"], fmt="o", capsize=5, color="tab:orange", label="Validation")
    ax.axhline(rst["q2"], color="tab:red", lw=1.2, label="Q2 RST")
    ax.axhline(rst["high"], color="tab:red", linestyle="--", lw=1.2, label="0.3-99.7% RST")
    ax.axhline(rst["low"], color="tab:red", linestyle="--", lw=1.2)
    ax.set_xticks(x, z["Naming"])
    ax.set_xlabel("Model", fontsize=16)
    ax.set_ylabel("Average Seismic Anomaly Ratio (%)", fontsize=15)
    ax.set_title(title, fontsize=17, weight="bold")
    ax.set_ylim(30, 100)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=11, loc="best")
    _save(fig, path)


def plot_validation_comparison(paired, model_label, title, path=None):
    """Validation anomalies (bars) and seismic ratios (points) per window: baseline vs model."""
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    x = np.arange(len(paired))
    for tag, label, color, shift in (("base", "Baseline LSTM-AE", "tab:blue", -0.17),
                                     ("model", model_label, "darkorange", 0.17)):
        axes[0].bar(x + shift, paired[f"anomalies_{tag}"], width=0.34, color=color, label=label)
        axes[1].errorbar(x + shift, paired[f"agg_value_{tag}"], yerr=paired[f"agg_error_{tag}"],
                         fmt="o", color=color, capsize=4)
    axes[0].set_ylabel("Detected anomalies")
    axes[1].set_ylabel("Seismic ratio (%)")
    axes[1].set_ylim(0, 105)
    axes[1].set_xticks(x, [f"w{w}" for w in paired["window"]])
    axes[1].set_xlabel("Validation window")
    axes[0].legend(frameon=False, ncol=2)
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
    fig.suptitle(title, fontsize=17)
    _save(fig, path)


def plot_fewer_anomalies_higher_ratio(per_window, summary, model_label, path=None):
    """Share of validation windows where the model gives fewer anomalies and a higher ratio than the baseline."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    bars = axes[0].bar(summary["Run"], summary["Fewer_anomalies_higher_ratio_percent"], color="teal")
    axes[0].bar_label(bars, labels=[f"{int(k)}/{int(n)}" for k, n in
                                    zip(summary["Windows_fewer_anomalies_higher_ratio"], summary["Windows"])], padding=4)
    axes[0].set_ylim(0, 110)
    axes[0].set_xlabel("Model")
    axes[0].set_ylabel("Matching validation windows (%)")
    axes[0].set_title(f"Fewer anomalies and higher seismic ratio ({model_label})")
    axes[0].tick_params(axis="x", rotation=45)

    selected = per_window[per_window["Fewer_anomalies_higher_ratio"]]
    for run, d in selected.groupby("Run", sort=False):
        axes[1].scatter(d["Anomalies_reduced"], d["Ratio_gain_pp"], label=run, s=55, alpha=0.8)
    axes[1].set_xlabel("Reduction in detected anomalies")
    axes[1].set_ylabel("Seismic ratio gain (percentage points)")
    axes[1].set_title("Changes in those windows")
    if selected.empty:
        axes[1].text(0.5, 0.5, "No window meets both conditions", transform=axes[1].transAxes, ha="center")
    else:
        axes[1].legend(title="Model", frameon=False)
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.set_axisbelow(True)
    _save(fig, path)
