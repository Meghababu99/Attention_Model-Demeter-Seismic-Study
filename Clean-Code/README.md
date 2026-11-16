# Clean-Code

Code of the DEMETER LSTM-AE study. The study conditions are described in the [main README](../README.md);
this file describes the notebooks, the `lstm` package and the outputs.

## Notebooks

All three notebooks share the same structure: **Setup** (PC or Colab) → **Settings** → rolling windows →
loop over windows → results. They differ only in how the model of each window is obtained.

| Notebook | Model of each window | Output folder |
|---|---|---|
| `DEMETER_LSTM_AE-Training.ipynb` | trained (baseline LSTM-AE) | `outputs/`, `Model-Retraining/` |
| `DEMETER_LSTM_AE-Evaluation-Pretrained.ipynb` | loaded from `Models-Trained/SW22_TW48/` | `outputs_pretrained/<run label>/` |
| `DEMETER_STSM-Training.ipynb` | trained, for each model in `MODELS` | `STSM-Results/<model>/<run label>/` |

`DEMETER_STSM-Results-Analysis.ipynb` analyses the results of one STSM model (`MODEL_NAME`) for all runs
(A–E, In-A–In-E): per-window ratio and anomaly plots, Z-factor and average ratio against the random baseline
(RST), and comparison with the baseline LSTM-AE results in `Result Files/SW22_TW48/`. It writes to
`Result Files/STSM-Results/<model>/Analysis_results/`.

`DEMETER_STSM-Evaluation-NormalThreshold.ipynb` re-evaluates saved best models without training, with the
anomaly threshold taken from the normal (label-0) training pairs. It evaluates the STSM model and the baseline
LSTM-AE (thesis models in `Models-Trained/SW22_TW48/`) and writes `<model>-NormalThr` results. Analyse them with
`MODEL_NAME = "LSTM-TA-NormalThr"` and `BASELINE_NAME = "LSTM-AE-NormalThr"` in the analysis notebook.

### Main settings

| Setting | Meaning |
|---|---|
| `tw`, `sw`, `pc` | temporal window (h), spatial width (°), anomaly percentile |
| `WINDOW_START`, `WINDOW_END` | range of the rolling windows |
| `input_size`, `hidden_size`, `num_layers`, `latent_dim` | autoencoder size |
| `batch_size`, `lr`, `num_epochs`, `patience` | training |
| `RESTORE_BEST_EPOCH` | `False`: save the model of the stopping epoch; `True`: of the lowest validation loss |
| `COSINE_T_MAX` | `None`: constant learning rate; a number: `CosineAnnealingLR(T_max)` |
| `WARM_START` | `True`: mode A; `False`: mode In-A |
| `RESUME` | skip windows already in the result CSV, to continue an interrupted run |

`num_epochs` is set to a short test value; set the number of epochs for the real run.

### STSM model choice

Each entry of `MODELS` is `(CELL, BIDIRECTIONAL, ATTENTION)`:

```python
MODELS = [
    ("LSTM", False, "temporal"),          # LSTM-TA
    ("GRU",  True,  "feature_temporal"),  # BiGRU-FTA
    ("LSTM", True,  None),                # BiLSTM (no attention)
]
```

| Argument | Choices |
|---|---|
| `CELL` | `"LSTM"`, `"GRU"` |
| `BIDIRECTIONAL` | `False`, `True` (encoder and decoder) |
| `ATTENTION` | `None` (last hidden state), `"temporal"` (TA), `"feature_temporal"` (FTA) |

With `("LSTM", False, None)` the model is identical to the baseline LSTM-AE, and saved baseline or
attention models from the thesis code can be loaded directly.

## `lstm` package

| Module | Contents |
|---|---|
| `dataset.py` | `HalfOrbitPairDataset`: half-orbit pairs, optional label-0 filter, `create_half_orbit_sequences` (timestamps and locations of each pair) |
| `scaling.py` | `scale_datasets`: fits a `StandardScaler` on the normal training pairs and applies it to other datasets |
| `models.py` | `LSTMAutoencoder` (baseline) |
| `training_modes.py` | `train_lstm_ae_mode`: Adam + MSE, early stopping, optional best-epoch restore and cosine learning rate |
| `anomaly_detection.py` | `AnomalyDetector`: aggregate reconstruction errors and threshold detection |
| `SeismicCriteria.py` | `SeismicCriteria.is_eq`: earthquakes inside the spatial square and time window |
| `Threshold_anomaly.py` | `SeismicAnalysis.agg_analysis`: seismic anomalies and matched earthquakes |
| `pipeline.py` | steps shared by the notebooks: `generate_windows`, `load_eq_catalog`, `prepare_window`, `train_window`, `process_split`, `run_seismic_analysis`, `plot_error_distributions`, `evaluate_window` |
| `analysis.py` | results analysis: `rst_levels`, `z_factor`, `paired_validation`, `compare_with_baseline`, `compare_z` and the plots |
| `STSM/models.py` | `RecurrentAutoencoder` (LSTM/GRU, uni/bidirectional, none/TA/FTA) and `model_name` |

## Outputs of one run

```text
<output folder>/
├── Model-Retraining/
│   ├── best_model_Hp_{m}_w{i}.pth          # model of window i
│   ├── check_Hp_{m}_w{i}/                  # model after every epoch
│   ├── loss_curve_model_Hp_{m}_w{i}.png
│   └── loss_history_Hp_{m}_w{i}.csv
└── outputs/
    ├── result_Hp_{m}.csv                   # one Val and one Train row per window (saved after each window)
    └── Aggregate_Errordistribution-pc98-HP_m{m}_w{i}.png
```

`{m}` is the run label, e.g. `tw48-30dBG_A-re` or `tw48-30dBG_In-A-BiGRU-FTA`.

Columns of `result_Hp_{m}.csv`: `window`, `split`, `train_start`, `train_end`, `val_start`, `val_end`,
`anomalies`, `agg_value`, `agg_error`, `agg_total_eq`, `seismic_indices`, `threshold_agg`.

The pretrained-evaluation notebook also writes `comparison_with_reference_{m}.csv` (True/False per row and
column against the saved result file). The STSM notebook writes `STSM-Results/summary_tw{tw}-30dBG_{RUN}.csv`
(mean over windows per model and split).

Generated folders are listed in `.gitignore` and are not pushed to GitHub.
