# LSTM-AE for the DEMETER Seismic Study

Unsupervised detection of ionospheric electric-field perturbations in DEMETER ICE satellite data with an
LSTM autoencoder, and statistical evaluation of their association with global earthquakes (M ≥ 5.0, 2005–2010).

The autoencoder learns what *normal* (non-seismic) DEMETER half-orbit sequences look like. Sequences it
reconstructs poorly are flagged as anomalies, and each anomaly is checked against the earthquake catalogue
with a fixed spatial and temporal window.

> **Scientific scope.** This is a statistical anomaly-detection and seismic-association study. It does not
> claim deterministic earthquake prediction or a direct physical cause. Detected events are candidate
> ionospheric anomalies whose association with seismicity is evaluated statistically.

**Paper:** Babu, M., Cristoforetti, M., Battiston, R., & Iuppa, R. (2026). *Data-Driven Deep Learning Model for
Detecting Ionospheric Electric Field Perturbations and Seismic Correlation.* Remote Sensing, 18(9), 1324.
https://doi.org/10.3390/rs18091324

---

## Contents

1. [Repository structure](#repository-structure)
2. [Quick start](#quick-start)
3. [Data](#data)
4. [Study conditions](#study-conditions)
5. [Training modes (A and In-A)](#training-modes-a-and-in-a)
6. [Clean code vs. the original thesis runs](#clean-code-vs-the-original-thesis-runs)
7. [Extension: attention, GRU and bidirectional encoders (STSM)](#extension-attention-gru-and-bidirectional-encoders-stsm)
8. [Result files](#result-files)
9. [Naming conventions](#naming-conventions)
10. [Author and data credit](#author-and-data-credit)

---

## Repository structure

```text
LSTM-AE-For-Demeter-Seismic-Study/
│
├── Clean-Code/                                    # all code (see Clean-Code/README.md)
│   ├── DEMETER_LSTM_AE-Training.ipynb             # rolling-window training + evaluation (baseline LSTM-AE)
│   ├── DEMETER_LSTM_AE-Evaluation-Pretrained.ipynb # evaluates saved models, compares with saved results
│   ├── DEMETER_STSM-Training.ipynb                # same pipeline with LSTM/GRU, Bi-, attention encoders
│   ├── lstm/                                      # Python package used by the notebooks
│   │   └── STSM/                                  # recurrent autoencoder with selectable cell/attention
│   └── requirements.txt
│
├── Data/
│   ├── EQ.csv                                     # earthquake catalogue
│   ├── Bg_window_data/                            # background-corrected data, one file per rolling window
│   ├── Label_data/                                # seismic (1) / non-seismic (0) label of each sequence
│   ├── RST-Label/                                 # labels with randomised times (random baseline)
│   ├── Descriptions/                              # DEMETER ICE and data-production documentation
│   ├── storm_data.pkl                             # Kp, Dst, AE geomagnetic indices
│   └── ...                                        # source and intermediate files of the study
│
├── Result Files/
│   ├── SW22_TW48/                                 # rolling-window results of the selected criterion
│   ├── Z-factor-files/                            # Z-factor and mean seismic-ratio summaries
│   └── STSM/                                      # attention vs. baseline comparison
│
└── README.md
```

Trained models are not part of the repository. To evaluate saved models, place them in
`Models-Trained/SW22_TW48/` (see [Naming conventions](#naming-conventions)).

---

## Quick start

**On a PC**

```bash
pip install -r Clean-Code/requirements.txt
```

Open a notebook in `Clean-Code/` and set `LOCAL_PROJECT` in its first cell to the repository folder
(default `D:\GIT\LSTM-AE-For-Demeter-Seismic-Study`).

**On Google Colab**

Open the notebook in Colab and run it. The first cell mounts Google Drive and uses
`MyDrive/LSTM-AE-For-Demeter-Seismic-Study`. If that folder does not exist, the cell clones this repository
(code and data) into it, so results are saved to Drive. Colab already has all required packages.

| Notebook | Use it to |
|---|---|
| `DEMETER_LSTM_AE-Training.ipynb` | Train and evaluate the baseline LSTM autoencoder on all rolling windows |
| `DEMETER_LSTM_AE-Evaluation-Pretrained.ipynb` | Re-evaluate saved models and check the results against `Result Files/` |
| `DEMETER_STSM-Training.ipynb` | Train and evaluate attention / GRU / bidirectional variants under the same conditions |

---

## Data

| File / folder | Content |
|---|---|
| `Bg_window_data/Background_data-window_{i}.pkl` | DEMETER ICE data of rolling window *i* (20 windows), indexed by time, with `lat`, `lon` and, for each of the 11 frequency bands *k*: `Q3_fb_k`, its 30-day background fit `fit_Q3_fb_k`, and the residual `Res_Q3_fb_k` |
| `Label_data/summary_df_{train,val}_30D-{SW}SW-tw{TW}_w{i}.csv` | One row per half-orbit pair: `index`, `label` (1 seismic, 0 non-seismic), matched earthquakes. One set per spatial/temporal criterion |
| `RST-Label/RDM_df_train_30D-{SW}SW-tw{TW}_w{i}.csv` | Same labels computed with randomised times, for the random (RST) baseline |
| `EQ.csv` | Earthquake catalogue: `Time`, `lat`, `long`, `depth`, `mag`; 12,442 events with M ≥ 5.0, 2005–2010 |
| `storm_data.pkl` | Geomagnetic indices `Kp`, `Dst`, `AE` (used for the storm-corrected results of the thesis) |
| `Down_Orbits-Q3-loc-Train-limitedgrids_RES.pkl` | Full DEMETER residual data set from which the windows were built |
| `Descriptions/` | DEMETER ICE instrument and data-production documentation |

The model input is the 11 residual features `Res_Q3_fb_1` … `Res_Q3_fb_11`. Raw `Q3_*` columns are dropped.

---

## Study conditions

These are the conditions used by the notebooks in `Clean-Code/`. Values are set in each notebook's
**Settings** cell.

### 1. Rolling windows

| Setting | Value |
|---|---|
| Training period | 12 months |
| Validation period | the 3 months right after the training period |
| Stride | 3 months (validation periods do not overlap; training periods do) |
| Range | from 2005-01-01: 16 windows up to 2010-01-02 (baseline), 20 windows up to 2011-01-02 (STSM) |
| Background | each window has its own 30-day background-corrected data file (`30dBG` in file names) |

### 2. Sequences (model input)

| Setting | Value |
|---|---|
| Half-orbit | data split wherever the time gap exceeds **1 hour** |
| Length | the first **17 samples** of each half-orbit (shorter half-orbits are dropped) |
| Pair | consecutive half-orbits (0–1, 2–3, …) whose start times are at most **2 hours** apart |
| Features | 11 residual bands `Res_Q3_fb_1…11` |

Each pair is one sequence of 2 × 17 = 34 samples. The autoencoder is applied to each half-orbit separately
(17 × 11), and the pair's error is the mean of the two.

### 3. Seismic criterion and labels

A sequence is **seismic** if at least one earthquake (M ≥ 5.0) lies:

- inside a **22° × 22°** square (spatial width `sw = 22`) centred on any of the sequence's 34 locations, and
- within **48 hours** (`tw = 48`) after the sequence's last timestamp.

The label files apply this criterion in advance; `label = 0` marks normal sequences. Label files for other
criteria are also in `Label_data/`: SW 5°, 10°, 20° (`30D-tw48`, `20SW_full`), rectangles 20° × 22° and
20° × 23°, and TW 6 h and 12 h with SW 22°.

### 4. Scaling

A `StandardScaler` is fitted **only on the normal (label-0) training pairs** of the window and applied to
all train and validation pairs.

### 5. Model

| Setting | Value |
|---|---|
| Encoder | 2-layer LSTM (input 11, hidden 8); the last hidden state of the top layer → linear layer → latent vector of size **2** |
| Decoder | latent vector repeated over the 17 time steps → 2-layer LSTM (hidden 8) → linear layer → 11 features |
| Loss | mean squared error, averaged over the two half-orbits of a pair |

### 6. Training

| Setting | Value |
|---|---|
| Training data | normal (label-0) training pairs only |
| Early stopping | on the normal (label-0) validation pairs; patience 150 epochs |
| Optimiser | Adam, learning rate 0.001, batch size 8, shuffled |
| Saved model | the model at the epoch where training stopped |
| Warm start | mode A: window *i > 0* starts from the model of window *i − 1* (see below) |
| Seed | 201894 (Python, NumPy, PyTorch) |

### 7. Anomaly detection

| Setting | Value |
|---|---|
| Score | aggregate reconstruction error (MSE) of each pair, computed one pair at a time |
| Data | **all** train and validation pairs (seismic and non-seismic) |
| Threshold | **98th percentile** of the training errors of the window; the same threshold is applied to validation |
| Anomaly | error ≥ threshold |

### 8. Seismic association

For each split (train, validation) of each window, with *n* anomalies of which *k* are seismic:

| Quantity | Definition |
|---|---|
| Seismic ratio *p* | *k / n*; when *k = n* it is (*k* − 1) / *n*, and when *k* = 0 it is 1 / *n* |
| Ratio (%) `agg_value` | 100 · *p* |
| Error (%) `agg_error` | 100 · √(*p*(1 − *p*) / *n*) (binomial) |
| Matched earthquakes `agg_total_eq` | number of distinct earthquakes matched by the anomalies |

These ratios are compared with the random baseline (`RST-Label/`) and summarised as Z-factors in
`Result Files/Z-factor-files/`.

---

## Training modes (A and In-A)

| Mode | Window 0 | Window *i > 0* | Purpose |
|---|---|---|---|
| **A** (weight updating) | random initialisation | starts from the model of window *i − 1* | keeps learning continuous across windows |
| **In-A** (re-initialising) | random initialisation | random initialisation | tests that the association does not depend on carried-over weights |

Set `WARM_START = True` (A) or `False` (In-A) in the notebook. The mode appears in every file name
(`…30dBG_A…` or `…30dBG_In-A…`).

---

## Clean code vs. the original thesis runs

The code in `Clean-Code/` is a cleaned version of the thesis code: unused functions and repeated code were
removed, without changing the method. Running `DEMETER_LSTM_AE-Evaluation-Pretrained.ipynb` with the saved
thesis models (`tw48-30dBG_A`, 16 windows) reproduces `Result Files/SW22_TW48/Retrain_result_Hp_B-tw48-30dBG_A.csv`
exactly (32 of 32 rows: anomalies, ratios, errors, matched earthquakes and seismic indices).

Two conditions differ from the thesis runs **when training new models**:

| | Thesis runs | Clean code |
|---|---|---|
| Early-stopping validation set | all validation pairs | normal (label-0) validation pairs only |
| Storm correction | an extra result set (`*_sc` columns) excluded sequences overlapping geomagnetic storms (Dst < −50 nT, Kp > 3 or AE > 500 nT) | not applied |

---

## Extension: attention, GRU and bidirectional encoders (STSM)

`DEMETER_STSM-Training.ipynb` runs the same pipeline with other encoders. Only the model changes, plus
**20 windows** and a **cosine-annealed learning rate** (`CosineAnnealingLR`, T_max = 50). The other
conditions are the same as above.

| Option | Choices |
|---|---|
| Recurrent cell | `LSTM`, `GRU` |
| Direction | unidirectional, bidirectional (`Bi…`) |
| Encoder summary | last hidden state; **TA**: temporal attention (softmax-weighted sum over the 17 time steps); **FTA**: feature attention over the 11 bands, then temporal attention |

Each model writes to `Clean-Code/STSM-Results/<model>/<run label>/`, e.g. `BiGRU-FTA/tw48-30dBG_A-BiGRU-FTA/`.
Details are in [`Clean-Code/README.md`](Clean-Code/README.md).

---

## Result files

| Folder | Content |
|---|---|
| `Result Files/SW22_TW48/Retrain_result_Hp_B-tw48-30dBG_{A…E, In-A…In-E}.csv` | Rolling-window results of the thesis models (SW 22°, TW 48 h): one Val and one Train row per window; `*_sc` columns are storm-corrected |
| `Result Files/SW22_TW48/Blind/` | Blind-test evaluation outputs (`Accuracy_*.csv`) |
| `Result Files/Z-factor-files/` | Z-factor and mean seismic-ratio summaries for each model and each SW–TW criterion |
| `Result Files/STSM/` | Temporal-attention results and the attention vs. baseline comparison |

---

## Naming conventions

```text
best_model_Hp_tw{TW}-30dBG_{MODE}_w{WINDOW}.pth     e.g. best_model_Hp_tw48-30dBG_A_w0.pth
result_Hp_tw{TW}-30dBG_{MODE}.csv                   e.g. result_Hp_tw48-30dBG_In-A.csv
```

| Part | Meaning |
|---|---|
| `tw48` | 48-hour temporal window |
| `30dBG` | 30-day background-corrected data |
| `A`, `In-A` | run name (`A`–`E` in the thesis results); the `In-` prefix marks re-initialising mode, no prefix weight-updating mode |
| `w0`, `w1`, … | rolling-window index |
| `-LSTM-TA`, `-BiGRU-FTA`, … | STSM runs: model name added to the run label |

---

## Author and data credit

**Megha Babu**. Data Science for Industry and Physics, Fondazione Bruno Kessler, and Department of Physics,
University of Trento, Italy.

DEMETER data are provided by CNES/CDPP and must be cited as required by the source:
https://cdpp-archive.cnes.fr/
