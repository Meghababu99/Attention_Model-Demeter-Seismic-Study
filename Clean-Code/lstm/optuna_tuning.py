"""
Optional Optuna search for `batch_size` and `lr` of one model, using the base training code unchanged.

Each trial trains the first rolling windows exactly as the base notebook does (same seed, same
`train_lstm_ae_mode`, same early stopping, warm start between windows) and scores the trial by the mean
per-pair validation reconstruction error (batch_size=1, so batch sizes are compared on the same scale).
No earthquake information enters the objective.
"""

import contextlib
import io
import json
import random
import tempfile
from pathlib import Path

import numpy as np
import optuna
import torch
from torch.utils.data import DataLoader

from lstm import AnomalyDetector, train_lstm_ae_mode


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tune(build_model, window_data, out_dir, seed, n_trials, batch_sizes, lr_range,
         warm_start=True, val_set="Val", **train_kwargs):
    """
    Args:
        build_model: function returning a new (untrained) model.
        window_data: list of `scaled` dicts (from prepare_window) of the tuning windows.
        out_dir: folder for best_params.json and trials.csv. If best_params.json exists it is
                 returned without tuning (delete it to tune again).
        val_set: "Val" (all validation pairs, as early stopping) or "Val_normal" (label-0 pairs).
        train_kwargs: passed to train_lstm_ae_mode (num_epochs, patience, restore_best_epoch, cosine_t_max).

    Returns:
        {"batch_size": ..., "lr": ...}
    """
    out_dir = Path(out_dir)
    best_path = out_dir / "best_params.json"
    if best_path.exists():
        print(f"[Optuna] Loaded {best_path}")
        return json.loads(best_path.read_text())

    def objective(trial):
        bs = trial.suggest_categorical("batch_size", list(batch_sizes))
        lr = trial.suggest_float("lr", *lr_range, log=True)
        seed_everything(seed)
        prev_state, scores = None, []
        with tempfile.TemporaryDirectory() as tmp:
            for k, scaled in enumerate(window_data):
                model = build_model()
                if warm_start and prev_state is not None:
                    model.load_state_dict(prev_state)
                with contextlib.redirect_stdout(io.StringIO()):  # silence per-epoch prints
                    model, _, _ = train_lstm_ae_mode(
                        model,
                        DataLoader(scaled["Train_normal"], batch_size=bs, shuffle=True),
                        DataLoader(scaled["Val"], batch_size=bs, shuffle=False),
                        lr=lr, best_model_path=str(Path(tmp) / "model.pth"), **train_kwargs,
                    )
                prev_state = model.state_dict()
                errors = AnomalyDetector(model).compute_reconstruction_errors_agg(
                    DataLoader(scaled[val_set], batch_size=1))
                scores.append(errors.mean())
                trial.report(float(np.mean(scores)), k)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        return float(np.mean(scores))

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=seed),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5))
    study.optimize(objective, n_trials=n_trials)

    out_dir.mkdir(parents=True, exist_ok=True)
    study.trials_dataframe().to_csv(out_dir / "trials.csv", index=False)
    best_path.write_text(json.dumps(study.best_params, indent=2))
    print(f"[Optuna] Best {study.best_params}, val error {study.best_value:.5f}")
    return study.best_params
