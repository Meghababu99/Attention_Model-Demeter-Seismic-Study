"""Training loop for the LSTM autoencoder on half-orbit pairs."""

import os
from copy import deepcopy

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim


def _run_epoch(model, loader, criterion, device, optimizer=None):
    """One pass over `loader`; trains when an optimizer is given. Returns the mean pair loss."""
    training = optimizer is not None
    model.train(training)
    total = 0.0
    with torch.set_grad_enabled(training):
        for x1, x2 in loader:
            x1, x2 = x1.to(device), x2.to(device)
            loss = (criterion(model(x1), x1) + criterion(model(x2), x2)) / 2
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item()
    return total / len(loader)


def train_lstm_ae_mode(
    model,
    train_data_loader,
    val_data_loader,
    num_epochs=450,
    lr=0.001,
    patience=100,
    best_model_path="best_model.pth",
    early_stopping=True,
    save_all_epochs=False,
    all_epochs_dir=None,
    plot_loss=False,
    loss_plot_path=None,
    device=None,
    restore_best_epoch=False,
    cosine_t_max=None,
):
    """
    Train with Adam + MSE. Early stopping counts epochs without validation-loss
    improvement. The model saved to `best_model_path` (and returned) is:
        restore_best_epoch=False: the model at the epoch where training stopped
        restore_best_epoch=True:  the model of the epoch with the lowest validation loss
    cosine_t_max: if given, the learning rate follows CosineAnnealingLR(T_max=cosine_t_max), stepped every epoch.

    Returns:
        model, train_losses, val_losses
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_t_max) if cosine_t_max else None

    train_losses, val_losses = [], []
    best_val_loss, epochs_no_improve, best_state = float("inf"), 0, None

    for epoch in range(1, num_epochs + 1):
        train_losses.append(_run_epoch(model, train_data_loader, criterion, device, optimizer))
        val_losses.append(_run_epoch(model, val_data_loader, criterion, device))
        if scheduler:
            scheduler.step()
        print(f"Epoch [{epoch}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

        if save_all_epochs and all_epochs_dir:
            os.makedirs(all_epochs_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(all_epochs_dir, f"model_epoch_{epoch}.pth"))

        if val_losses[-1] < best_val_loss:
            best_val_loss, epochs_no_improve = val_losses[-1], 0
            if restore_best_epoch:
                best_state = deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1
            if early_stopping and epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), best_model_path)

    if plot_loss and loss_plot_path:
        _plot_losses(model, train_data_loader, lr, train_losses, val_losses, loss_plot_path)

    return model, train_losses, val_losses


def _plot_losses(model, train_data_loader, lr, train_losses, val_losses, loss_plot_path):
    """Train/validation loss curves with the model settings in a text box."""
    settings = {
        **model.hparams,
        "Batch Size": train_data_loader.batch_size,
        "Learning Rate": f"{lr:.1e}",
        "Sequence Length": 2 * train_data_loader.dataset[0][0].shape[0],
    }
    details = "\n".join(f"{k}: {v}" for k, v in settings.items())

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs. Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.text(0.5, 0.95, details, transform=plt.gca().transAxes, fontsize=9, va="top", ha="center",
             bbox=dict(facecolor="white", edgecolor="gray", boxstyle="round,pad=0.3", alpha=0.6))

    os.makedirs(os.path.dirname(loss_plot_path), exist_ok=True)
    plt.savefig(loss_plot_path, dpi=600, bbox_inches="tight")
    plt.show()
    plt.close()
