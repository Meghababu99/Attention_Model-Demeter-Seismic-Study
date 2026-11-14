"""Standard scaling of half-orbit pair datasets."""

import torch
from sklearn.preprocessing import StandardScaler


def scale_datasets(train_dataset_normal, *datasets):
    """
    Fit one StandardScaler on every half-orbit of `train_dataset_normal` (label-0
    training pairs) and apply it to each dataset in `datasets`.

    Returns:
        Tuple of scaled datasets (lists of (x1, x2) tensors), in the order given.
    """
    all_half_orbits = torch.cat([x for pair in train_dataset_normal for x in pair]).numpy()
    scaler = StandardScaler().fit(all_half_orbits)

    def apply_scaling(dataset):
        return [
            tuple(torch.from_numpy(scaler.transform(x.numpy())) for x in pair)
            for pair in dataset
        ]

    return tuple(apply_scaling(dataset) for dataset in datasets)
