"""
Paired half-orbit dataset.

A DEMETER time series is split into half-orbits wherever the time gap exceeds
`gap_threshold_hours`. Each half-orbit is cut to its first `min_data_points`
samples, and adjacent half-orbits (0-1, 2-3, ...) are paired when their start
times are at most `max_pair_gap_hours` apart.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset


class HalfOrbitPairDataset(Dataset):
    def __init__(
        self,
        df,
        df_labels=None,
        min_data_points=17,
        use_label_0_only=False,
        feature_regex=r"^Res_Q3_",
        lat_col="lat",
        lon_col="lon",
        gap_threshold_hours=1.0,
        max_pair_gap_hours=2.0,
    ):
        """
        Args:
            df: DataFrame with a DatetimeIndex and feature columns matching `feature_regex`.
            df_labels: Optional DataFrame with columns ['index', 'label'] (pair index -> label).
            min_data_points: Samples kept per half-orbit.
            use_label_0_only: Keep only pairs whose label is 0 (requires df_labels).
        """
        self.min_data_points = int(min_data_points)
        self.lat_col, self.lon_col = lat_col, lon_col
        self.gap_threshold = pd.Timedelta(hours=gap_threshold_hours)
        self.max_pair_gap = pd.Timedelta(hours=max_pair_gap_hours)

        self.features = df.filter(regex=feature_regex).columns
        if len(self.features) == 0:
            raise ValueError(f"No feature columns match regex '{feature_regex}'.")

        self.all_pairs = self.extract_half_orbit_pairs(df)

        if use_label_0_only and df_labels is not None:
            self.label_0_pair_indices = set(df_labels.loc[df_labels["label"] == 0, "index"])
            self.pairs = self.filter_pairs()
        else:
            if use_label_0_only:
                print("[WARN] use_label_0_only=True but df_labels=None; keeping all pairs.")
            self.pairs = self.all_pairs

    def _paired_segments(self, df):
        """Return [(segment_1, segment_2), ...] as DataFrames (all columns, first min_data_points rows)."""
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("The DataFrame index must be a DatetimeIndex.")
        df = df.sort_index()

        gaps = df.index.to_series().diff() > self.gap_threshold
        bounds = [df.index[0]] + gaps[gaps].index.tolist() + [df.index[-1]]

        half_orbits = []  # (start_time, segment)
        for start, end in zip(bounds[:-1], bounds[1:]):
            segment = df[(df.index >= start) & (df.index < end)]  # left-closed, right-open
            if len(segment) >= self.min_data_points:
                half_orbits.append((start, segment.iloc[: self.min_data_points]))

        return [
            (seg1, seg2)
            for (t1, seg1), (t2, seg2) in zip(half_orbits[0:-1:2], half_orbits[1::2])
            if t2 - t1 <= self.max_pair_gap
        ]

    def extract_half_orbit_pairs(self, df):
        """Feature arrays of each pair: [(x1, x2), ...]."""
        return [
            (seg1[self.features].values, seg2[self.features].values)
            for seg1, seg2 in self._paired_segments(df)
        ]

    def filter_pairs(self):
        """Keep only pairs whose index is labelled 0."""
        return [pair for idx, pair in enumerate(self.all_pairs) if idx in self.label_0_pair_indices]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        x1, x2 = self.pairs[idx]
        return torch.tensor(x1, dtype=torch.float32), torch.tensor(x2, dtype=torch.float32)

    def create_half_orbit_sequences(self, df):
        """
        Combined (34-sample) sequences of every pair, in the same order as `all_pairs`.

        Returns:
            sequences: feature arrays
            datetime_sequences: timestamp arrays
            lat_long_sequences: [lat, lon] arrays
        """
        if self.lat_col not in df.columns or self.lon_col not in df.columns:
            raise ValueError(f"Expected columns '{self.lat_col}' and '{self.lon_col}' in df.")

        combined = [pd.concat([seg1, seg2]) for seg1, seg2 in self._paired_segments(df)]
        print(f"[INFO] Total valid combined sequences: {len(combined)}")

        sequences = [seq[self.features].values for seq in combined]
        datetime_sequences = [seq.index.values for seq in combined]
        lat_long_sequences = [seq[[self.lat_col, self.lon_col]].values for seq in combined]
        return sequences, datetime_sequences, lat_long_sequences
