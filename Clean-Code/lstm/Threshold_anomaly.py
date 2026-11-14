"""Association of anomalous sequences with earthquakes (aggregate analysis)."""

import pandas as pd


class SeismicAnalysis:
    def __init__(self, dataset, earthquake_catalog, create_half_orbit_sequences, is_eq_fn):
        """
        Args:
            dataset: Split DataFrame (DatetimeIndex, features, lat/lon).
            earthquake_catalog: DataFrame with 'Time' (datetime64), 'lat', 'long', 'mag'.
            create_half_orbit_sequences: HalfOrbitPairDataset.create_half_orbit_sequences.
            is_eq_fn: SeismicCriteria.is_eq.
        """
        self.eq_catalog = earthquake_catalog
        self.is_eq_fn = is_eq_fn
        _, self.datetime_sequences, self.lat_long_sequences = create_half_orbit_sequences(dataset)

    def agg_analysis(self, anomalous_indices):
        """
        A sequence is seismic if any of its locations has an earthquake inside the
        spatial window within the time window after the sequence's last timestamp.

        Returns:
            seismic_indices: anomalous indices classified as seismic
            unique_eq_df: unique earthquakes matched by the anomalies
        """
        seismic_indices = []
        unique_earthquakes = set()

        for idx in anomalous_indices:
            anomaly_time = self.datetime_sequences[idx][-1]
            is_seismic_seq = False

            for location in self.lat_long_sequences[idx]:
                is_seismic, inside = self.is_eq_fn(anomaly_time, location, self.eq_catalog)
                unique_earthquakes.update(tuple(eq.items()) for eq in inside)
                is_seismic_seq |= bool(is_seismic)

            if is_seismic_seq:
                seismic_indices.append(idx)

        unique_eq_df = pd.DataFrame([dict(eq) for eq in unique_earthquakes])
        return seismic_indices, unique_eq_df
