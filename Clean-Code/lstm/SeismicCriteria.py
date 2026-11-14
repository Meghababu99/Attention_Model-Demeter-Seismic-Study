"""Spatio-temporal criterion linking an anomaly to earthquakes."""

import pandas as pd
from shapely.geometry import Point, Polygon


class SeismicCriteria:
    def __init__(self, spatial_width, time_window_hours):
        """
        Args:
            spatial_width: Side (degrees) of the square window centred on the anomaly.
            time_window_hours: Window length (hours) after the anomaly time.
        """
        self.spatial_width = spatial_width
        self.time_window = pd.Timedelta(hours=time_window_hours)

    def create_polygon(self, lat, lon):
        """Square polygon centred at (lat, lon); shapely uses (x, y) = (lon, lat)."""
        d = self.spatial_width / 2
        return Polygon([(lon - d, lat - d), (lon - d, lat + d), (lon + d, lat + d), (lon + d, lat - d)])

    def is_eq(self, timestamp, anomaly_location, earthquakes):
        """
        Earthquakes inside the square around `anomaly_location` within
        [timestamp, timestamp + time_window].

        `earthquakes` needs columns 'Time' (datetime64), 'lat', 'long', 'mag'.

        Returns:
            label: 1 if any earthquake is inside the window, else 0
            inside: list of dicts {'lat', 'long', 'Time', 'mag'}
        """
        start = pd.to_datetime(timestamp)
        in_time = earthquakes[(earthquakes["Time"] >= start) & (earthquakes["Time"] <= start + self.time_window)]

        lat, lon = anomaly_location
        polygon = self.create_polygon(lat, lon)

        inside = [
            eq[["lat", "long", "Time", "mag"]].to_dict()
            for _, eq in in_time.iterrows()
            if polygon.contains(Point(eq["long"], eq["lat"]))
        ]
        return int(bool(inside)), inside
