import pandas as pd
from pathlib import Path
from .events import EventMatcher, _normalize_active_region


class DatasetBuilder:
    """
    Maps image timestamps to flare labels using a configurable strategy.

    Parameters
    ----------
    prediction_window : int
        Hours after each image timestamp (or, for sequences, after the
        final image in the sequence) to search for flares.
    strategy : BinaryThresholdStrategy | MaxFlareStrategy
        Labeling strategy that converts a list of FlareEvents to a label.
    sequence_length : int
        Number of consecutive images per sample. 1 (default) preserves the
        original single-image behavior and output schema exactly.
    stride : int
        Step size between the start of consecutive sequences. Only used
        when sequence_length > 1.
    cadence_minutes : int | None
        Expected number of minutes between consecutive images inside a
        sequence. When set, a candidate sequence is skipped (not raised on)
        if any adjacent pair inside it doesn't differ by exactly this many
        minutes. Only used when sequence_length > 1.
    target : str
        "full_disk" (default) matches any flare in the window, preserving
        original behavior. "active_region" matches only flares from the
        specific active region each image (or, for sequences, each image in
        the sequence) is assigned to (requires an active_region column in
        the image index).
    """

    VALID_TARGETS = {"full_disk", "active_region"}

    def __init__(
        self,
        prediction_window,
        strategy,
        sequence_length=1,
        stride=1,
        cadence_minutes=None,
        target="full_disk",
    ):
        if prediction_window <= 0:
            raise ValueError(f"prediction_window must be > 0, got {prediction_window}")
        if sequence_length < 1:
            raise ValueError(f"sequence_length must be >= 1, got {sequence_length}")
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        if cadence_minutes is not None and cadence_minutes <= 0:
            raise ValueError(f"cadence_minutes must be > 0, got {cadence_minutes}")
        if target not in self.VALID_TARGETS:
            raise ValueError(f"target must be one of {sorted(self.VALID_TARGETS)}, got {target!r}")

        self.prediction_window = prediction_window
        self.strategy = strategy
        self.sequence_length = sequence_length
        self.stride = stride
        self.cadence_minutes = cadence_minutes
        self.target = target

    def build(self, image_index_path: str | Path, event_catalog_path: str | Path) -> pd.DataFrame:
        """
        Build a labeled dataset.

        For sequence_length == 1 (default), returns a DataFrame with columns
        (timestamp, label) -- identical to v0.1.0 behavior. An image index
        with a header row but zero data rows is valid and produces an empty
        result with the same columns.

        For sequence_length > 1, returns one row per sequence with columns
        (sequence_start, sequence_end, timestamps, n_images, label) -- or,
        when target="active_region", (sequence_start, sequence_end,
        timestamps, n_images, active_region, label) -- plus one additional
        list-column per any other column present in the image index (e.g.
        an image path or ID column), preserving that column's values for
        each image in the sequence. Timestamps must be sorted ascending and
        contain no duplicates in this mode. Fewer rows than sequence_length
        produces an empty result with the same (sequence-mode) columns.

        A completely empty image index (no header) raises ValueError, as
        does an image index missing the required timestamp column.
        """
        image_index_path = Path(image_index_path)

        try:
            index_df = pd.read_csv(image_index_path)
        except pd.errors.EmptyDataError:
            raise ValueError(
                f"Image index at {image_index_path} is empty: no header row / columns found."
            ) from None

        if "timestamp" not in index_df.columns:
            raise ValueError(
                f"Image index at {image_index_path} is missing required column: timestamp"
            )

        if self.target == "active_region" and "active_region" not in index_df.columns:
            raise ValueError(
                f"Image index at {image_index_path} is missing required column: "
                "active_region (required when target='active_region')"
            )

        index_df = index_df.copy()
        index_df["timestamp"] = pd.to_datetime(index_df["timestamp"])

        matcher = EventMatcher(event_catalog_path)

        if self.sequence_length == 1:
            if self.target == "active_region":
                return self._build_single_image_active_region(index_df, matcher)
            return self._build_single_image(index_df, matcher)

        self._validate_sequence_timestamps(index_df["timestamp"], image_index_path)
        if self.target == "active_region":
            return self._build_sequences_active_region(index_df, matcher, image_index_path)
        return self._build_sequences(index_df, matcher)

    def _build_single_image(self, index_df: pd.DataFrame, matcher: EventMatcher) -> pd.DataFrame:
        records = []
        for ts in index_df["timestamp"]:
            flares = matcher.query(ts, self.prediction_window)
            label = self.strategy.label(flares)
            records.append({"timestamp": ts, "label": label})

        if not records:
            return pd.DataFrame(columns=["timestamp", "label"])
        return pd.DataFrame(records)

    def _build_single_image_active_region(self, index_df: pd.DataFrame, matcher: EventMatcher) -> pd.DataFrame:
        records = []
        for ts, raw_active_region in zip(index_df["timestamp"], index_df["active_region"]):
            active_region = _normalize_active_region(raw_active_region)
            if active_region is None:
                # Missing image-side active_region must not silently match
                # anything -- skip the query entirely rather than falling
                # back to full-disk matching.
                flares = []
            else:
                flares = matcher.query(ts, self.prediction_window, active_region=active_region)
            label = self.strategy.label(flares)
            records.append({"timestamp": ts, "label": label})

        if not records:
            return pd.DataFrame(columns=["timestamp", "label"])
        return pd.DataFrame(records)

    def _validate_sequence_timestamps(self, timestamps: pd.Series, path: Path) -> None:
        if not timestamps.is_monotonic_increasing:
            raise ValueError(
                f"Image index at {path} must have timestamps sorted in ascending "
                "order to build sequences."
            )

        duplicated = timestamps[timestamps.duplicated()]
        if not duplicated.empty:
            dupe_list = ", ".join(str(t) for t in duplicated.unique())
            raise ValueError(
                f"Image index at {path} contains duplicate timestamp(s): {dupe_list}"
            )

    def _sequence_violates_cadence(self, window_timestamps: list) -> bool:
        if self.cadence_minutes is None:
            return False
        expected_gap = pd.Timedelta(minutes=self.cadence_minutes)
        for earlier, later in zip(window_timestamps, window_timestamps[1:]):
            if (later - earlier) != expected_gap:
                return True
        return False

    def _build_sequences(self, index_df: pd.DataFrame, matcher: EventMatcher) -> pd.DataFrame:
        timestamps = index_df["timestamp"].tolist()
        extra_columns = [c for c in index_df.columns if c != "timestamp"]
        n = len(timestamps)
        length = self.sequence_length

        columns = ["sequence_start", "sequence_end", "timestamps", "n_images", "label"] + extra_columns

        records = []
        start = 0
        while start + length <= n:
            window = timestamps[start:start + length]

            if not self._sequence_violates_cadence(window):
                sequence_end = window[-1]
                flares = matcher.query(sequence_end, self.prediction_window)
                label = self.strategy.label(flares)

                record = {
                    "sequence_start": window[0],
                    "sequence_end": sequence_end,
                    "timestamps": list(window),
                    "n_images": len(window),
                    "label": label,
                }
                for col in extra_columns:
                    record[col] = index_df[col].iloc[start:start + length].tolist()
                records.append(record)

            start += self.stride

        if not records:
            return pd.DataFrame(columns=columns)
        return pd.DataFrame(records)[columns]

    def _sequence_active_region(self, window_active_regions: list, path: Path, start: int, length: int) -> str | None:
        normalized = [_normalize_active_region(v) for v in window_active_regions]
        distinct_non_missing = {v for v in normalized if v is not None}

        if len(distinct_non_missing) > 1:
            raise ValueError(
                f"Image index at {path} contains a sequence (rows {start}:{start + length}) "
                f"with mixed active_region values: {sorted(distinct_non_missing)}. "
                "A sequence must refer to a single active region."
            )

        if None in normalized:
            # A missing active_region on any row must not silently match
            # anything, even if every other row agrees on a region --
            # consistent with the single-image missing-AR behavior.
            return None

        return next(iter(distinct_non_missing))

    def _build_sequences_active_region(self, index_df: pd.DataFrame, matcher: EventMatcher, path: Path) -> pd.DataFrame:
        timestamps = index_df["timestamp"].tolist()
        active_regions = index_df["active_region"].tolist()
        extra_columns = [c for c in index_df.columns if c not in ("timestamp", "active_region")]
        n = len(timestamps)
        length = self.sequence_length

        columns = ["sequence_start", "sequence_end", "timestamps", "n_images", "active_region", "label"] + extra_columns

        records = []
        start = 0
        while start + length <= n:
            window = timestamps[start:start + length]

            if not self._sequence_violates_cadence(window):
                active_region = self._sequence_active_region(
                    active_regions[start:start + length], path, start, length
                )
                sequence_end = window[-1]
                if active_region is None:
                    flares = []
                else:
                    flares = matcher.query(sequence_end, self.prediction_window, active_region=active_region)
                label = self.strategy.label(flares)

                record = {
                    "sequence_start": window[0],
                    "sequence_end": sequence_end,
                    "timestamps": list(window),
                    "n_images": len(window),
                    "active_region": active_region,
                    "label": label,
                }
                for col in extra_columns:
                    record[col] = index_df[col].iloc[start:start + length].tolist()
                records.append(record)

            start += self.stride

        if not records:
            return pd.DataFrame(columns=columns)
        return pd.DataFrame(records)[columns]
