import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path
from .events import EventMatcher, _normalize_active_region, VALID_EVENT_TIMES, VALID_INTERVAL_MODES

_ROW_BASED_PARAM_NAMES = ("sequence_length", "stride", "cadence_minutes")
_TIME_BASED_PARAM_NAMES = ("start", "cadence", "observation_window", "sliding_window")

# Cap on how many skipped-sequence-starts / missing-timestamps a
# SequenceBuildReport keeps verbatim, so a very long-running build doesn't
# blow up memory just to report on itself. Counts are always exact.
_MAX_SEQUENCE_REPORT_SAMPLES = 50


@dataclass
class SequenceBuildReport:
    """
    Validation/bookkeeping report for DatasetBuilder.build(return_report=True)
    in time-based sequence mode.

    inferred_source_cadence : the base cadence detected from the image
        index's own timestamps (see DatasetBuilder's "Source cadence
        inference" docs).
    requested_cadence, observation_window, sliding_window : the
        (parsed, pd.Timedelta) values this build used.
    effective_start : the actual starting timestamp used (either the
        explicit `start`, or the dataset's earliest timestamp).
    total_candidate_sequences : every candidate start considered, whether
        or not it was ultimately emitted.
    emitted_sequences / skipped_sequences : how many candidates were
        emitted vs. skipped because a required timestamp was missing.
    skipped_sequence_starts : up to the first `_MAX_SEQUENCE_REPORT_SAMPLES`
        candidate start timestamps that were skipped.
    missing_timestamps : up to the first `_MAX_SEQUENCE_REPORT_SAMPLES`
        specific timestamps that were missing and caused a skip.
    """

    inferred_source_cadence: pd.Timedelta
    requested_cadence: pd.Timedelta
    observation_window: pd.Timedelta
    sliding_window: pd.Timedelta
    effective_start: pd.Timestamp | None
    total_candidate_sequences: int
    emitted_sequences: int
    skipped_sequences: int
    skipped_sequence_starts: list = field(default_factory=list)
    missing_timestamps: list = field(default_factory=list)


def _parse_duration(value, name: str) -> pd.Timedelta:
    """Normalize a pandas-compatible duration value, requiring it be strictly positive."""
    try:
        duration = pd.Timedelta(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} must be a valid duration (e.g. '1h', '30min'), got {value!r}: {exc}") from None
    if pd.isna(duration):
        raise ValueError(f"{name} must be a valid duration (e.g. '1h', '30min'), got {value!r}")
    if duration <= pd.Timedelta(0):
        raise ValueError(f"{name} must be strictly positive, got {value!r} (parsed as {duration})")
    return duration


def _parse_timestamp_param(value, name: str) -> pd.Timestamp:
    """Normalize a pandas-compatible timestamp value."""
    try:
        timestamp = pd.Timestamp(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} must be a valid timestamp, got {value!r}: {exc}") from None
    if pd.isna(timestamp):
        raise ValueError(f"{name} must be a valid timestamp, got {value!r}")
    return timestamp


def _infer_source_cadence(timestamps: pd.Series) -> pd.Timedelta:
    """
    Infer an image index's base cadence as the mode (most common value) of
    its sorted-unique timestamps' positive consecutive differences.

    Using the mode rather than, say, the minimum or largest gap makes this
    robust to a dataset with occasional missing images (larger, rarer gaps
    don't affect the result as long as the "normal" spacing is still the
    most frequent one).
    """
    unique_sorted = pd.Series(timestamps.unique()).sort_values().reset_index(drop=True)
    if len(unique_sorted) < 2:
        raise ValueError(
            "Cannot infer source cadence: the image index has fewer than 2 unique "
            "timestamps, so no consecutive difference can be measured."
        )
    diffs = unique_sorted.diff().dropna()
    diffs = diffs[diffs > pd.Timedelta(0)]
    if diffs.empty:
        raise ValueError("Cannot infer source cadence: no positive consecutive timestamp differences found.")
    return diffs.mode().iloc[0]


def _validate_requested_cadence(requested: pd.Timedelta, inferred: pd.Timedelta) -> None:
    if requested < inferred:
        raise ValueError(
            f"cadence {requested} is smaller than the inferred source cadence {inferred}; "
            "cannot request a cadence finer than the image index actually provides."
        )
    if requested % inferred != pd.Timedelta(0):
        raise ValueError(
            f"cadence {requested} is not an exact multiple of the inferred source cadence {inferred}."
        )


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
    target : str
        "full_disk" (default) matches any flare in the window, preserving
        original behavior. "active_region" matches only flares from the
        specific active region each image (or, for sequences, each image in
        the sequence) is assigned to (requires an active_region column in
        the image index).
    event_time : str
        Which of a flare's timestamps EventMatcher checks against the
        prediction window: "peak" (default, preserves original behavior)
        or "start". See EventMatcher.query.
    interval_mode : str
        The prediction window's boundary convention: "left_closed"
        (default, preserves original behavior) for
        [t, t + prediction_window), or "right_closed" for
        (t, t + prediction_window]. See EventMatcher.query.

    Row-based sequence parameters (mutually exclusive with the time-based
    ones below -- see "Sequence mode selection")
    ----------------------------------------------------------------------
    sequence_length : int | None
        Number of consecutive images per sample. Omitted/None (default)
        resolves to 1, preserving the original single-image behavior and
        output schema exactly.
    stride : int | None
        Step size between the start of consecutive sequences. Omitted/None
        (default) resolves to 1. Only meaningful when sequence_length > 1.
    cadence_minutes : int | None
        Expected number of minutes between consecutive images inside a
        sequence. When set, a candidate sequence is skipped (not raised on)
        if any adjacent pair inside it doesn't differ by exactly this many
        minutes. Only used when sequence_length > 1.

    Time-based sequence parameters (mutually exclusive with the row-based
    ones above)
    ----------------------------------------------------------------------
    start : pandas-compatible timestamp | None
        The first candidate sequence's starting timestamp. Inclusive: an
        image must exist at exactly this timestamp, or build() raises
        ValueError -- it never silently moves to the next available image.
        If omitted, defaults to the image index's earliest timestamp.
    cadence : pandas-compatible duration | None
        The spacing between consecutive images *within* a sequence (e.g.
        "1h"). Required whenever any time-based parameter is used. Must be
        an exact multiple of the image index's own inferred cadence (see
        "Source cadence inference").
    observation_window : pandas-compatible duration | None
        The half-open span [sequence_start, sequence_start +
        observation_window) covered by one sequence. Required in
        time-based mode. Must be an exact multiple of `cadence`; the
        number of images per sequence is observation_window / cadence
        (e.g. cadence="1h", observation_window="6h" -> 6 images at
        t, t+1h, ..., t+5h -- note the *last* image is at
        t + observation_window - cadence, since the interval excludes its
        own right endpoint).
    sliding_window : pandas-compatible duration | None
        How far forward the next candidate sequence's start moves relative
        to the current one. Required in time-based mode. Must be an exact
        multiple of `cadence`.

    Sequence mode selection
    ----------------------------------------------------------------------
    Supplying any of start/cadence/observation_window/sliding_window
    activates time-based mode, in which cadence, observation_window, and
    sliding_window are all required (start remains optional). Explicitly
    mixing row-based parameters (sequence_length/stride/cadence_minutes)
    with time-based ones raises ValueError -- pick one mode.

    Source cadence inference
    ----------------------------------------------------------------------
    In time-based mode, build() infers the image index's own base cadence
    as the mode (most common value) of its sorted-unique timestamps'
    positive consecutive differences -- robust to occasional missing
    images/larger gaps, unlike using the timestamps' minimum or maximum
    gap. The requested `cadence` must be greater than or equal to this
    inferred cadence and an exact integer multiple of it (e.g. a 30-minute
    source image index accepts cadence="1h", producing a compatible
    downsampled sequence -- but rejects cadence="1min", since that would
    require images finer than the source actually provides).
    """

    VALID_TARGETS = {"full_disk", "active_region"}

    def __init__(
        self,
        prediction_window,
        strategy,
        sequence_length=None,
        stride=None,
        cadence_minutes=None,
        target="full_disk",
        event_time="peak",
        interval_mode="left_closed",
        start=None,
        cadence=None,
        observation_window=None,
        sliding_window=None,
    ):
        if prediction_window <= 0:
            raise ValueError(f"prediction_window must be > 0, got {prediction_window}")
        if target not in self.VALID_TARGETS:
            raise ValueError(f"target must be one of {sorted(self.VALID_TARGETS)}, got {target!r}")
        if event_time not in VALID_EVENT_TIMES:
            raise ValueError(f"event_time must be one of {sorted(VALID_EVENT_TIMES)}, got {event_time!r}")
        if interval_mode not in VALID_INTERVAL_MODES:
            raise ValueError(f"interval_mode must be one of {sorted(VALID_INTERVAL_MODES)}, got {interval_mode!r}")

        row_based_given = [
            name for name, value in zip(_ROW_BASED_PARAM_NAMES, (sequence_length, stride, cadence_minutes))
            if value is not None
        ]
        time_based_given = [
            name for name, value in zip(_TIME_BASED_PARAM_NAMES, (start, cadence, observation_window, sliding_window))
            if value is not None
        ]

        if row_based_given and time_based_given:
            raise ValueError(
                "Cannot mix row-based sequence parameters "
                f"({', '.join(row_based_given)}) with time-based sequence parameters "
                f"({', '.join(time_based_given)}). Choose either row-based mode "
                "(sequence_length/stride/cadence_minutes) or time-based mode "
                "(start/cadence/observation_window/sliding_window), not both."
            )

        self.prediction_window = prediction_window
        self.strategy = strategy
        self.target = target
        self.event_time = event_time
        self.interval_mode = interval_mode
        self.time_based = bool(time_based_given)

        if self.time_based:
            missing_required = [
                name for name, value in (
                    ("cadence", cadence),
                    ("observation_window", observation_window),
                    ("sliding_window", sliding_window),
                )
                if value is None
            ]
            if missing_required:
                raise ValueError(
                    f"Time-based sequence mode also requires: {', '.join(missing_required)} "
                    "(start is the only optional time-based parameter)."
                )

            self.cadence = _parse_duration(cadence, "cadence")
            self.observation_window = _parse_duration(observation_window, "observation_window")
            self.sliding_window = _parse_duration(sliding_window, "sliding_window")
            self.start = _parse_timestamp_param(start, "start") if start is not None else None

            if self.observation_window % self.cadence != pd.Timedelta(0):
                raise ValueError(
                    f"observation_window ({observation_window}) must be an exact multiple "
                    f"of cadence ({cadence})."
                )
            if self.sliding_window % self.cadence != pd.Timedelta(0):
                raise ValueError(
                    f"sliding_window ({sliding_window}) must be an exact multiple of cadence ({cadence})."
                )

            self.number_of_images = round(self.observation_window / self.cadence)
            self.slide_steps = round(self.sliding_window / self.cadence)

            self.sequence_length = None
            self.stride = None
            self.cadence_minutes = None
        else:
            resolved_sequence_length = 1 if sequence_length is None else sequence_length
            resolved_stride = 1 if stride is None else stride
            if resolved_sequence_length < 1:
                raise ValueError(f"sequence_length must be >= 1, got {resolved_sequence_length}")
            if resolved_stride < 1:
                raise ValueError(f"stride must be >= 1, got {resolved_stride}")
            if cadence_minutes is not None and cadence_minutes <= 0:
                raise ValueError(f"cadence_minutes must be > 0, got {cadence_minutes}")

            self.sequence_length = resolved_sequence_length
            self.stride = resolved_stride
            self.cadence_minutes = cadence_minutes

    def build(
        self, image_index_path: str | Path, event_catalog_path: str | Path, return_report: bool = False,
    ) -> pd.DataFrame:
        """
        Build a labeled dataset.

        For sequence_length == 1 (default, row-based mode), returns a
        DataFrame with columns (timestamp, label) -- identical to v0.1.0
        behavior. An image index with a header row but zero data rows is
        valid and produces an empty result with the same columns.

        For sequence_length > 1 (row-based mode), returns one row per
        sequence with columns (sequence_start, sequence_end, timestamps,
        n_images, label) -- or, when target="active_region",
        (sequence_start, sequence_end, timestamps, n_images, active_region,
        label) -- plus one additional list-column per any other column
        present in the image index (e.g. an image path or ID column),
        preserving that column's values for each image in the sequence.
        Timestamps must be sorted ascending and contain no duplicates in
        this mode. Fewer rows than sequence_length produces an empty
        result with the same (sequence-mode) columns.

        In time-based mode (see the class docstring), returns the same
        sequence-mode schema as row-based sequences, built instead from an
        exact timestamp grid (candidate_start + i * cadence) rather than
        positional row-slicing -- this is what allows compatible
        downsampling of a finer-cadence image index. A candidate whose
        required grid is missing any timestamp is skipped, not
        interpolated or substituted.

        return_report=True is only supported in time-based mode, where it
        returns (DataFrame, SequenceBuildReport) instead of just the
        DataFrame; passing it in row-based/single-image mode raises
        ValueError.

        A completely empty image index (no header) raises ValueError, as
        does an image index missing the required timestamp column.
        """
        if return_report and not self.time_based:
            raise ValueError(
                "return_report=True is only supported in time-based sequence mode "
                "(i.e. when start/cadence/observation_window/sliding_window were used)."
            )

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

        if self.time_based:
            if self.target == "active_region":
                result, report = self._build_time_based_sequences_active_region(index_df, matcher, image_index_path)
            else:
                result, report = self._build_time_based_sequences(index_df, matcher, image_index_path)
            return (result, report) if return_report else result

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
            flares = matcher.query(
                ts, self.prediction_window,
                event_time=self.event_time, interval_mode=self.interval_mode,
            )
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
                flares = matcher.query(
                    ts, self.prediction_window, active_region=active_region,
                    event_time=self.event_time, interval_mode=self.interval_mode,
                )
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
        self._validate_unique_timestamps(timestamps, path)

    def _validate_unique_timestamps(self, timestamps: pd.Series, path: Path) -> None:
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
                flares = matcher.query(
                    sequence_end, self.prediction_window,
                    event_time=self.event_time, interval_mode=self.interval_mode,
                )
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

    def _sequence_active_region(self, window_active_regions: list, path: Path, descriptor: str) -> str | None:
        normalized = [_normalize_active_region(v) for v in window_active_regions]
        distinct_non_missing = {v for v in normalized if v is not None}

        if len(distinct_non_missing) > 1:
            raise ValueError(
                f"Image index at {path} contains a sequence ({descriptor}) "
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
                    active_regions[start:start + length], path, f"rows {start}:{start + length}"
                )
                sequence_end = window[-1]
                if active_region is None:
                    flares = []
                else:
                    flares = matcher.query(
                        sequence_end, self.prediction_window, active_region=active_region,
                        event_time=self.event_time, interval_mode=self.interval_mode,
                    )
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

    # ── Time-based sequence mode ─────────────────────────────────────────

    def _time_based_candidates(self, index_df: pd.DataFrame, path: Path) -> tuple[list, SequenceBuildReport]:
        """
        Shared candidate-generation for time-based sequence mode, used by
        both the full-disk and active-region builders below.

        Returns (emitted, report) where `emitted` is a list of
        (window_timestamps, row_positions) for each candidate that had
        every required timestamp present -- row_positions are positions
        into index_df, suitable for .iloc, used to fetch extra columns
        (and, for active-region mode, the active_region column) for
        exactly those rows.
        """
        timestamps = index_df["timestamp"]
        self._validate_unique_timestamps(timestamps, path)

        inferred_cadence = _infer_source_cadence(timestamps)
        _validate_requested_cadence(self.cadence, inferred_cadence)

        ts_to_position = {ts: i for i, ts in enumerate(timestamps)}
        max_ts = timestamps.max()

        if self.start is not None:
            effective_start = self.start
            if effective_start not in ts_to_position:
                raise ValueError(
                    f"start={effective_start} has no exact matching image timestamp in "
                    f"the image index at {path}."
                )
        else:
            effective_start = timestamps.min()

        offsets = [i * self.cadence for i in range(self.number_of_images)]

        emitted = []
        total_candidates = 0
        skipped_starts = []
        missing_timestamps = []

        candidate_start = effective_start
        while candidate_start + offsets[-1] <= max_ts:
            total_candidates += 1
            required_timestamps = [candidate_start + offset for offset in offsets]

            positions = []
            missing = None
            for required_ts in required_timestamps:
                position = ts_to_position.get(required_ts)
                if position is None:
                    missing = required_ts
                    break
                positions.append(position)

            if missing is not None:
                if len(skipped_starts) < _MAX_SEQUENCE_REPORT_SAMPLES:
                    skipped_starts.append(candidate_start)
                if len(missing_timestamps) < _MAX_SEQUENCE_REPORT_SAMPLES:
                    missing_timestamps.append(missing)
            else:
                emitted.append((required_timestamps, positions))

            candidate_start = candidate_start + self.sliding_window

        report = SequenceBuildReport(
            inferred_source_cadence=inferred_cadence,
            requested_cadence=self.cadence,
            observation_window=self.observation_window,
            sliding_window=self.sliding_window,
            effective_start=effective_start,
            total_candidate_sequences=total_candidates,
            emitted_sequences=len(emitted),
            skipped_sequences=total_candidates - len(emitted),
            skipped_sequence_starts=skipped_starts,
            missing_timestamps=missing_timestamps,
        )
        return emitted, report

    def _build_time_based_sequences(
        self, index_df: pd.DataFrame, matcher: EventMatcher, path: Path,
    ) -> tuple[pd.DataFrame, SequenceBuildReport]:
        candidates, report = self._time_based_candidates(index_df, path)
        extra_columns = [c for c in index_df.columns if c != "timestamp"]
        columns = ["sequence_start", "sequence_end", "timestamps", "n_images", "label"] + extra_columns

        records = []
        for window_timestamps, positions in candidates:
            sequence_end = window_timestamps[-1]
            flares = matcher.query(
                sequence_end, self.prediction_window,
                event_time=self.event_time, interval_mode=self.interval_mode,
            )
            label = self.strategy.label(flares)

            record = {
                "sequence_start": window_timestamps[0],
                "sequence_end": sequence_end,
                "timestamps": list(window_timestamps),
                "n_images": len(window_timestamps),
                "label": label,
            }
            for col in extra_columns:
                record[col] = index_df[col].iloc[positions].tolist()
            records.append(record)

        if not records:
            return pd.DataFrame(columns=columns), report
        return pd.DataFrame(records)[columns], report

    def _build_time_based_sequences_active_region(
        self, index_df: pd.DataFrame, matcher: EventMatcher, path: Path,
    ) -> tuple[pd.DataFrame, SequenceBuildReport]:
        candidates, report = self._time_based_candidates(index_df, path)
        extra_columns = [c for c in index_df.columns if c not in ("timestamp", "active_region")]
        columns = ["sequence_start", "sequence_end", "timestamps", "n_images", "active_region", "label"] + extra_columns

        records = []
        for window_timestamps, positions in candidates:
            window_active_regions = index_df["active_region"].iloc[positions].tolist()
            active_region = self._sequence_active_region(
                window_active_regions, path, f"starting at {window_timestamps[0]}"
            )
            sequence_end = window_timestamps[-1]
            if active_region is None:
                flares = []
            else:
                flares = matcher.query(
                    sequence_end, self.prediction_window, active_region=active_region,
                    event_time=self.event_time, interval_mode=self.interval_mode,
                )
            label = self.strategy.label(flares)

            record = {
                "sequence_start": window_timestamps[0],
                "sequence_end": sequence_end,
                "timestamps": list(window_timestamps),
                "n_images": len(window_timestamps),
                "active_region": active_region,
                "label": label,
            }
            for col in extra_columns:
                record[col] = index_df[col].iloc[positions].tolist()
            records.append(record)

        if not records:
            return pd.DataFrame(columns=columns), report
        return pd.DataFrame(records)[columns], report
