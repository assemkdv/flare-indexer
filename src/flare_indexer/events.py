import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass


# A simple container that holds the details of one flare event
# Think of it like a row from the NOAA catalog, but as a Python object
@dataclass
class FlareEvent:
    peak_time: pd.Timestamp          # when the flare was at its strongest
    goes_class: str                  # flare class letter + number, e.g. "M2.3"
    start_time: pd.Timestamp         # when the flare started
    active_region: str | None        # which region of the sun it came from (can be empty)
    end_time: pd.Timestamp | None = None   # when the flare ended (not all catalogs carry this)
    fl_lon: float | None = None            # heliographic longitude, when the source catalog has it
    fl_lat: float | None = None            # heliographic latitude, when the source catalog has it


VALID_EVENT_TIMES = {"start", "peak"}
VALID_INTERVAL_MODES = {"left_closed", "right_closed"}


def _normalize_active_region(value) -> str | None:
    # A catalog with any missing active_region cells gets read by pandas as
    # float64 (NaN for missing), so a present value like 4456 arrives as the
    # Python float 4456.0, not a string. Normalize both cases here so
    # FlareEvent.active_region actually matches its str | None type.
    if pd.isna(value):
        return None
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _normalize_hhmm(value, column: str, path) -> str:
    """
    Validate and zero-pad a raw start/peak/end cell into a 4-digit HHMM
    string (e.g. 331 -> "0331"), raising a clear ValueError for anything
    that isn't a real time of day.
    """
    if pd.isna(value):
        raise ValueError(
            f"Flare catalog at {path} has a missing value in column '{column}': "
            "expected a 4-digit HHMM time (e.g. 0309)."
        )

    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(
                f"Flare catalog at {path} has an invalid value {value!r} in "
                f"column '{column}': expected a 4-digit HHMM time (e.g. 0309)."
            )
        value = int(value)

    text = str(value).strip()
    if not text.isdigit() or len(text) > 4:
        raise ValueError(
            f"Flare catalog at {path} has an invalid value {text!r} in "
            f"column '{column}': expected a 4-digit HHMM time (e.g. 0309)."
        )

    text = text.zfill(4)
    hour, minute = int(text[:2]), int(text[2:])
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(
            f"Flare catalog at {path} has an invalid time {text!r} in column "
            f"'{column}': hour must be 00-23 and minute must be 00-59."
        )

    return text


def _combine_date_and_hhmm(df: pd.DataFrame, date_col: str, time_col: str, path) -> pd.Series:
    """
    Build full timestamps from a date column and an HHMM time-of-day column.

    NOAA-style catalogs give start/peak/end as bare 4-digit times with no
    date attached (e.g. "0309" for 03:09). Calling pd.to_datetime() on that
    value alone misreads it as a Unix timestamp near 1970-01-01, so the date
    and time must always be combined and parsed together with an explicit
    format, never pandas' automatic inference.
    """
    normalized_times = df[time_col].apply(lambda v: _normalize_hhmm(v, time_col, path))
    combined = df[date_col].astype(str).str.strip() + " " + normalized_times
    return pd.to_datetime(combined, format="%Y-%m-%d %H%M")


def _parse_full_timestamp_column(df: pd.DataFrame, column: str, path) -> pd.Series:
    """
    Parse a column of full datetime strings (e.g. "2010-05-01 01:39:00"),
    raising a clear ValueError naming the file/column/value on failure.

    Unlike the legacy date+HHMM schema, these columns already carry their
    own complete calendar date, so no date-combination step is needed --
    and none must be applied, or events that cross midnight (start_time on
    one date, peak_time/end_time on the next) would be silently corrupted.
    """
    try:
        return pd.to_datetime(df[column])
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Flare catalog at {path} has an unparseable value in column "
            f"'{column}': {exc}"
        ) from None


class EventMatcher:
    """
    Given an image timestamp, finds all flares that happened
    within a certain number of hours after that image was taken.

    Accepts either of two catalog schemas, auto-detected from the CSV's
    columns:

    - The legacy scraper schema: date, start, peak, end, class,
      active_region -- start/peak/end are bare 4-digit HHMM times combined
      with the date column.
    - The full-timestamp schema: start_time, peak_time, end_time, class,
      active_region (optionally fl_lon, fl_lat) -- each already a complete
      datetime string, used as-is with no date-combination step. This is
      the schema `flare_indexer.loaders.adapt_goes_catalog` produces from
      the lab's integrated GOES catalog, and it's what to reach for when a
      flare's start/peak/end may fall on different calendar dates (e.g. a
      flare that starts just before midnight).
    """

    REQUIRED_CATALOG_COLUMNS = {"date", "start", "peak", "end", "class", "active_region"}
    FULL_TIMESTAMP_CATALOG_COLUMNS = {"start_time", "peak_time", "end_time", "class", "active_region"}

    def __init__(self, catalog_path: str | Path):
        # Load the catalog once when the object is created
        # so we don't re-read the file every time we query
        self._catalog = self._load_catalog(catalog_path)
        # Lazily built, cached {event_time: (sorted_values, original_positions)}
        # views used by query()'s searchsorted fast path -- see _sorted_view().
        self._sorted_view_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def _load_catalog(self, path: str | Path) -> pd.DataFrame:
        """
        Load and validate the flare catalog CSV.

        Raises ValueError if the file has no columns at all (completely
        empty file) or if it matches neither recognized schema. A catalog
        with a header row but zero data rows is valid: it produces an
        empty catalog, so every query() call against it simply returns no
        flares.
        """
        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            raise ValueError(
                f"Flare catalog at {path} is empty: no header row / columns found."
            ) from None

        columns = set(df.columns)
        legacy_missing = self.REQUIRED_CATALOG_COLUMNS - columns
        full_missing = self.FULL_TIMESTAMP_CATALOG_COLUMNS - columns

        if not legacy_missing:
            return self._load_legacy_hhmm_catalog(df, path)
        if not full_missing:
            return self._load_full_timestamp_catalog(df, path)

        # Neither schema is fully satisfied -- report whichever the caller
        # most likely intended (fewer missing columns), preferring the
        # legacy schema on a tie for backward-compatible error messages.
        if len(legacy_missing) <= len(full_missing):
            raise ValueError(
                f"Flare catalog at {path} is missing required column(s): "
                f"{', '.join(sorted(legacy_missing))}"
            )
        raise ValueError(
            f"Flare catalog at {path} is missing required column(s): "
            f"{', '.join(sorted(full_missing))}"
        )

    def _load_legacy_hhmm_catalog(self, df: pd.DataFrame, path) -> pd.DataFrame:
        # Rename to the internal names, then build real timestamps by
        # combining each HHMM time-of-day value with the date column.
        df = df.rename(columns={
            "peak": "peak_time",
            "start": "start_time",
            "end": "end_time",
            "class": "goes_class",
        })
        df["peak_time"] = _combine_date_and_hhmm(df, "date", "peak_time", path)
        df["start_time"] = _combine_date_and_hhmm(df, "date", "start_time", path)
        df["end_time"] = _combine_date_and_hhmm(df, "date", "end_time", path)

        return df.sort_values("peak_time").reset_index(drop=True)

    def _load_full_timestamp_catalog(self, df: pd.DataFrame, path) -> pd.DataFrame:
        df = df.rename(columns={"class": "goes_class"})
        df["start_time"] = _parse_full_timestamp_column(df, "start_time", path)
        df["peak_time"] = _parse_full_timestamp_column(df, "peak_time", path)
        df["end_time"] = _parse_full_timestamp_column(df, "end_time", path)

        return df.sort_values("peak_time").reset_index(drop=True)

    def _sorted_view(self, event_time: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Cached, sorted (values, original_positions) for the given event-time
        basis, built once per basis and reused across query() calls so a
        DatasetBuilder run over tens of thousands of images doesn't re-sort
        the catalog every time -- see query()'s searchsorted fast path.
        """
        # Tests (and any other caller) sometimes build an EventMatcher via
        # __new__() and set _catalog directly, bypassing __init__ -- fall
        # back to an empty cache rather than requiring that pattern to also
        # know about this private cache attribute.
        cache = self.__dict__.setdefault("_sorted_view_cache", {})
        if event_time not in cache:
            column = "start_time" if event_time == "start" else "peak_time"
            values = self._catalog[column].to_numpy()
            order = np.argsort(values, kind="stable")
            cache[event_time] = (values[order], order)
        return cache[event_time]

    def query(
        self,
        image_time: pd.Timestamp,
        prediction_window_hours: int,
        active_region: str | None = None,
        event_time: str = "peak",
        interval_mode: str = "left_closed",
    ) -> list[FlareEvent]:
        """
        Returns all flares whose event time falls inside the prediction
        window.

        event_time selects which of a flare's timestamps is checked against
        the window:
        - "peak" (default, preserves original behavior): the flare's
          peak_time.
        - "start": the flare's start_time. This matches how Dr. Pandey's
          legacy full-disk labeling script defines its prediction window.

        interval_mode selects the window's boundary convention, given
        window_end = image_time + prediction_window_hours:
        - "left_closed" (default, preserves original behavior):
          [image_time, window_end) -- includes image_time itself, excludes
          window_end exactly.
        - "right_closed": (image_time, window_end] -- excludes image_time
          itself, includes window_end exactly. This matches the legacy
          full-disk script's convention.

        Example: image taken at 10:00, window = 24 hours, defaults
        → returns all flares with peak_time in [10:00, 10:00 next day)

        If active_region is given, only flares attributed to that exact
        active region are returned; flares with a missing active_region
        never match, and flares from any other active region are excluded
        regardless of strength. If active_region is None (the default),
        matching is full-disk: any flare in the window counts.
        """
        if event_time not in VALID_EVENT_TIMES:
            raise ValueError(f"event_time must be one of {sorted(VALID_EVENT_TIMES)}, got {event_time!r}")
        if interval_mode not in VALID_INTERVAL_MODES:
            raise ValueError(f"interval_mode must be one of {sorted(VALID_INTERVAL_MODES)}, got {interval_mode!r}")

        window_end = image_time + pd.Timedelta(hours=prediction_window_hours)

        sorted_values, sort_positions = self._sorted_view(event_time)
        image_time_np = np.datetime64(image_time)
        window_end_np = np.datetime64(window_end)

        if interval_mode == "left_closed":
            lo = np.searchsorted(sorted_values, image_time_np, side="left")
            hi = np.searchsorted(sorted_values, window_end_np, side="left")
        else:
            lo = np.searchsorted(sorted_values, image_time_np, side="right")
            hi = np.searchsorted(sorted_values, window_end_np, side="right")

        candidate_positions = sort_positions[lo:hi]
        candidates = self._catalog.iloc[candidate_positions]

        if active_region is not None:
            normalized_catalog_ar = candidates["active_region"].apply(_normalize_active_region)
            candidates = candidates[normalized_catalog_ar.to_numpy() == active_region]

        # Convert the matching rows into FlareEvent objects and return them
        # If no flares match, this returns an empty list → label will be 0
        return [
            FlareEvent(
                peak_time=row["peak_time"],
                goes_class=row["goes_class"],
                start_time=row["start_time"],
                active_region=_normalize_active_region(row.get("active_region")),
                end_time=row.get("end_time"),
                fl_lon=row.get("fl_lon"),
                fl_lat=row.get("fl_lat"),
            )
            for _, row in candidates.iterrows()
        ]
