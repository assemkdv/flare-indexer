import pandas as pd
from pathlib import Path
from dataclasses import dataclass


# A simple container that holds the details of one flare event
# Think of it like a row from the NOAA catalog, but as a Python object
@dataclass
class FlareEvent:
    peak_time: pd.Timestamp      # when the flare was at its strongest
    goes_class: str              # flare class letter + number, e.g. "M2.3"
    start_time: pd.Timestamp     # when the flare started
    active_region: str | None    # which region of the sun it came from (can be empty)


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


class EventMatcher:
    """
    Given an image timestamp, finds all flares that happened
    within a certain number of hours after that image was taken.
    """

    REQUIRED_CATALOG_COLUMNS = {"date", "start", "peak", "end", "class", "active_region"}

    def __init__(self, catalog_path: str | Path):
        # Load the catalog once when the object is created
        # so we don't re-read the file every time we query
        self._catalog = self._load_catalog(catalog_path)

    def _load_catalog(self, path: str | Path) -> pd.DataFrame:
        """
        Load and validate the flare catalog CSV.

        Raises ValueError if the file has no columns at all (completely
        empty file) or is missing any required column. A catalog with a
        header row but zero data rows is valid: it produces an empty
        catalog, so every query() call against it simply returns no flares.
        """
        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            raise ValueError(
                f"Flare catalog at {path} is empty: no header row / columns found."
            ) from None

        missing = self.REQUIRED_CATALOG_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(
                f"Flare catalog at {path} is missing required column(s): "
                f"{', '.join(sorted(missing))}"
            )

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
        # end_time isn't exposed on FlareEvent, but it's a required column
        # and must still be validated so a malformed value doesn't pass silently.
        df["end_time"] = _combine_date_and_hhmm(df, "date", "end_time", path)

        # Sort by peak_time so we can do fast lookups later
        df = df.sort_values("peak_time").reset_index(drop=True)

        return df

    def query(
        self,
        image_time: pd.Timestamp,
        prediction_window_hours: int,
        active_region: str | None = None,
    ) -> list[FlareEvent]:
        """
        Returns all flares whose peak falls inside the prediction window.

        Example: image taken at 10:00, window = 24 hours
        → returns all flares with peak_time between 10:00 and 10:00 next day

        If active_region is given, only flares attributed to that exact
        active region are returned; flares with a missing active_region
        never match, and flares from any other active region are excluded
        regardless of strength. If active_region is None (the default),
        matching is full-disk: any flare in the window counts.
        """

        # Calculate the end of the prediction window
        window_end = image_time + pd.Timedelta(hours=prediction_window_hours)

        # Create a True/False mask for every row in the catalog
        # True = this flare's peak falls inside our window
        mask = (
            (self._catalog["peak_time"] >= image_time) &
            (self._catalog["peak_time"] < window_end)
        )

        if active_region is not None:
            normalized_catalog_ar = self._catalog["active_region"].apply(_normalize_active_region)
            mask &= normalized_catalog_ar == active_region

        # Convert the matching rows into FlareEvent objects and return them
        # If no flares match, this returns an empty list → label will be 0
        return [
            FlareEvent(
                peak_time=row["peak_time"],
                goes_class=row["goes_class"],
                start_time=row["start_time"],
                active_region=_normalize_active_region(row.get("active_region")),
            )
            for _, row in self._catalog[mask].iterrows()
        ]
