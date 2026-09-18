import re
from pathlib import Path
from typing import Iterable, Union

import pandas as pd

from .events import _normalize_active_region

# Matches filenames like "HMI.m2010.12.21_21.00.00.jpg", regardless of how
# deep the surrounding directory path is -- only the basename is checked.
_FILENAME_PATTERN = re.compile(
    r"^HMI\.m(?P<year>\d{4})\.(?P<month>\d{2})\.(?P<day>\d{2})_"
    r"(?P<hour>\d{2})\.(?P<minute>\d{2})\.(?P<second>\d{2})\.jpg$"
)

GOES_CATALOG_REQUIRED_COLUMNS = {
    "start_time",
    "peak_time",
    "end_time",
    "goes_class",
    "noaa_active_region",
}


def _read_paths_file(path: Path) -> list[str]:
    # utf-8-sig transparently strips a leading BOM if present (some sources
    # export these path lists with one) and behaves like plain utf-8 otherwise.
    with open(path, encoding="utf-8-sig") as f:
        return [line.strip() for line in f if line.strip()]


def build_image_index_from_filenames(
    paths: Union[str, Path, Iterable[str]],
    output_path: Union[str, Path, None] = None,
) -> pd.DataFrame:
    """
    Build a DatasetBuilder-compatible image index from HMI image file paths.

    `paths` is either an iterable of file paths, or a path to a headerless
    single-column text/CSV file listing one path per line. Each filename is
    expected to look like "HMI.m{YYYY}.{MM}.{DD}_{HH}.{MM}.{SS}.jpg";
    directory depth doesn't matter, only the basename is parsed.

    Returns a DataFrame with a `timestamp` column (pandas datetime, sorted
    ascending -- required for DatasetBuilder's sequence mode) and an
    `image_path` column preserving the original path. Raises ValueError
    naming the offending filename if any path doesn't match the expected
    pattern.
    """
    if isinstance(paths, (str, Path)):
        paths = _read_paths_file(Path(paths))
    else:
        paths = list(paths)

    records = []
    for path in paths:
        filename = Path(path).name
        match = _FILENAME_PATTERN.match(filename)
        if not match:
            raise ValueError(
                f"Filename {filename!r} (from path {str(path)!r}) does not match "
                "the expected pattern 'HMI.m{YYYY}.{MM}.{DD}_{HH}.{MM}.{SS}.jpg'"
            )
        timestamp = pd.Timestamp(
            year=int(match["year"]),
            month=int(match["month"]),
            day=int(match["day"]),
            hour=int(match["hour"]),
            minute=int(match["minute"]),
            second=int(match["second"]),
        )
        records.append({"timestamp": timestamp, "image_path": str(path)})

    df = pd.DataFrame(records, columns=["timestamp", "image_path"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    if output_path is not None:
        df.to_csv(output_path, index=False)

    return df


def adapt_goes_catalog(
    input_path: Union[str, Path],
    output_path: Union[str, Path, None] = None,
) -> pd.DataFrame:
    """
    Adapt a lab GOES flare catalog (goes_flares_integrated.csv-shaped) into
    the `date, start, peak, end, class, active_region` schema EventMatcher
    expects.

    Source columns used: `start_time`, `peak_time`, `end_time` (full
    datetime strings), `goes_class`, `noaa_active_region` (nullable float).
    All other source columns are ignored. `date` is derived from
    `peak_time`'s date; `start`/`peak`/`end` are re-encoded as 4-digit
    zero-padded HHMM strings (EventMatcher parses bare HHMM, not full
    datetimes). `noaa_active_region` is normalized to the same
    string-or-None convention as `events._normalize_active_region`.
    """
    input_path = Path(input_path)

    try:
        df = pd.read_csv(input_path)
    except pd.errors.EmptyDataError:
        raise ValueError(
            f"GOES catalog at {input_path} is empty: no header row / columns found."
        ) from None

    missing = GOES_CATALOG_REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"GOES catalog at {input_path} is missing required column(s): "
            f"{', '.join(sorted(missing))}"
        )

    start = pd.to_datetime(df["start_time"])
    peak = pd.to_datetime(df["peak_time"])
    end = pd.to_datetime(df["end_time"])

    adapted = pd.DataFrame({
        "date": peak.dt.strftime("%Y-%m-%d"),
        "start": start.dt.strftime("%H%M"),
        "peak": peak.dt.strftime("%H%M"),
        "end": end.dt.strftime("%H%M"),
        "class": df["goes_class"],
        "active_region": df["noaa_active_region"].apply(_normalize_active_region),
    })

    if output_path is not None:
        adapted.to_csv(output_path, index=False)

    return adapted
