import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Union

import pandas as pd

from .events import _normalize_active_region

# Matches filenames like "HMI.m2010.12.21_21.00.00.jpg", regardless of how
# deep the surrounding directory path is -- only the basename is checked.
# Year/month/day/hour/minute/second are captured as raw 2-4 digit groups;
# calendar validity (e.g. month <= 12) is checked separately when the
# timestamp is actually constructed, so a shape-valid but date-invalid
# filename (e.g. "HMI.m2010.13.40_...") is still caught and reported.
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
GOES_CATALOG_OPTIONAL_COLUMNS = ("fl_lon", "fl_lat")

VALID_DUPLICATE_POLICIES = {"error", "first", "last"}

# Cap on how many excluded/duplicate rows a report keeps verbatim, so a
# badly-formed multi-million-row file doesn't blow up memory just to report
# on itself. Counts in the report are always exact regardless of this cap.
_MAX_REPORT_SAMPLES = 200


@dataclass
class ImageIndexReport:
    """
    Validation report for build_image_index_from_filenames().

    total_input_rows : how many non-blank lines/paths were given.
    valid_image_rows_before_dedup : how many parsed as valid HMI image
        filenames, before duplicate-timestamp handling.
    excluded_row_count : total rows that didn't parse as a valid HMI image
        filename (malformed shape, or shape-valid but an invalid calendar
        date/time). Always accurate, even if `excluded_rows` is truncated.
    excluded_rows : up to the first `_MAX_REPORT_SAMPLES` excluded raw path
        strings, for inspection.
    duplicate_timestamp_count : number of distinct timestamps that appeared
        more than once among the valid rows.
    duplicate_rows : up to the first `_MAX_REPORT_SAMPLES` image_path values
        involved in a duplicate timestamp, for inspection.
    final_row_count : row count of the returned DataFrame, after applying
        duplicate_policy.
    duplicate_policy : the policy that was applied ("error" raises before a
        report is ever produced, so this will only ever be "first" or
        "last" on a report that was actually returned).
    """

    total_input_rows: int
    valid_image_rows_before_dedup: int
    excluded_row_count: int
    excluded_rows: list = field(default_factory=list)
    duplicate_timestamp_count: int = 0
    duplicate_rows: list = field(default_factory=list)
    final_row_count: int = 0
    duplicate_policy: str = "error"


def _read_paths_file(path: Path) -> list[str]:
    # utf-8-sig transparently strips a leading BOM if present (some sources
    # export these path lists with one) and behaves like plain utf-8 otherwise.
    with open(path, encoding="utf-8-sig") as f:
        return [line.strip() for line in f if line.strip()]


def build_image_index_from_filenames(
    paths: Union[str, Path, Iterable[str]],
    output_path: Union[str, Path, None] = None,
    strict: bool = True,
    duplicate_policy: str = "error",
    return_report: bool = False,
):
    """
    Build a DatasetBuilder-compatible image index from HMI image file paths.

    `paths` is either an iterable of file paths, or a path to a headerless
    single-column text/CSV file listing one path per line (a UTF-8 BOM at
    the start of such a file, if present, is stripped automatically). Each
    filename is expected to look like
    "HMI.m{YYYY}.{MM}.{DD}_{HH}.{MM}.{SS}.jpg"; directory depth doesn't
    matter, only the basename is parsed.

    strict controls how rows that aren't valid HMI image filenames are
    handled (real source files, e.g. a directory walk over an entire data
    root, can contain unrelated non-image paths mixed in):
    - True (default, matches the original behavior): raise ValueError
      naming the first offending path.
    - False: exclude those rows rather than raising. They are never
      silently dropped without a trace -- pass return_report=True to see
      exactly what was excluded and why.

    duplicate_policy controls what happens when multiple rows parse to the
    exact same timestamp:
    - "error" (default): raise ValueError naming the duplicate timestamps.
    - "first": keep the first-encountered row for each duplicated timestamp.
    - "last": keep the last-encountered row for each duplicated timestamp.

    Returns a DataFrame with a `timestamp` column (pandas datetime, sorted
    ascending -- required for DatasetBuilder's sequence mode) and an
    `image_path` column preserving the original path. If return_report is
    True, instead returns (DataFrame, ImageIndexReport).
    """
    if duplicate_policy not in VALID_DUPLICATE_POLICIES:
        raise ValueError(
            f"duplicate_policy must be one of {sorted(VALID_DUPLICATE_POLICIES)}, "
            f"got {duplicate_policy!r}"
        )

    if isinstance(paths, (str, Path)):
        paths = _read_paths_file(Path(paths))
    else:
        paths = list(paths)

    total_input_rows = len(paths)
    records = []
    excluded_rows: list[str] = []

    for path in paths:
        filename = Path(path).name
        match = _FILENAME_PATTERN.match(filename)
        timestamp = None
        if match:
            try:
                timestamp = pd.Timestamp(
                    year=int(match["year"]),
                    month=int(match["month"]),
                    day=int(match["day"]),
                    hour=int(match["hour"]),
                    minute=int(match["minute"]),
                    second=int(match["second"]),
                )
            except ValueError as exc:
                if strict:
                    raise ValueError(
                        f"Filename {filename!r} (from path {str(path)!r}) has the right shape "
                        f"but an invalid date/time: {exc}"
                    ) from None
        if timestamp is None:
            if match is None and strict:
                raise ValueError(
                    f"Filename {filename!r} (from path {str(path)!r}) does not match "
                    "the expected pattern 'HMI.m{YYYY}.{MM}.{DD}_{HH}.{MM}.{SS}.jpg'"
                )
            excluded_rows.append(str(path))
            continue
        records.append({"timestamp": timestamp, "image_path": str(path)})

    valid_image_rows_before_dedup = len(records)

    df = pd.DataFrame(records, columns=["timestamp", "image_path"])
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)

    dup_mask = df["timestamp"].duplicated(keep=False)
    duplicate_timestamp_count = int(df.loc[dup_mask, "timestamp"].nunique())
    duplicate_rows_sample = df.loc[dup_mask, "image_path"].tolist()[:_MAX_REPORT_SAMPLES]

    if dup_mask.any():
        if duplicate_policy == "error":
            dupe_timestamps = sorted(df.loc[dup_mask, "timestamp"].unique())
            raise ValueError(
                f"Image paths contain {len(dupe_timestamps)} duplicate timestamp(s): "
                f"{', '.join(str(t) for t in dupe_timestamps[:10])}"
                + (", ..." if len(dupe_timestamps) > 10 else "")
            )
        keep = "first" if duplicate_policy == "first" else "last"
        df = df.drop_duplicates(subset="timestamp", keep=keep).reset_index(drop=True)

    if output_path is not None:
        df.to_csv(output_path, index=False)

    if not return_report:
        return df

    report = ImageIndexReport(
        total_input_rows=total_input_rows,
        valid_image_rows_before_dedup=valid_image_rows_before_dedup,
        excluded_row_count=len(excluded_rows),
        excluded_rows=excluded_rows[:_MAX_REPORT_SAMPLES],
        duplicate_timestamp_count=duplicate_timestamp_count,
        duplicate_rows=duplicate_rows_sample,
        final_row_count=len(df),
        duplicate_policy=duplicate_policy,
    )
    return df, report


def adapt_goes_catalog(
    input_path: Union[str, Path],
    output_path: Union[str, Path, None] = None,
) -> pd.DataFrame:
    """
    Adapt a lab GOES flare catalog (goes_flares_integrated.csv-shaped) into
    the full-timestamp `start_time, peak_time, end_time, class,
    active_region[, fl_lon, fl_lat]` schema EventMatcher accepts directly.

    Source columns used: `start_time`, `peak_time`, `end_time` (full
    datetime strings), `goes_class`, `noaa_active_region` (nullable float),
    and, if present, `fl_lon`/`fl_lat`. All other source columns are
    ignored. `noaa_active_region` is normalized to the same string-or-None
    convention as `events._normalize_active_region`.

    Unlike earlier versions of this function, timestamps are NOT re-encoded
    into a single date plus bare HHMM times: that round-trip assumed
    start/peak/end always share one calendar date, which silently corrupts
    any flare that crosses midnight (e.g. start_time on one date,
    peak_time/end_time on the next). Each timestamp keeps its own complete,
    independent date.
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

    adapted = pd.DataFrame({
        "start_time": pd.to_datetime(df["start_time"]),
        "peak_time": pd.to_datetime(df["peak_time"]),
        "end_time": pd.to_datetime(df["end_time"]),
        "class": df["goes_class"],
        "active_region": df["noaa_active_region"].apply(_normalize_active_region),
    })

    for optional_column in GOES_CATALOG_OPTIONAL_COLUMNS:
        if optional_column in df.columns:
            adapted[optional_column] = df[optional_column]

    if output_path is not None:
        adapted.to_csv(output_path, index=False)

    return adapted
