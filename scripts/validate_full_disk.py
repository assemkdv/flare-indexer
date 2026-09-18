"""
Full-disk real-data validation and legacy-comparison script.

Builds a full-disk, M-class binary-threshold labeled dataset from the lab's
real HMI image list and integrated GOES catalog, configured to match
Dr. Pandey's legacy full-disk labeling convention (start_time-based
matching, (t, t+24h] windows -- see labeling.py in the explainingFullDisk
repo). Optionally compares the result against the legacy script's own
output, by timestamp, and reports agreement.

Usage:
    python scripts/validate_full_disk.py
    python scripts/validate_full_disk.py --legacy-labels <path-or-url>
    python scripts/validate_full_disk.py --image-list <path> --event-catalog <path> --output out.csv

Local paths and http(s) URLs are both accepted for --image-list,
--event-catalog, and --legacy-labels. URLs are downloaded once into
scripts/.cache/ (gitignored) and reused on subsequent runs; delete that
directory to force a re-download.
"""

import argparse
import re
import time
import urllib.request
from pathlib import Path

import pandas as pd

from flare_indexer import BinaryThresholdStrategy, DatasetBuilder, EventMatcher, FlareClassifier, MaxFlareStrategy
from flare_indexer.loaders import adapt_goes_catalog, build_image_index_from_filenames

DEFAULT_IMAGE_LIST_URL = (
    "https://raw.githubusercontent.com/chetrajpandey/explainingFullDisk/"
    "main/labeling/data_labels/totalfiles_jpg_512.csv"
)
DEFAULT_EVENT_CATALOG_URL = (
    "https://raw.githubusercontent.com/chetrajpandey/explainingFullDisk/"
    "main/labeling/data_source/goes_flares_integrated.csv"
)
DEFAULT_LEGACY_LABELS_URL = (
    "https://raw.githubusercontent.com/chetrajpandey/explainingFullDisk/"
    "main/labeling/data_labels/NEW_full_dataset_cleaned_1_hours_with_loc_and_time_new.csv"
)

CACHE_DIR = Path(__file__).parent / ".cache"
OUTPUT_DIR = Path(__file__).parent / "output"

_LEGACY_NO_FLARE_VALUES = {"nf", "", "0", "0.0", "nan"}
_GOES_CLASS_RE = re.compile(r"^[ABCMX]\d+(\.\d+)?$", re.IGNORECASE)


def _is_url(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")


def _resolve_source(value: str, cache_name: str) -> Path:
    """
    Accepts either a local path or an http(s) URL. URLs are downloaded once
    into CACHE_DIR and reused on later runs (delete the cache to refresh).
    """
    if not _is_url(value):
        return Path(value)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dest = CACHE_DIR / cache_name
    if not dest.exists():
        print(f"Downloading {value} -> {dest}")
        urllib.request.urlretrieve(value, dest)
    else:
        print(f"Using cached download: {dest}")
    return dest


def _binarize_legacy_class(raw) -> tuple[int | None, str]:
    """
    Returns (binary_label, status). status is "ok" for a value cleanly
    parsed (flare class or an explicit no-flare marker), "malformed" for
    anything unrecognized (excluded from comparison, not silently coerced).

    Handles the legacy script's various no-flare representations: the
    literal string "NF", blank, NaN, or "0"/"0.0".
    """
    if pd.isna(raw):
        return 0, "ok"
    text = str(raw).strip()
    if text.lower() in _LEGACY_NO_FLARE_VALUES:
        return 0, "ok"
    if _GOES_CLASS_RE.match(text):
        return (1 if text[0].upper() in ("M", "X") else 0), "ok"
    return None, "malformed"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image-list", default=DEFAULT_IMAGE_LIST_URL,
                         help="Local path or URL to the headerless HMI image path list.")
    parser.add_argument("--event-catalog", default=DEFAULT_EVENT_CATALOG_URL,
                         help="Local path or URL to the integrated GOES flare catalog.")
    parser.add_argument("--legacy-labels", default=None,
                         help="Local path or URL to Dr. Pandey's legacy labeled output, for comparison. "
                              f"Pass 'default' to use the known URL ({DEFAULT_LEGACY_LABELS_URL}).")
    parser.add_argument("--output", default=str(OUTPUT_DIR / "full_disk_labels.csv"),
                         help="Where to write the generated (timestamp, image_path, label, max_flux) CSV.")
    parser.add_argument("--duplicate-policy", default="first", choices=["error", "first", "last"],
                         help="How to resolve duplicate image timestamps (default: first).")
    args = parser.parse_args()
    if args.legacy_labels == "default":
        args.legacy_labels = DEFAULT_LEGACY_LABELS_URL
    return args


def main():
    args = parse_args()
    t0 = time.monotonic()

    image_list_path = _resolve_source(args.image_list, "totalfiles_jpg_512.csv")
    event_catalog_path = _resolve_source(args.event_catalog, "goes_flares_integrated.csv")

    print("\n=== Building image index (strict=False, duplicate_policy=%r) ===" % args.duplicate_policy)
    image_index, image_report = build_image_index_from_filenames(
        image_list_path, strict=False, duplicate_policy=args.duplicate_policy, return_report=True,
    )
    print(f"  total source rows:          {image_report.total_input_rows}")
    print(f"  valid image rows before dedup: {image_report.valid_image_rows_before_dedup}")
    print(f"  excluded malformed/non-image rows: {image_report.excluded_row_count}")
    if image_report.excluded_rows:
        print(f"    e.g. {image_report.excluded_rows[:5]}")
    print(f"  duplicate timestamps:        {image_report.duplicate_timestamp_count}")
    print(f"  final normalized images:     {image_report.final_row_count}")
    print(f"  timestamp range: {image_index['timestamp'].min()} -> {image_index['timestamp'].max()}")

    print("\n=== Adapting GOES catalog ===")
    catalog = adapt_goes_catalog(event_catalog_path)
    catalog_cache_path = CACHE_DIR / "flare_catalog_adapted.csv"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(catalog_cache_path, index=False)
    print(f"  total GOES events: {len(catalog)}")
    print(f"  event date range (peak_time): {catalog['peak_time'].min()} -> {catalog['peak_time'].max()}")

    image_index_cache_path = CACHE_DIR / "image_index_normalized.csv"
    image_index.to_csv(image_index_cache_path, index=False)

    print("\n=== Building labels (full-disk, event_time=start, interval_mode=right_closed, M-threshold) ===")
    binary_builder = DatasetBuilder(
        prediction_window=24,
        strategy=BinaryThresholdStrategy(threshold="M"),
        target="full_disk",
        event_time="start",
        interval_mode="right_closed",
    )
    flux_builder = DatasetBuilder(
        prediction_window=24,
        strategy=MaxFlareStrategy(),
        target="full_disk",
        event_time="start",
        interval_mode="right_closed",
    )
    binary_result = binary_builder.build(image_index_cache_path, catalog_cache_path)
    flux_result = flux_builder.build(image_index_cache_path, catalog_cache_path)

    assert len(binary_result) == len(image_index) == len(flux_result)
    assert (binary_result["timestamp"].values == image_index["timestamp"].values).all()

    output = image_index[["timestamp", "image_path"]].copy()
    output["label"] = binary_result["label"].values
    output["max_flux"] = flux_result["label"].values

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)

    positives = int((output["label"] == 1).sum())
    negatives = int((output["label"] == 0).sum())
    print(f"  total generated labels: {len(output)}")
    print(f"  positive (>=M): {positives}")
    print(f"  negative:       {negatives}")
    print(f"  label distribution: {dict(output['label'].value_counts())}")
    print(f"  output written to: {output_path}")

    if args.legacy_labels:
        _compare_to_legacy(args.legacy_labels, output, image_report, catalog_cache_path)

    runtime = time.monotonic() - t0
    print(f"\n=== Runtime: {runtime:.2f}s ===")


def _compare_to_legacy(legacy_source: str, generated: pd.DataFrame, image_report, catalog_path: Path) -> None:
    print("\n=== Comparing against legacy labels ===")
    legacy_path = _resolve_source(legacy_source, "legacy_labels.csv")
    legacy_raw = pd.read_csv(legacy_path)

    if "label" not in legacy_raw.columns:
        print("  legacy file has no 'label' column -- cannot compare.")
        return

    # legacy's "label" column holds relative paths like
    # "2010/12/06/HMI.m2010.12.06_07.00.00.jpg" -- reuse the same filename
    # parser used for the real image list to extract timestamps from it.
    legacy_index, legacy_parse_report = build_image_index_from_filenames(
        legacy_raw["label"].astype(str).tolist(), strict=False, duplicate_policy="first", return_report=True,
    )
    if legacy_parse_report.excluded_row_count:
        print(f"  {legacy_parse_report.excluded_row_count} legacy rows had unparseable 'label' filenames "
              "(excluded from comparison).")

    # Map each legacy row's raw 'label' path string back to the timestamp
    # build_image_index_from_filenames parsed from it. Built as a plain
    # dict (first-occurrence wins) rather than an index lookup, since a
    # duplicate 'label' string (or one dropped by duplicate_policy="first"
    # above) must resolve predictably rather than ambiguously.
    path_to_timestamp: dict[str, pd.Timestamp] = {}
    for _, idx_row in legacy_index.iterrows():
        path_to_timestamp.setdefault(idx_row["image_path"], idx_row["timestamp"])

    legacy_df = legacy_raw.copy()
    legacy_df["timestamp"] = legacy_raw["label"].astype(str).map(path_to_timestamp)
    legacy_df = legacy_df.dropna(subset=["timestamp"])

    binarized = legacy_df["goes_class"].apply(_binarize_legacy_class)
    legacy_df["legacy_label"] = binarized.apply(lambda t: t[0])
    legacy_df["legacy_status"] = binarized.apply(lambda t: t[1])

    malformed_count = int((legacy_df["legacy_status"] == "malformed").sum())
    legacy_eligible = legacy_df[legacy_df["legacy_status"] == "ok"].copy()
    legacy_eligible["legacy_label"] = legacy_eligible["legacy_label"].astype(int)
    legacy_eligible = legacy_eligible.drop_duplicates(subset="timestamp", keep="first")

    generated_eligible = generated.drop_duplicates(subset="timestamp", keep="first")

    merged = generated_eligible.merge(
        legacy_eligible[["timestamp", "legacy_label", "goes_class"]],
        on="timestamp", how="inner",
    )
    generated_only = set(generated_eligible["timestamp"]) - set(legacy_eligible["timestamp"])
    legacy_only = set(legacy_eligible["timestamp"]) - set(generated_eligible["timestamp"])

    matches = merged["label"] == merged["legacy_label"]
    mismatches = merged[~matches]
    agreement_pct = 100.0 * matches.sum() / len(merged) if len(merged) else float("nan")

    print(f"  generated rows eligible for comparison: {len(generated_eligible)}")
    print(f"  legacy rows eligible for comparison:     {len(legacy_eligible)} "
          f"(excluded {malformed_count} malformed legacy rows)")
    print(f"  overlapping timestamps:                  {len(merged)}")
    print(f"  generated timestamps not in legacy:       {len(generated_only)}")
    print(f"  legacy timestamps not in generated:       {len(legacy_only)}")
    print(f"  exact binary-label matches:               {int(matches.sum())}")
    print(f"  binary-label mismatches:                  {len(mismatches)}")
    print(f"  agreement percentage:                     {agreement_pct:.4f}%")

    false_positives = mismatches[(mismatches["label"] == 1) & (mismatches["legacy_label"] == 0)]
    false_negatives = mismatches[(mismatches["label"] == 0) & (mismatches["legacy_label"] == 1)]
    print(f"  false positives (generated=1, legacy=0):  {len(false_positives)}")
    print(f"  false negatives (generated=0, legacy=1):  {len(false_negatives)}")

    if len(mismatches):
        print("\n  representative mismatch rows:")
        for _, row in mismatches.head(10).iterrows():
            print(f"    {row['timestamp']}  generated={row['label']}  legacy={row['legacy_label']}"
                  f"  generated_max_flux={row['max_flux']:.2e}  legacy_goes_class={row['goes_class']!r}")

    _categorize_mismatches(mismatches, image_report, catalog_path)


def _categorize_mismatches(mismatches: pd.DataFrame, image_report, catalog_path: Path) -> None:
    """
    Best-effort, per-row explanation of remaining binary-label mismatches.

    "legacy_23h59m59s_boundary" is checked directly rather than guessed at:
    Dr. Pandey's legacy script computes its window as
    (pws, pws + 23:59:59] -- one second short of a true 24 hours (see
    labeling.py's create_labels(): pwe starts at pws + '23:59:59', not
    pws + 24h, and both are advanced by the same hourly step together). A
    flare whose start_time lands exactly on pws + 24:00:00 is therefore
    INSIDE our interval_mode="right_closed" window (t, t+24h] but OUTSIDE
    legacy's (t, t+23:59:59] window -- a real, deterministic boundary-
    precision difference, not a bug in either implementation.
    """
    print("\n=== Mismatch categorization (Task 7) ===")
    if not len(mismatches):
        print("  no mismatches to categorize.")
        return

    matcher = EventMatcher(catalog_path)
    classifier = FlareClassifier()
    duplicated_paths = set(image_report.duplicate_rows)

    counts = {
        "legacy_23h59m59s_boundary": 0,
        "duplicate_policy_candidate": 0,
        "unknown_unexplained": 0,
    }
    boundary_examples = []
    for _, row in mismatches.iterrows():
        ts = row["timestamp"]
        window_end = ts + pd.Timedelta(hours=24)
        flares = matcher.query(ts, 24, event_time="start", interval_mode="right_closed")
        qualifying = [f for f in flares if classifier.is_strong(f.goes_class, "M")]
        at_boundary = bool(qualifying) and all(f.start_time == window_end for f in qualifying)

        if at_boundary:
            counts["legacy_23h59m59s_boundary"] += 1
            boundary_examples.append((ts, qualifying[0].goes_class, qualifying[0].start_time))
        elif row["image_path"] in duplicated_paths:
            counts["duplicate_policy_candidate"] += 1
        else:
            counts["unknown_unexplained"] += 1

    for category, count in counts.items():
        print(f"  {category}: {count}")
    if boundary_examples:
        print("\n  legacy_23h59m59s_boundary examples (image_time, qualifying_class, flare_start_time):")
        for ts, goes_class, start_time in boundary_examples[:10]:
            print(f"    {ts}  {goes_class}  start={start_time}  (== t + exactly 24h)")

    print(
        "\n  malformed_legacy_row: excluded upstream, before comparison "
        "(see 'excluded malformed legacy rows' above) -- 0 by definition here."
    )
    print(
        "  numerical_goes_ordering_vs_lexical: structurally cannot cause a binary-label "
        "mismatch -- both orderings agree on whether ANY M/X-class flare is present in a "
        "window (the first character alone already sorts correctly both ways); it only "
        "affects which single flare legacy reports as 'the' max, not the M-threshold label."
    )
    print(
        "  event_time / interval_boundary (basis) differences: not applicable to remaining "
        "mismatches by construction -- this run already uses event_time='start', "
        "interval_mode='right_closed' to match the legacy convention. Run this script "
        "with the library's *default* settings (peak/left_closed) separately to see the "
        "aggregate agreement-rate impact of that choice."
    )


if __name__ == "__main__":
    main()
