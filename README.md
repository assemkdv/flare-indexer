flare-indexer builds **labeled datasets for solar flare prediction**:
given solar image timestamps and a NOAA flare catalog, it looks forward in
time from each image (or each sequence of images), checks whether a flare
occurred within a configurable prediction window, and produces a
ready-to-train table of image/sequence -> label pairs.

## Installation

Not yet published to PyPI. Install from source:

```bash
git clone https://github.com/assemkdv/solarflare-labeler.git
cd solarflare-labeler
pip install .
```

> **Note:** the import package and PyPI distribution name are now
> `flare_indexer` / `flare-indexer`, but the GitHub repository itself may
> still be reachable only at the old `solarflare-labeler` URL above until
> the remote repository is renamed too. If `git clone` with a
> `flare-indexer` URL fails, use the URL shown here.

## Where the data comes from

This package does not fetch or maintain flare data itself. Flare catalogs
are produced by the companion [solar-event-scraper](https://github.com/assemkdv/solar-event-scraper)
repository, which handles NOAA data acquisition, deduplication, and
updates. `flare-indexer`'s job starts where the scraper's output ends:
point `DatasetBuilder` at a scraper-produced catalog CSV and an image index
CSV, and it does the labeling.

## How it works (the pipeline)

```
image timestamp(s)
      |
      v
[ EventMatcher ]   -> finds all flares within N hours after the image
      |              (optionally filtered to a specific active region)
      v
[ Strategy ]       -> turns that list of flares into a single label
      |
      v
[ DatasetBuilder ] -> repeats for every image or sequence, returns a table
```

**`FluxConverter`** — converts flare strength text to a number. Flares are
named with a letter + number ("M2.3", "X1.0"). Letters go A < B < C < M < X,
and each step is 10x stronger (a log scale). `to_flux()` converts that text
into a real number (e.g. "M2.3" -> 2.3e-5) so flares can be compared
**numerically**, not lexically (a plain string sort would incorrectly rank
"M9.0" above "M10.0" — see "Numerical vs. lexical GOES ordering" below).

**`FlareClassifier`** — classification checks built on `FluxConverter`.
Answers "is this flare strong enough to matter?" (`is_strong`, default
threshold = M-class).

**`EventMatcher`** — searches the catalog by time (and, optionally, by
active region). Given a reference timestamp and a window (e.g. 24 hours),
it returns every flare in that window. No flares -> empty list -> "no
flare" label. Which flare timestamp is checked (`event_time`) and the
window's boundary convention (`interval_mode`) are both configurable —
see "Window semantics" below.

**`DatasetBuilder`** — runs the whole pipeline. Takes a file of image
timestamps and the flare catalog, walks through every image (or every
sequence of images), applies the matcher + strategy, and returns a
`DataFrame`.

## Flare catalog CSV (scraper output format)

This is the exact schema produced by `solar-event-scraper`:

| Column | Type | Description |
|---|---|---|
| `date` | `YYYY-MM-DD` | Calendar date the flare occurred |
| `start` | 4-digit `HHMM` | When the flare began, e.g. `0309` for 03:09 |
| `peak` | 4-digit `HHMM` | When the flare reached maximum intensity — **this is what matching uses** |
| `end` | 4-digit `HHMM` | When the flare ended |
| `class` | string | GOES class, e.g. `"M2.3"`, `"X1.0"` |
| `active_region` | NOAA AR number or empty | Which active region the flare came from, if known |

Example row: `2024-01-23,0309,0331,0338,M5.1,3559`.

`start`/`peak`/`end` are bare times-of-day with no date attached, so
`flare-indexer` combines each one with the `date` column to build a
real timestamp — never parses them on their own (a bare `0331` parsed
alone would be misread as a Unix timestamp near 1970-01-01, not 03:31).
Malformed values (out-of-range hour/minute, non-numeric, missing) raise a
`ValueError` naming the file, column, and bad value.

All six columns are required. A file with only the header row is valid and
produces an empty catalog (nothing will ever match). A completely empty
file, or one missing any required column, raises `ValueError`.

### Alternative: full-timestamp catalog schema

`EventMatcher` also accepts a second schema, auto-detected from the CSV's
columns, where `start_time`/`peak_time`/`end_time` are already complete
datetime strings rather than bare HHMM times combined with a shared `date`
column:

| Column | Type | Description |
|---|---|---|
| `start_time` | datetime string | When the flare began |
| `peak_time` | datetime string | When the flare reached maximum intensity |
| `end_time` | datetime string | When the flare ended |
| `class` | string | GOES class, e.g. `"M2.3"` |
| `active_region` | NOAA AR number or empty | Which active region the flare came from |

This is what `flare_indexer.loaders.adapt_goes_catalog` produces (see
"Real-world data" below) and exists specifically so that a flare whose
start, peak, and end fall on *different* calendar dates (e.g. one that
starts just before midnight) is represented correctly — re-encoding into a
single `date` + HHMM would silently corrupt it onto one date.

## Image index CSV

| Column | Required for | Type | Description |
|---|---|---|---|
| `timestamp` | always | datetime string | Timestamp of the solar image (any pandas-parseable format) |
| `active_region` | `target="active_region"` only | NOAA AR number or empty | Which active region the image is centered on |

Any other columns (e.g. `image_path`, `image_id`) are optional and are
preserved in the output.

A file with only the header row and no data rows is valid and produces an
empty result. A completely empty file raises `ValueError`, as does a file
missing a required column for the mode you're using.

**Sequence mode** (`sequence_length > 1`) additionally requires
`timestamp` values to be sorted ascending with no duplicates — both raise
a clear `ValueError` naming the file.

## Window semantics

For a reference time `t` (an image's timestamp, or a sequence's *final*
image timestamp) and a `prediction_window` of `N` hours, `DatasetBuilder`
looks for flares in a window of `[t, t + N hours)` by default. Two
independent settings control exactly which flare timestamp is checked and
where the window's boundaries fall — both configurable on `DatasetBuilder`
and `EventMatcher.query()`, and both default to the original behavior so
existing code is unaffected.

### `event_time`: which flare timestamp is checked

- **`event_time="peak"`** (default) — matches against each flare's
  `peak_time`.
- **`event_time="start"`** — matches against each flare's `start_time`
  instead. This is the convention used by some published full-disk
  labeling pipelines (see "Real-world data" below).

### `interval_mode`: the window's boundary convention

- **`interval_mode="left_closed"`** (default) — `[t, t + N hours)`:
  includes `t` itself, excludes the far endpoint exactly. A flare landing
  at exactly `t + N` hours is **not** counted.
- **`interval_mode="right_closed"`** — `(t, t + N hours]`: excludes `t`
  itself, includes the far endpoint exactly. A flare landing at exactly
  `t + N` hours **is** counted.

There is no lead-time / gap in either mode — the window starts
immediately at (or just after) the reference timestamp.

```python
from flare_indexer import DatasetBuilder, BinaryThresholdStrategy

builder = DatasetBuilder(
    prediction_window=24,
    strategy=BinaryThresholdStrategy(threshold="M"),
    event_time="start",
    interval_mode="right_closed",
)
```

Invalid values for either parameter raise a `ValueError` naming the
parameter.

## Full-disk vs. active-region matching (`target`)

- **`target="full_disk"`** (default) — any flare in the time window
  counts, regardless of which part of the sun it came from.
- **`target="active_region"`** — only flares attributed to the *same*
  active region as the image (or, for a sequence, the region shared by
  every image in it) count. A flare from a different region is ignored
  even if it's stronger; a flare with no recorded active region never
  matches. If the image-side active region is missing, the image (or
  sequence) matches nothing — it never silently falls back to full-disk
  matching.

Active region values are normalized before comparison (`3559`, `3559.0`,
and `"3559"` are all treated as the same region), on both the catalog side
and the image-index side.

## Single-image vs. sequence labeling

- **`sequence_length=1`** (default) — one image timestamp produces one
  label.
- **`sequence_length > 1`** — groups every `sequence_length` consecutive
  images into one sample. `stride` controls how far forward the next
  sequence starts (`stride=1` gives overlapping sequences; `stride=sequence_length`
  gives non-overlapping ones). `cadence_minutes`, if set, skips (not
  raises on) any candidate sequence whose images aren't spaced exactly
  that many minutes apart — useful for series with occasional data gaps.
  The prediction reference time for a sequence is its **final** image's
  timestamp.

## Output schemas

| Mode | Columns |
|---|---|
| Single-image, full-disk (`sequence_length=1, target="full_disk"`) | `timestamp, label` |
| Single-image, active-region (`sequence_length=1, target="active_region"`) | `timestamp, label` |
| Sequence, full-disk (`sequence_length>1, target="full_disk"`) | `sequence_start, sequence_end, timestamps, n_images, label` |
| Sequence, active-region (`sequence_length>1, target="active_region"`) | `sequence_start, sequence_end, timestamps, n_images, active_region, label` |

In sequence mode, any extra image-index column (e.g. `image_path`) is
preserved as a list-valued column holding that field for every image in
the sequence.

## Labeling strategies

- **`BinaryThresholdStrategy(threshold="M")`** — label is `1` if any
  matched flare is at least the given GOES class, else `0`. Defaults to
  M-class.
- **`MaxFlareStrategy()`** — label is the numeric flux of the strongest
  matched flare, or `0.0` if none matched.

Both take the list of `FlareEvent`s returned by `EventMatcher.query()` and
reduce it to a single label.

### Numerical vs. lexical GOES ordering

`MaxFlareStrategy` always compares flares by their `FluxConverter.to_flux()`
numeric value, never by sorting the `goes_class` strings themselves. This
matters once a flare's number exceeds 9: a lexical ("string") sort would
rank `"M9.0"` above `"M10.0"` (comparing the characters `'9'` and `'1'`),
even though `M10.0` is numerically the stronger flare (and, in fact, exactly
as strong as `X1.0`). `flare_indexer` always selects the numerically
strongest flare.

```python
from flare_indexer import BinaryThresholdStrategy

# Count C-class and above as positive instead of the M-class default
strategy = BinaryThresholdStrategy(threshold="C")
```

## Examples

### 1. Full-disk, single-image

```python
import pandas as pd
import flare_indexer as fidx

pd.DataFrame({
    "timestamp": pd.to_datetime(["2024-02-01T00:00:00", "2024-02-10T00:00:00"]),
}).to_csv("image_index.csv", index=False)

pd.DataFrame({
    "date": ["2024-02-01"],
    "start": ["1150"],
    "peak": ["1200"],
    "end": ["1210"],
    "class": ["C3.0"],
    "active_region": [11111],
}).to_csv("flare_catalog.csv", index=False)

builder = fidx.DatasetBuilder(prediction_window=24, strategy=fidx.BinaryThresholdStrategy(threshold="C"))
print(builder.build("image_index.csv", "flare_catalog.csv"))
```
```
   timestamp  label
0 2024-02-01      1   <- catches the C3.0 flare 12 hours later
1 2024-02-10      0   <- no flares in this window
```

### 2. Full-disk, sequence

```python
pd.DataFrame({
    "timestamp": pd.to_datetime([
        "2024-02-01T11:00:00", "2024-02-01T11:30:00", "2024-02-01T12:00:00",
    ]),
}).to_csv("image_index.csv", index=False)

builder = fidx.DatasetBuilder(
    prediction_window=1,
    strategy=fidx.BinaryThresholdStrategy(threshold="C"),
    sequence_length=3,
    stride=1,
    cadence_minutes=30,
)
result = builder.build("image_index.csv", "flare_catalog.csv")
print(result[["sequence_start", "sequence_end", "n_images", "label"]])
```
```
       sequence_start        sequence_end  n_images  label
0 2024-02-01 11:00:00 2024-02-01 12:00:00         3      1
```
The sequence's prediction window is measured from its *last* image
(12:00), which is exactly when the C3.0 flare peaks — so it's included.

### 3. Active-region, single-image

```python
pd.DataFrame({
    "date": ["2024-01-23", "2024-01-23"],
    "start": ["0250", "0250"],
    "peak": ["0300", "0300"],
    "end": ["0310", "0310"],
    "class": ["M5.1", "X9.0"],
    "active_region": [3559, 9999],
}).to_csv("flare_catalog.csv", index=False)

pd.DataFrame({
    "timestamp": pd.to_datetime(["2024-01-23T00:00:00"]),
    "active_region": [3559],
}).to_csv("image_index.csv", index=False)

builder = fidx.DatasetBuilder(
    prediction_window=24, strategy=fidx.MaxFlareStrategy(), target="active_region"
)
print(builder.build("image_index.csv", "flare_catalog.csv"))
```
```
   timestamp     label
0 2024-01-23  0.000051
```
The far stronger X9.0 flare belongs to a *different* active region (9999
vs. the image's 3559) and is correctly excluded — if full-disk matching
had been used instead, the label would be `0.0009` (X9.0's flux), not
`0.000051` (M5.1's).

### 4. Active-region, sequence

```python
pd.DataFrame({
    "timestamp": pd.to_datetime([
        "2024-01-23T02:00:00", "2024-01-23T02:30:00", "2024-01-23T03:00:00",
    ]),
    "active_region": [3559, 3559, 3559],
}).to_csv("image_index.csv", index=False)

builder = fidx.DatasetBuilder(
    prediction_window=1,
    strategy=fidx.BinaryThresholdStrategy(),
    target="active_region",
    sequence_length=3,
    stride=1,
    cadence_minutes=30,
)
result = builder.build("image_index.csv", "flare_catalog.csv")
print(result[["sequence_start", "sequence_end", "n_images", "active_region", "label"]])
```
```
       sequence_start        sequence_end  n_images active_region  label
0 2024-01-23 02:00:00 2024-01-23 03:00:00         3          3559      1
```
Every image in the sequence agrees on active region `3559`, so that
region's M5.1 flare (peaking at 03:00, exactly the sequence's reference
time) is matched; region `9999`'s X9.0 is still excluded. If the images
disagreed on active region, `build()` would raise `ValueError` instead of
silently guessing; if any image's active region were missing, the
sequence would match nothing.

## Real-world data (HMI image lists + integrated GOES catalogs)

Real data doesn't always arrive in the exact shapes `DatasetBuilder` and
`EventMatcher` expect out of the box. `flare_indexer.loaders` bridges two
common real-world shapes into the package's schemas.

### Loading a headerless HMI image list

`build_image_index_from_filenames()` takes an iterable of image paths (or a
path to a headerless single-column text/CSV file listing one per line —
`utf-8-sig` BOMs, if present, are stripped automatically) shaped like:

```
/data/hmi_jpgs_512/2010/12/21/HMI.m2010.12.21_21.00.00.jpg
```

and parses the embedded `HMI.m{YYYY}.{MM}.{DD}_{HH}.{MM}.{SS}.jpg` timestamp
out of the filename (directory depth doesn't matter). The result is sorted
ascending and ready for `DatasetBuilder.build()`.

**Strict vs. tolerant parsing.** A raw directory listing can contain rows
that aren't image paths at all — for example, a plain `os.walk()` over a
lab's data root can pick up unrelated bookkeeping files sitting in the same
tree.

- **`strict=True`** (default) — raise `ValueError` naming the first
  offending row. Matches the original behavior.
- **`strict=False`** — exclude non-matching rows (malformed shape, or a
  shape-valid but calendar-invalid date/time like month `13`) rather than
  raising, and expose exactly what was excluded — nothing is silently
  hidden. Pass `return_report=True` to get back `(DataFrame,
  ImageIndexReport)` instead of just the `DataFrame`; the report has
  `total_input_rows`, `valid_image_rows_before_dedup`,
  `excluded_row_count`/`excluded_rows`, `duplicate_timestamp_count`/
  `duplicate_rows`, `final_row_count`, and `duplicate_policy`.

**Duplicate timestamps.** Real image lists can contain more than one path
for the same nominal timestamp. `duplicate_policy` controls what happens:

- **`"error"`** (default) — raise `ValueError` naming the duplicated
  timestamps.
- **`"first"`** — keep the first-encountered row for each duplicated
  timestamp.
- **`"last"`** — keep the last-encountered row.

```python
from flare_indexer.loaders import build_image_index_from_filenames

image_index, report = build_image_index_from_filenames(
    "totalfiles_jpg_512.csv",
    strict=False,
    duplicate_policy="first",
    return_report=True,
)
print(report.excluded_row_count, report.duplicate_timestamp_count)
```

### Loading an integrated GOES catalog

`adapt_goes_catalog()` takes a CSV shaped like an integrated GOES event
catalog — full datetime `start_time`/`peak_time`/`end_time` columns, a
`goes_class` column, and a nullable-float `noaa_active_region` column
(plus, typically, many unrelated columns, which are ignored) — and adapts
it into the full-timestamp catalog schema `EventMatcher` accepts directly
(see "Alternative: full-timestamp catalog schema" above). `fl_lon`/`fl_lat`
are carried through when present in the source.

```python
from flare_indexer.loaders import adapt_goes_catalog

catalog = adapt_goes_catalog("goes_flares_integrated.csv", output_path="flare_catalog.csv")
```

### Running the full-disk validation / legacy-comparison script

`scripts/validate_full_disk.py` builds a full-disk, M-class labeled dataset
from real HMI + GOES data and, optionally, compares it against a legacy
labeling pipeline's own output by timestamp:

```bash
python scripts/validate_full_disk.py
python scripts/validate_full_disk.py --legacy-labels default
python scripts/validate_full_disk.py --image-list <path-or-url> --event-catalog <path-or-url> --output out.csv
```

`--image-list`, `--event-catalog`, and `--legacy-labels` each accept either
a local path or an `http(s)` URL; URLs are downloaded once into
`scripts/.cache/` (gitignored — nothing downloaded or generated by this
script is tracked in git) and reused on later runs. The benchmark
configuration matches Dr. Pandey's legacy full-disk convention:
`event_time="start"`, `interval_mode="right_closed"` (i.e. `(t, t+24h]`),
M-class binary threshold, `duplicate_policy="first"`.

When `--legacy-labels` is given, the script normalizes both sides by
timestamp (never by row position), handles the legacy no-flare markers
`NF`/blank/`NaN`/`"0"`, reports rows only on one side separately from
actual label mismatches, and prints a best-effort categorization of any
remaining mismatches. On the real dataset (63,649 images, 14,401 events),
this configuration reaches 99.99% agreement with the legacy output, with
every remaining mismatch traced to one specific, deterministic cause: the
legacy script's window is actually `(t, t+23:59:59]` (one second short of
a true 24 hours), so a flare starting at *exactly* `t + 24h` is included by
this package's mathematically exact `right_closed` window but excluded by
legacy's. This is not claimed to generalize to 100% agreement in every
case — it's what this specific benchmark run found, and the script's
`_categorize_mismatches` output shows its reasoning rather than asserting
a conclusion.

## Development

```bash
pip install -e ".[dev]"
pytest
```
