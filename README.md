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

**`FlareClassifier`** — translates flare strength. Flares are named with a
letter + number ("M2.3", "X1.0"). Letters go A < B < C < M < X, and each
step is 10x stronger (a log scale). This class converts that text into a
real number (e.g. "M2.3" -> 2.3e-5) so flares can be compared, and answers
"is this flare strong enough to matter?" (`is_strong`, default threshold =
M-class).

**`EventMatcher`** — searches the catalog by time (and, optionally, by
active region). Given a reference timestamp and a window (e.g. 24 hours),
it returns every flare whose peak fell inside that window. No flares ->
empty list -> "no flare" label.

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
looks for flares whose peak falls in the half-open interval:

```
[t, t + N hours)
```

- Matching uses `peak_time` only — `start` and `end` are not used for the
  window check.
- The interval includes `t` itself and excludes the far endpoint exactly:
  a flare peaking at exactly `t + N` hours is **not** counted.
- There is no lead-time / gap — the window starts immediately at the
  reference timestamp.

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

## Development

```bash
pip install -e ".[dev]"
pytest
```
