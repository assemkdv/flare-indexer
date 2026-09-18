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

"Strategy" here is anything exposing a `label(flares) -> label` method.
`BinaryThresholdStrategy` and `MaxFlareStrategy` (below) are two
ready-to-use ones. Internally, both — and any custom pipeline built from
`DatasetBuildingPipeline` — follow the same three-stage shape: extract
events, reduce them to one value, assign a final label. See "Modular
pipeline architecture" for the composable, inspectable form of this same
computation.

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

This row-based interface (`sequence_length`/`stride`/`cadence_minutes`)
groups images by *position* in the image index. There's also a **time-based**
interface (`start`/`cadence`/`observation_window`/`sliding_window`) that
groups images by an explicit timestamp grid instead — see "Time-based
sequence generation" below. The two are mutually exclusive per
`DatasetBuilder`.

## Output schemas

| Mode | Columns |
|---|---|
| Single-image, full-disk (`sequence_length=1, target="full_disk"`) | `timestamp, label` |
| Single-image, active-region (`sequence_length=1, target="active_region"`) | `timestamp, label` |
| Sequence, full-disk (`sequence_length>1, target="full_disk"`) | `sequence_start, sequence_end, timestamps, n_images, label` |
| Sequence, active-region (`sequence_length>1, target="active_region"`) | `sequence_start, sequence_end, timestamps, n_images, active_region, label` |

In sequence mode, any extra image-index column (e.g. `image_path`) is
preserved as a list-valued column holding that field for every image in
the sequence. Time-based sequences (below) use the identical schemas,
column for column.

## Time-based sequence generation

The row-based interface above groups images by *position*: `sequence_length=3`
always means "the next 3 rows," whatever timestamps happen to be there. The
time-based interface instead builds each sequence from an **explicit
timestamp grid**, which is what makes it possible to safely downsample a
finer-cadence image index (e.g. select an hourly cadence out of 30-minute
source images) and to skip — rather than silently misalign — a sequence
that's missing one of its required images.

```python
from flare_indexer import DatasetBuilder, BinaryThresholdStrategy

builder = DatasetBuilder(
    prediction_window=24,
    strategy=BinaryThresholdStrategy(),
    start="2014-01-01 00:00:00",   # optional -- defaults to the earliest image timestamp
    cadence="1h",                  # spacing between images *within* a sequence
    observation_window="6h",       # span covered by one sequence
    sliding_window="3h",           # how far the next candidate's start moves
)
```

Supplying any of `start`/`cadence`/`observation_window`/`sliding_window`
activates time-based mode. In that mode, `cadence`, `observation_window`,
and `sliding_window` are all required (`start` is the only optional one).
**Explicitly mixing** row-based parameters (`sequence_length`/`stride`/
`cadence_minutes`) with time-based ones raises `ValueError` — pick one
mode per `DatasetBuilder`. All four accept any pandas-compatible duration
(`"12min"`, `"1h"`, `datetime.timedelta(hours=1)`, `pd.Timedelta(...)`) or,
for `start`, any pandas-compatible timestamp.

### The observation window is half-open

`observation_window` is a half-open interval `[sequence_start,
sequence_start + observation_window)`. With `cadence="1h"` and
`observation_window="6h"`, a sequence therefore contains exactly six
images:

```
t, t+1h, t+2h, t+3h, t+4h, t+5h
```

**not** seven — `t + 6h` falls outside the interval's own right endpoint,
exactly like the package's `[t, t+N hours)` prediction-window convention
elsewhere. In general, `number_of_images = observation_window / cadence`,
and both this and `slide_steps = sliding_window / cadence` must come out
to exact positive integers — `observation_window` and `sliding_window`
each must be an exact multiple of `cadence` (e.g. `cadence="1h"` with
`observation_window="90min"` raises `ValueError`, since 90 minutes isn't a
whole number of 1-hour steps).

The next candidate sequence always starts at `sequence_start +
sliding_window`, regardless of how many images the current one ended up
with. Labels are always computed from the sequence's **final** image
timestamp, exactly as in row-based sequence mode.

### `start` semantics

- **Omitted** (default) — the first candidate starts at the image index's
  earliest timestamp.
- **Supplied** — inclusive, and an image must exist at *exactly* that
  timestamp. If it doesn't (whether your requested `start` falls before,
  after, or simply doesn't align with the dataset), `build()` raises
  `ValueError` rather than silently rounding to the nearest available
  image.

Candidate sequence starts are `start + k * sliding_window` for
`k = 0, 1, 2, ...`, continuing for as long as the *final* required image
timestamp of the candidate still falls within the image index's own
timestamp range.

### Source cadence inference and compatible downsampling

`build()` infers the image index's own base cadence as the **mode** (most
common value) of its sorted-unique timestamps' positive consecutive
differences — not the smallest or largest gap, so a dataset with a handful
of missing images or occasional larger gaps still infers the correct
"normal" cadence. The requested `cadence` must be:

1. Greater than or equal to the inferred source cadence (you can't request
   images finer than what's actually there), and
2. An exact integer multiple of it.

So an hourly-cadence image index accepts `cadence="1h"` or `cadence="3h"`,
but rejects `cadence="1min"` (finer than the source) and `cadence="90min"`
(not a whole multiple). A 30-minute-cadence image index accepts
`cadence="1h"` — each generated sequence then picks out only the
on-the-hour rows (`00:00, 01:00, 02:00, ...`), never the `:30` rows in
between; see the downsampling example below. If the image index has fewer
than two unique timestamps, the base cadence can't be inferred at all and
`build()` raises `ValueError`.

### Missing images: skip, don't interpolate

For each candidate start, the exact required timestamps
(`candidate_start + i * cadence` for `i` in `0 .. number_of_images - 1`)
are looked up directly in the image index. If **any** of them is missing,
the **entire candidate is skipped** — never interpolated, never filled by
repeating or substituting a neighboring image. Generation simply continues
to the next candidate start; a later candidate with complete data is still
emitted normally.

### `SequenceBuildReport`

`build(..., return_report=True)` returns `(DataFrame, SequenceBuildReport)`
instead of just the `DataFrame` (the default, `return_report=False`,
preserves the plain-`DataFrame` return exactly). The report carries
`inferred_source_cadence`, `requested_cadence`, `observation_window`,
`sliding_window`, `effective_start`, `total_candidate_sequences`,
`emitted_sequences`, `skipped_sequences`, and samples of
`skipped_sequence_starts`/`missing_timestamps` for diagnosing why a
particular run emitted fewer sequences than expected. `return_report=True`
is only meaningful in time-based mode; passing it in row-based or
single-image mode raises `ValueError`.

```python
result, report = builder.build("image_index.csv", "flare_catalog.csv", return_report=True)
print(report.inferred_source_cadence, report.emitted_sequences, report.skipped_sequences)
```

### Examples

**1. Full-disk, time-based**

```python
import pandas as pd
from flare_indexer import DatasetBuilder, BinaryThresholdStrategy

pd.DataFrame({
    "timestamp": pd.date_range("2024-01-01T00:00:00", periods=10, freq="1h"),
}).to_csv("image_index.csv", index=False)

builder = DatasetBuilder(
    prediction_window=24,
    strategy=BinaryThresholdStrategy(threshold="M"),
    cadence="1h", observation_window="6h", sliding_window="3h",
)
result = builder.build("image_index.csv", "flare_catalog.csv")
print(result[["sequence_start", "sequence_end", "n_images", "label"]])
```

**2. Active-region, time-based**

```python
pd.DataFrame({
    "timestamp": pd.date_range("2024-01-01T08:00:00", periods=6, freq="1h"),
    "active_region": ["3559"] * 6,
}).to_csv("image_index.csv", index=False)

builder = DatasetBuilder(
    prediction_window=1,
    strategy=BinaryThresholdStrategy(),
    target="active_region",
    cadence="1h", observation_window="6h", sliding_window="6h",
)
result = builder.build("image_index.csv", "flare_catalog.csv")
print(result[["sequence_start", "sequence_end", "active_region", "label"]])
```

**3. Downsampling a 30-minute image index to an hourly cadence**

```python
pd.DataFrame({
    "timestamp": pd.date_range("2024-01-01T00:00:00", periods=9, freq="30min"),
}).to_csv("image_index.csv", index=False)

builder = DatasetBuilder(
    prediction_window=24,
    strategy=BinaryThresholdStrategy(),
    cadence="1h", observation_window="4h", sliding_window="4h",
)
result = builder.build("image_index.csv", "flare_catalog.csv")
print(result.iloc[0]["timestamps"])
```
```
[Timestamp('2024-01-01 00:00:00'), Timestamp('2024-01-01 01:00:00'),
 Timestamp('2024-01-01 02:00:00'), Timestamp('2024-01-01 03:00:00')]
```
The `:30`-past-the-hour rows in the source data are never selected — only
the on-the-hour timestamps that match the requested `cadence="1h"` grid.

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

## Modular pipeline architecture

`BinaryThresholdStrategy` and `MaxFlareStrategy` are convenient, but each
bundles "reduce a list of flares to one value" and "turn that value into a
label" into a single opaque `label()` call. `DatasetBuildingPipeline`
splits that into three separate, independently callable, inspectable
stages — the same computation, decomposed rather than replaced:

```
image/sequence reference time
      |
      v
[ EventMatcher ]   -> extract_events()  -> list[FlareEvent]
      |
      v
[ EventReducer ]   -> reduce_events()   -> a single reduced value
      |
      v
[ LabelAssigner ]  -> assign_label()    -> the final label
      |
      v
     label
```

This is a **scikit-learn-inspired modular composition** — small,
swappable components joined by a simple method-per-stage contract — not a
claim of scikit-learn estimator-API compatibility (no `fit`/`transform`,
no `Pipeline` object from `sklearn`).

- **`EventReducer`** — a `typing.Protocol` for anything with a
  `reduce(events: list[FlareEvent])` method. `MaxFluxReducer` (the only
  one provided so far) reduces to the numeric maximum flux, always via
  `FluxConverter` — never by comparing `goes_class` strings lexically.
  Returns `0.0` for an empty event list. Cumulative or other scientific
  reducers can be added later as new classes implementing this same
  `reduce()` method, with no changes to `DatasetBuilder` or
  `DatasetBuildingPipeline` required.
- **`LabelAssigner`** — a `typing.Protocol` for anything with an
  `assign(reduced_value)` method. Two are provided:
  - **`BinaryThresholdLabeler(threshold="M")`** — `1` if the reduced value
    meets or exceeds the threshold's flux, else `0`. `threshold` may be a
    plain letter (`"M"`) or a full GOES class with a number (`"C4.2"`).
  - **`RegressionLabeler()`** — passes the reduced value through
    unchanged, normalizing numeric scalar types (e.g. a numpy `float64`)
    to plain Python `int`/`float`.
- **`DatasetBuildingPipeline(reducer, labeler)`** — coordinates the three
  stages. `reducer` and `labeler` can be the provided classes above or
  **any custom, duck-typed object** exposing the matching method — no
  base class or registration needed. A component missing its required
  method raises a clear `TypeError` at pipeline construction time.

### Full end-to-end pipeline example

```python
from flare_indexer import (
    DatasetBuilder, DatasetBuildingPipeline, MaxFluxReducer, BinaryThresholdLabeler,
)

pipeline = DatasetBuildingPipeline(
    reducer=MaxFluxReducer(),
    labeler=BinaryThresholdLabeler("M"),
)

# A DatasetBuildingPipeline implements label(events), so it's a drop-in
# `strategy=` for DatasetBuilder -- no DatasetBuilder changes needed.
builder = DatasetBuilder(prediction_window=24, strategy=pipeline)
result = builder.build("image_index.csv", "flare_catalog.csv")
```

### Stopping after event extraction

```python
from flare_indexer import EventMatcher
import pandas as pd

matcher = EventMatcher("flare_catalog.csv")
events = pipeline.extract_events(
    matcher, pd.Timestamp("2024-02-01"), prediction_window=24,
)
# Inspect the matched FlareEvents directly -- nothing has been reduced
# or labeled yet.
for event in events:
    print(event.goes_class, event.peak_time)
```

### Stopping after reduction

```python
reduced = pipeline.reduce_events(events)
print(reduced)  # the numeric maximum flux, e.g. 2.3e-05
```

### Running only label assignment on a previously reduced value

```python
label = pipeline.assign_label(reduced)
```

### All three stages at once, with every intermediate value preserved

```python
result = pipeline.run_one(matcher, pd.Timestamp("2024-02-01"), prediction_window=24)
result.events          # list[FlareEvent]
result.reduced_value   # the numeric maximum flux
result.label            # the final label
```

`extract_events`, `reduce_events`, and `assign_label` never depend on one
another having been called first, and `DatasetBuildingPipeline` stores no
per-call intermediate state. Reuse or concurrent sharing is safe when the
supplied reducer and labeler are themselves stateless or thread-safe --
the pipeline does not make a stateful or non-thread-safe reducer/labeler
safe to share on its own.

### Custom reducer

A custom reducer and the labeler it's paired with must agree on the
meaning and units of the intermediate value -- here, a raw flare *count*,
not a physical flux, so it's paired with a labeler that thresholds on a
count rather than `BinaryThresholdLabeler` (which expects a GOES flux):

```python
class CumulativeCountReducer:
    """Not provided by the package -- an example of a fully custom reducer."""
    def reduce(self, events):
        return len(events)

class MinimumCountLabeler:
    """Not provided by the package -- pairs with CumulativeCountReducer's count output."""
    def __init__(self, minimum=1):
        self.minimum = minimum

    def assign(self, count):
        return int(count >= self.minimum)

pipeline = DatasetBuildingPipeline(
    reducer=CumulativeCountReducer(),
    labeler=MinimumCountLabeler(minimum=2),
)
```

### Custom labeler

```python
class ThreeLevelLabeler:
    """Not provided by the package -- an example of a fully custom labeler."""
    def assign(self, reduced_value):
        if reduced_value == 0.0:
            return "none"
        return "major" if reduced_value >= 1e-5 else "minor"

pipeline = DatasetBuildingPipeline(reducer=MaxFluxReducer(), labeler=ThreeLevelLabeler())
```

### Backward compatibility

`BinaryThresholdStrategy` and `MaxFlareStrategy` still work exactly as
before — internally, they now delegate to this same modular machinery
(`MaxFluxReducer` + `BinaryThresholdLabeler`, and `MaxFluxReducer` +
`RegressionLabeler`, respectively), so their numeric output is unchanged:

```python
from flare_indexer import BinaryThresholdStrategy, MaxFlareStrategy

# Still works exactly as before.
builder = DatasetBuilder(prediction_window=24, strategy=BinaryThresholdStrategy(threshold="M"))
builder = DatasetBuilder(prediction_window=24, strategy=MaxFlareStrategy())
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
