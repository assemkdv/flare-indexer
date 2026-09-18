from pathlib import Path

import pandas as pd

from flare_indexer.builder import DatasetBuilder
from flare_indexer.events import EventMatcher
from flare_indexer.strategies import BinaryThresholdStrategy, MaxFlareStrategy

FIXTURES_DIR = Path(__file__).parent / "fixtures"
CATALOG_PATH = FIXTURES_DIR / "catalog.csv"
IMAGE_INDEX_PATH = FIXTURES_DIR / "image_index.csv"

# tests/fixtures/catalog.csv has four flares:
#   C3.0 peak 2024-02-01T12:00:00  (flux=3e-6)
#   X1.0 peak 2024-02-02T00:00:00  (flux=1e-4)  <- exactly on a 24h window end
#   C1.0 peak 2024-02-05T06:00:00  (flux=1e-6)
#   M4.0 peak 2024-02-05T18:00:00  (flux=4e-5)
#
# tests/fixtures/image_index.csv has three timestamps, each with a
# prediction_window of 24 hours:
#   2024-02-01T00:00:00 -> window [2024-02-01T00:00, 2024-02-02T00:00)
#       Only C3.0 is inside. X1.0 peaks exactly at the window end and must
#       be excluded by the half-open interval.
#   2024-02-05T00:00:00 -> window [2024-02-05T00:00, 2024-02-06T00:00)
#       Both C1.0 and M4.0 are inside; M4.0 qualifies the binary threshold.
#   2024-02-10T00:00:00 -> window [2024-02-10T00:00, 2024-02-11T00:00)
#       No flares fall inside; the window is empty.


def test_end_to_end_binary_threshold_strategy():
    builder = DatasetBuilder(prediction_window=24, strategy=BinaryThresholdStrategy())
    result = builder.build(IMAGE_INDEX_PATH, CATALOG_PATH)

    assert list(result.columns) == ["timestamp", "label"]
    assert result["timestamp"].tolist() == [
        pd.Timestamp("2024-02-01T00:00:00"),
        pd.Timestamp("2024-02-05T00:00:00"),
        pd.Timestamp("2024-02-10T00:00:00"),
    ]
    assert result["label"].tolist() == [0, 1, 0]


def test_end_to_end_max_flare_strategy():
    builder = DatasetBuilder(prediction_window=24, strategy=MaxFlareStrategy())
    result = builder.build(IMAGE_INDEX_PATH, CATALOG_PATH)

    # Row 1: max is C3.0's 3e-6 — X1.0's 1e-4 must be excluded since its
    # peak lands exactly on the window boundary. If the boundary were
    # mistakenly inclusive, this would be 1e-4 instead.
    # Row 2: max of C1.0 (1e-6) and M4.0 (4e-5) is 4e-5.
    # Row 3: no flares in the window -> 0.0.
    assert result["label"].tolist() == [3e-6, 4e-5, 0.0]


def test_end_to_end_sequences_against_real_fixture_catalog(tmp_path):
    # Uses the same real-format (date + HHMM) tests/fixtures/catalog.csv as
    # the tests above, but drives it through sequence labeling rather than
    # single-image labeling.
    index_path = tmp_path / "sequence_index.csv"
    pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2024-02-01T09:00:00",
            "2024-02-01T10:00:00",
            "2024-02-01T11:00:00",
            "2024-02-01T12:00:00",
        ]),
    }).to_csv(index_path, index=False)

    builder = DatasetBuilder(
        prediction_window=1,
        strategy=BinaryThresholdStrategy(),
        sequence_length=2,
        stride=1,
        cadence_minutes=60,
    )
    result = builder.build(index_path, CATALOG_PATH)

    assert list(result.columns) == ["sequence_start", "sequence_end", "timestamps", "n_images", "label"]
    assert result["sequence_end"].tolist() == [
        pd.Timestamp("2024-02-01T10:00:00"),
        pd.Timestamp("2024-02-01T11:00:00"),
        pd.Timestamp("2024-02-01T12:00:00"),
    ]
    # Only the last sequence's 1-hour window ([12:00, 13:00)) catches a
    # flare (C3.0, peak exactly at 12:00) -- and it's C-class, so the
    # M-threshold binary label is still 0 for all three sequences.
    assert result["label"].tolist() == [0, 0, 0]

    max_result = DatasetBuilder(
        prediction_window=1,
        strategy=MaxFlareStrategy(),
        sequence_length=2,
        stride=1,
        cadence_minutes=60,
    ).build(index_path, CATALOG_PATH)
    assert max_result["label"].tolist() == [0.0, 0.0, 3e-6]


def test_end_to_end_active_region_against_real_fixture_catalog(tmp_path):
    # tests/fixtures/catalog.csv has two active regions: 11111 (C3.0, X1.0
    # on 2024-02-01/02) and 22222 (C1.0, M4.0 on 2024-02-05). Using
    # threshold="C" so C-class flares count as qualifying.
    index_path = tmp_path / "ar_index.csv"
    pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2024-02-01T00:00:00",  # region 11111, correct -> catches C3.0
            "2024-02-05T00:00:00",  # region 11111, wrong that day -> nothing
            "2024-02-05T00:00:00",  # region 22222, correct -> catches C1.0/M4.0
        ]),
        "active_region": ["11111", "11111", "22222"],
    }).to_csv(index_path, index=False)

    builder = DatasetBuilder(
        prediction_window=24,
        strategy=BinaryThresholdStrategy(threshold="C"),
        target="active_region",
    )
    result = builder.build(index_path, CATALOG_PATH)

    assert list(result.columns) == ["timestamp", "label"]
    assert result["label"].tolist() == [1, 0, 1]


def test_end_to_end_active_region_sequences_against_real_fixture_catalog(tmp_path):
    # tests/fixtures/catalog.csv has region 11111 (C3.0 peak 2024-02-01T12:00,
    # X1.0 peak 2024-02-02T00:00) and region 22222 (C1.0/M4.0 on 2024-02-05).
    # Using threshold="C" so C-class flares count as qualifying.
    index_path = tmp_path / "ar_sequence_index.csv"
    pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2024-02-01T11:00:00",
            "2024-02-01T11:30:00",
            "2024-02-01T12:00:00",
            "2024-02-05T09:00:00",
            "2024-02-05T09:30:00",
            "2024-02-05T10:00:00",
        ]),
        "active_region": ["11111", "11111", "11111", "22222", "22222", "22222"],
    }).to_csv(index_path, index=False)

    builder = DatasetBuilder(
        prediction_window=1,
        strategy=BinaryThresholdStrategy(threshold="C"),
        target="active_region",
        sequence_length=3,
        stride=3,
        cadence_minutes=30,
    )
    result = builder.build(index_path, CATALOG_PATH)

    assert list(result.columns) == [
        "sequence_start", "sequence_end", "timestamps", "n_images", "active_region", "label"
    ]
    assert result["sequence_end"].tolist() == [
        pd.Timestamp("2024-02-01T12:00:00"),
        pd.Timestamp("2024-02-05T10:00:00"),
    ]
    # First sequence ends exactly at region 11111's C3.0 peak -> label 1.
    # Second sequence's 1-hour window from 10:00 catches no region-22222
    # flare (C1.0 peaked at 06:00, M4.0 peaks later at 18:00) -> label 0.
    assert result["label"].tolist() == [1, 0]
    assert result["active_region"].tolist() == ["11111", "22222"]


def test_end_to_end_smoke_test_against_real_catalog(tmp_path):
    # Not the main correctness test — labels here aren't hardcoded since
    # the real development catalog can change over time. This just
    # confirms the full pipeline runs cleanly against actual data, now that
    # data/solar_events.csv has been converted to the real date+HHMM
    # scraper format.
    repo_root = Path(__file__).parent.parent
    real_catalog = repo_root / "data" / "solar_events.csv"

    # EventMatcher must be able to load the converted file at all (this
    # also validates every start/peak/end value is a real 4-digit HHMM
    # time, since _load_catalog raises ValueError on anything else).
    matcher = EventMatcher(real_catalog)
    assert len(matcher._catalog) > 0

    # Regression check: none of the converted peak_times were misparsed
    # into the ~1970-01-01 Unix-epoch bug this file's format fix addressed.
    assert (matcher._catalog["peak_time"].dt.year > 2000).all()

    index_path = tmp_path / "smoke_index.csv"
    pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-06-06T00:00:00", "2026-06-10T00:00:00"]),
    }).to_csv(index_path, index=False)

    builder = DatasetBuilder(prediction_window=24, strategy=BinaryThresholdStrategy())
    result = builder.build(index_path, real_catalog)

    assert list(result.columns) == ["timestamp", "label"]
    assert len(result) == 2
    assert set(result["label"].tolist()) <= {0, 1}
