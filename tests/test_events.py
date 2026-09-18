import pandas as pd
import pytest
from flare_indexer.events import EventMatcher


def test_query_normalizes_missing_active_region_to_none():
    # pandas represents a missing active_region cell as float NaN once the
    # column contains any missing values — this must come out as None.
    matcher = EventMatcher.__new__(EventMatcher)
    matcher._catalog = pd.DataFrame({
        "peak_time": pd.to_datetime(["2024-01-01 00:30"]),
        "start_time": pd.to_datetime(["2024-01-01 00:20"]),
        "goes_class": ["M2.3"],
        "active_region": [float("nan")],
    })

    results = matcher.query(
        image_time=pd.Timestamp("2024-01-01 00:00"),
        prediction_window_hours=1,
    )

    assert results[0].active_region is None


def test_query_preserves_non_empty_active_region():
    # A present active_region value in a column that also has missing cells
    # arrives from pandas as a float (e.g. 4456.0) — it must be preserved
    # as a proper string ("4456"), not dropped or left as a float.
    matcher = EventMatcher.__new__(EventMatcher)
    matcher._catalog = pd.DataFrame({
        "peak_time": pd.to_datetime(["2024-01-01 00:30"]),
        "start_time": pd.to_datetime(["2024-01-01 00:20"]),
        "goes_class": ["M2.3"],
        "active_region": [4456.0],
    })

    results = matcher.query(
        image_time=pd.Timestamp("2024-01-01 00:00"),
        prediction_window_hours=1,
    )

    assert results[0].active_region == "4456"
    assert isinstance(results[0].active_region, str)


def test_load_catalog_does_not_print(tmp_path, capsys):
    catalog_path = tmp_path / "catalog.csv"
    catalog_path.write_text(
        "date,start,peak,end,class,active_region\n"
        "2024-01-01,0020,0030,0040,M2.3,4456\n"
    )

    EventMatcher(catalog_path)

    captured = capsys.readouterr()
    assert captured.out == ""


def test_load_catalog_parses_hhmm_with_date(tmp_path):
    # The exact scraper example row: NOAA-style bare HHMM times combined
    # with the date column, not a full ISO timestamp per field.
    catalog_path = tmp_path / "catalog.csv"
    catalog_path.write_text(
        "date,start,peak,end,class,active_region\n"
        "2024-01-23,0309,0331,0338,M5.1,3559\n"
    )

    matcher = EventMatcher(catalog_path)

    assert matcher._catalog.loc[0, "peak_time"] == pd.Timestamp("2024-01-23 03:31:00")
    assert matcher._catalog.loc[0, "start_time"] == pd.Timestamp("2024-01-23 03:09:00")


def test_load_catalog_never_reproduces_1970_epoch_bug(tmp_path):
    # Regression test: pd.to_datetime() called directly on a bare HHMM value
    # like 331 used to be misread as a Unix timestamp near 1970-01-01.
    catalog_path = tmp_path / "catalog.csv"
    catalog_path.write_text(
        "date,start,peak,end,class,active_region\n"
        "2024-01-23,0309,0331,0338,M5.1,3559\n"
    )

    matcher = EventMatcher(catalog_path)

    assert matcher._catalog.loc[0, "peak_time"].year == 2024


def test_load_catalog_parses_midnight(tmp_path):
    catalog_path = tmp_path / "catalog.csv"
    catalog_path.write_text(
        "date,start,peak,end,class,active_region\n"
        "2024-01-23,0000,0000,0005,C1.0,3559\n"
    )

    matcher = EventMatcher(catalog_path)

    assert matcher._catalog.loc[0, "peak_time"] == pd.Timestamp("2024-01-23 00:00:00")


def test_load_catalog_rejects_invalid_hour(tmp_path):
    catalog_path = tmp_path / "catalog.csv"
    catalog_path.write_text(
        "date,start,peak,end,class,active_region\n"
        "2024-01-23,2460,0331,0338,M5.1,3559\n"
    )

    with pytest.raises(ValueError) as excinfo:
        EventMatcher(catalog_path)

    message = str(excinfo.value)
    assert "2460" in message
    assert str(catalog_path) in message


def test_load_catalog_rejects_invalid_minute(tmp_path):
    catalog_path = tmp_path / "catalog.csv"
    catalog_path.write_text(
        "date,start,peak,end,class,active_region\n"
        "2024-01-23,0309,1267,0338,M5.1,3559\n"
    )

    with pytest.raises(ValueError) as excinfo:
        EventMatcher(catalog_path)

    assert "1267" in str(excinfo.value)


def test_load_catalog_rejects_non_numeric_time(tmp_path):
    catalog_path = tmp_path / "catalog.csv"
    catalog_path.write_text(
        "date,start,peak,end,class,active_region\n"
        "2024-01-23,0309,abc,0338,M5.1,3559\n"
    )

    with pytest.raises(ValueError) as excinfo:
        EventMatcher(catalog_path)

    assert "abc" in str(excinfo.value)


def test_query_active_region_same_region_flare_included():
    matcher = EventMatcher.__new__(EventMatcher)
    matcher._catalog = pd.DataFrame({
        "peak_time": pd.to_datetime(["2024-01-01 00:30"]),
        "start_time": pd.to_datetime(["2024-01-01 00:20"]),
        "goes_class": ["M2.3"],
        "active_region": ["3559"],
    })

    results = matcher.query(
        image_time=pd.Timestamp("2024-01-01 00:00"),
        prediction_window_hours=1,
        active_region="3559",
    )

    assert len(results) == 1
    assert results[0].goes_class == "M2.3"


def test_query_active_region_different_region_flare_ignored():
    matcher = EventMatcher.__new__(EventMatcher)
    matcher._catalog = pd.DataFrame({
        "peak_time": pd.to_datetime(["2024-01-01 00:30"]),
        "start_time": pd.to_datetime(["2024-01-01 00:20"]),
        "goes_class": ["M2.3"],
        "active_region": ["9999"],
    })

    results = matcher.query(
        image_time=pd.Timestamp("2024-01-01 00:00"),
        prediction_window_hours=1,
        active_region="3559",
    )

    assert results == []


def test_query_active_region_stronger_flare_from_other_region_ignored():
    matcher = EventMatcher.__new__(EventMatcher)
    matcher._catalog = pd.DataFrame({
        "peak_time": pd.to_datetime(["2024-01-01 00:30", "2024-01-01 00:31"]),
        "start_time": pd.to_datetime(["2024-01-01 00:20", "2024-01-01 00:21"]),
        "goes_class": ["M5.1", "X9.0"],
        "active_region": ["3559", "9999"],
    })

    results = matcher.query(
        image_time=pd.Timestamp("2024-01-01 00:00"),
        prediction_window_hours=1,
        active_region="3559",
    )

    assert len(results) == 1
    assert results[0].goes_class == "M5.1"


def test_query_active_region_missing_catalog_active_region_does_not_match():
    matcher = EventMatcher.__new__(EventMatcher)
    matcher._catalog = pd.DataFrame({
        "peak_time": pd.to_datetime(["2024-01-01 00:30"]),
        "start_time": pd.to_datetime(["2024-01-01 00:20"]),
        "goes_class": ["X1.0"],
        "active_region": [float("nan")],
    })

    results = matcher.query(
        image_time=pd.Timestamp("2024-01-01 00:00"),
        prediction_window_hours=1,
        active_region="3559",
    )

    assert results == []


def test_load_catalog_rejects_missing_time_value(tmp_path):
    catalog_path = tmp_path / "catalog.csv"
    catalog_path.write_text(
        "date,start,peak,end,class,active_region\n"
        "2024-01-23,0309,,0338,M5.1,3559\n"
    )

    with pytest.raises(ValueError) as excinfo:
        EventMatcher(catalog_path)

    assert "missing" in str(excinfo.value).lower()


def test_load_catalog_missing_one_required_column(tmp_path):
    catalog_path = tmp_path / "catalog.csv"
    catalog_path.write_text(
        "date,start,peak,end,class\n"
        "2024-01-01,2024-01-01T00:20:00,2024-01-01T00:30:00,2024-01-01T00:40:00,M2.3\n"
    )

    with pytest.raises(ValueError) as excinfo:
        EventMatcher(catalog_path)

    message = str(excinfo.value)
    assert "active_region" in message
    assert str(catalog_path) in message


def test_load_catalog_missing_multiple_required_columns(tmp_path):
    catalog_path = tmp_path / "catalog.csv"
    catalog_path.write_text(
        "date,peak,class\n"
        "2024-01-01,2024-01-01T00:30:00,M2.3\n"
    )

    with pytest.raises(ValueError) as excinfo:
        EventMatcher(catalog_path)

    message = str(excinfo.value)
    assert "start" in message
    assert "end" in message
    assert "active_region" in message


def test_load_catalog_completely_empty_file_raises(tmp_path):
    catalog_path = tmp_path / "catalog.csv"
    catalog_path.write_text("")

    with pytest.raises(ValueError) as excinfo:
        EventMatcher(catalog_path)

    assert "empty" in str(excinfo.value).lower()


def test_load_catalog_header_only_is_valid_and_empty(tmp_path):
    catalog_path = tmp_path / "catalog.csv"
    catalog_path.write_text("date,start,peak,end,class,active_region\n")

    matcher = EventMatcher(catalog_path)
    results = matcher.query(pd.Timestamp("2024-01-01"), prediction_window_hours=24)

    assert results == []


# ── Full-timestamp catalog schema (adapt_goes_catalog-shaped) ──────────


def test_load_catalog_full_timestamp_schema_is_recognized(tmp_path):
    catalog_path = tmp_path / "catalog.csv"
    catalog_path.write_text(
        "start_time,peak_time,end_time,class,active_region\n"
        "2024-01-23 03:09:00,2024-01-23 03:31:00,2024-01-23 03:38:00,M5.1,3559\n"
    )

    matcher = EventMatcher(catalog_path)

    assert matcher._catalog.loc[0, "peak_time"] == pd.Timestamp("2024-01-23 03:31:00")
    assert matcher._catalog.loc[0, "start_time"] == pd.Timestamp("2024-01-23 03:09:00")
    assert matcher._catalog.loc[0, "end_time"] == pd.Timestamp("2024-01-23 03:38:00")
    assert matcher._catalog.loc[0, "goes_class"] == "M5.1"


def test_load_catalog_full_timestamp_schema_preserves_dates_crossing_midnight(tmp_path):
    # A flare that starts just before midnight and peaks/ends just after it
    # must keep three DIFFERENT calendar dates -- not get flattened onto a
    # single date the way the old date+HHMM schema would require.
    catalog_path = tmp_path / "catalog.csv"
    catalog_path.write_text(
        "start_time,peak_time,end_time,class,active_region\n"
        "2020-01-01 23:58:00,2020-01-02 00:04:00,2020-01-02 00:10:00,C1.0,4456\n"
    )

    matcher = EventMatcher(catalog_path)
    row = matcher._catalog.iloc[0]

    assert row["start_time"] == pd.Timestamp("2020-01-01 23:58:00")
    assert row["peak_time"] == pd.Timestamp("2020-01-02 00:04:00")
    assert row["end_time"] == pd.Timestamp("2020-01-02 00:10:00")
    assert row["start_time"].date() == pd.Timestamp("2020-01-01").date()
    assert row["peak_time"].date() == pd.Timestamp("2020-01-02").date()
    assert row["end_time"].date() == pd.Timestamp("2020-01-02").date()

    results = matcher.query(
        pd.Timestamp("2020-01-01 23:00:00"), prediction_window_hours=2, event_time="start"
    )
    assert len(results) == 1
    assert results[0].start_time == pd.Timestamp("2020-01-01 23:58:00")
    assert results[0].peak_time == pd.Timestamp("2020-01-02 00:04:00")
    assert results[0].end_time == pd.Timestamp("2020-01-02 00:10:00")


def test_full_timestamp_schema_takes_priority_error_message_when_ambiguous(tmp_path):
    # Missing "active_region" with otherwise full-timestamp-shaped columns
    # should report against the full-timestamp schema, not the legacy one.
    catalog_path = tmp_path / "catalog.csv"
    catalog_path.write_text(
        "start_time,peak_time,end_time,class\n"
        "2024-01-23 03:09:00,2024-01-23 03:31:00,2024-01-23 03:38:00,M5.1\n"
    )

    with pytest.raises(ValueError) as excinfo:
        EventMatcher(catalog_path)

    assert "active_region" in str(excinfo.value)


# ── event_time (start vs. peak matching) ────────────────────────────────


def test_event_time_defaults_to_peak():
    matcher = EventMatcher.__new__(EventMatcher)
    matcher._catalog = pd.DataFrame({
        "peak_time": pd.to_datetime(["2024-01-01 01:00"]),
        "start_time": pd.to_datetime(["2024-01-01 00:30"]),
        "goes_class": ["M2.3"],
        "active_region": [None],
    })

    # Window [00:00, 00:45) catches the flare by start_time (00:30) but not
    # by peak_time (01:00) -- default (peak) must exclude it.
    results = matcher.query(pd.Timestamp("2024-01-01 00:00"), prediction_window_hours=0.75)
    assert results == []


def test_event_time_start_vs_peak_produce_different_results():
    matcher = EventMatcher.__new__(EventMatcher)
    matcher._catalog = pd.DataFrame({
        "peak_time": pd.to_datetime(["2024-01-01 01:00"]),
        "start_time": pd.to_datetime(["2024-01-01 00:30"]),
        "goes_class": ["M2.3"],
        "active_region": [None],
    })
    window_start = pd.Timestamp("2024-01-01 00:00")

    peak_results = matcher.query(window_start, prediction_window_hours=0.75, event_time="peak")
    start_results = matcher.query(window_start, prediction_window_hours=0.75, event_time="start")

    assert peak_results == []
    assert len(start_results) == 1
    assert start_results[0].goes_class == "M2.3"


def test_event_time_invalid_value_raises():
    matcher = EventMatcher.__new__(EventMatcher)
    matcher._catalog = pd.DataFrame({
        "peak_time": pd.to_datetime(["2024-01-01 01:00"]),
        "start_time": pd.to_datetime(["2024-01-01 00:30"]),
        "goes_class": ["M2.3"],
        "active_region": [None],
    })

    with pytest.raises(ValueError) as excinfo:
        matcher.query(pd.Timestamp("2024-01-01 00:00"), prediction_window_hours=1, event_time="end")

    assert "event_time" in str(excinfo.value)


# ── interval_mode (window boundary convention) ──────────────────────────


def _boundary_matcher():
    matcher = EventMatcher.__new__(EventMatcher)
    matcher._catalog = pd.DataFrame({
        "peak_time": pd.to_datetime([
            "2024-01-01 00:00:00",  # exactly at t
            "2024-01-01 00:00:01",  # one second after t
            "2024-01-01 00:59:59",  # one second before t + window
            "2024-01-01 01:00:00",  # exactly at t + window
        ]),
        "start_time": pd.to_datetime([
            "2024-01-01 00:00:00",
            "2024-01-01 00:00:01",
            "2024-01-01 00:59:59",
            "2024-01-01 01:00:00",
        ]),
        "goes_class": ["A1.0", "A1.0", "A1.0", "A1.0"],
        "active_region": [None, None, None, None],
    })
    return matcher


def test_interval_mode_left_closed_default_boundaries():
    matcher = _boundary_matcher()
    window_start = pd.Timestamp("2024-01-01 00:00:00")

    results = matcher.query(window_start, prediction_window_hours=1)  # default left_closed
    peak_times = sorted(r.peak_time for r in results)

    # [t, t+1h): includes t itself and t+1h-1s, excludes t+1h exactly.
    assert peak_times == [
        pd.Timestamp("2024-01-01 00:00:00"),
        pd.Timestamp("2024-01-01 00:00:01"),
        pd.Timestamp("2024-01-01 00:59:59"),
    ]


def test_interval_mode_right_closed_boundaries():
    matcher = _boundary_matcher()
    window_start = pd.Timestamp("2024-01-01 00:00:00")

    results = matcher.query(window_start, prediction_window_hours=1, interval_mode="right_closed")
    peak_times = sorted(r.peak_time for r in results)

    # (t, t+1h]: excludes t itself, includes t+1h exactly.
    assert peak_times == [
        pd.Timestamp("2024-01-01 00:00:01"),
        pd.Timestamp("2024-01-01 00:59:59"),
        pd.Timestamp("2024-01-01 01:00:00"),
    ]


def test_interval_mode_invalid_value_raises():
    matcher = _boundary_matcher()

    with pytest.raises(ValueError) as excinfo:
        matcher.query(
            pd.Timestamp("2024-01-01 00:00:00"), prediction_window_hours=1, interval_mode="both_closed"
        )

    assert "interval_mode" in str(excinfo.value)
