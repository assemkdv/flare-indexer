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
