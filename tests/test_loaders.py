import pandas as pd
import pytest

from flare_indexer.loaders import ImageIndexReport, adapt_goes_catalog, build_image_index_from_filenames

# ── build_image_index_from_filenames ────────────────────────────────────


def test_build_image_index_from_filename_list():
    paths = [
        "/data/hmi_jpgs_512/2010/12/21/HMI.m2010.12.21_21.00.00.jpg",
        "/data/hmi_jpgs_512/2010/12/21/HMI.m2010.12.21_20.00.00.jpg",
        "/data/hmi_jpgs_512/2010/12/22/HMI.m2010.12.22_00.30.15.jpg",
    ]

    result = build_image_index_from_filenames(paths)

    assert list(result.columns) == ["timestamp", "image_path"]
    assert pd.api.types.is_datetime64_any_dtype(result["timestamp"])
    # Sorted ascending, required for DatasetBuilder sequence mode.
    assert result["timestamp"].is_monotonic_increasing
    assert result["timestamp"].tolist() == [
        pd.Timestamp("2010-12-21 20:00:00"),
        pd.Timestamp("2010-12-21 21:00:00"),
        pd.Timestamp("2010-12-22 00:30:15"),
    ]
    # Original path preserved, associated with the right row after sorting.
    assert result.loc[result["timestamp"] == pd.Timestamp("2010-12-22 00:30:15"), "image_path"].iloc[0] == (
        "/data/hmi_jpgs_512/2010/12/22/HMI.m2010.12.22_00.30.15.jpg"
    )


def test_build_image_index_varying_directory_depth():
    # Directory depth must not matter -- only the basename is parsed.
    paths = [
        "HMI.m2010.01.01_00.00.00.jpg",
        "/a/b/c/d/e/HMI.m2010.01.02_00.00.00.jpg",
    ]
    result = build_image_index_from_filenames(paths)
    assert len(result) == 2


def test_build_image_index_malformed_filename_raises():
    paths = [
        "/data/hmi_jpgs_512/2010/12/21/HMI.m2010.12.21_21.00.00.jpg",
        "/data/hmi_jpgs_512/not_a_valid_filename.jpg",
    ]

    with pytest.raises(ValueError) as excinfo:
        build_image_index_from_filenames(paths)

    assert "not_a_valid_filename.jpg" in str(excinfo.value)


def test_build_image_index_from_headerless_paths_file(tmp_path):
    paths_file = tmp_path / "paths.txt"
    paths_file.write_text(
        "/data/HMI.m2010.12.21_21.00.00.jpg\n"
        "/data/HMI.m2010.12.21_20.00.00.jpg\n"
    )

    result = build_image_index_from_filenames(paths_file)

    assert len(result) == 2
    assert result["timestamp"].is_monotonic_increasing


def test_build_image_index_strips_utf8_bom(tmp_path):
    paths_file = tmp_path / "paths.txt"
    # Write the raw bytes so the BOM lands at the very start of the file,
    # attached to the first path -- exactly how the lab's actual
    # totalfiles_jpg_512.csv is encoded.
    paths_file.write_bytes(
        "﻿/data/HMI.m2010.12.21_21.00.00.jpg\n/data/HMI.m2010.12.21_20.00.00.jpg\n".encode("utf-8")
    )

    result = build_image_index_from_filenames(paths_file)

    assert len(result) == 2
    # The BOM must not leak into the basename match or the preserved path.
    assert result["image_path"].iloc[0] == "/data/HMI.m2010.12.21_20.00.00.jpg"
    assert result["image_path"].iloc[1] == "/data/HMI.m2010.12.21_21.00.00.jpg"


def test_build_image_index_unsorted_paths_become_sorted():
    paths = [
        "HMI.m2010.12.21_23.00.00.jpg",
        "HMI.m2010.12.21_00.00.00.jpg",
        "HMI.m2010.12.21_12.00.00.jpg",
    ]

    result = build_image_index_from_filenames(paths)

    assert result["timestamp"].tolist() == [
        pd.Timestamp("2010-12-21 00:00:00"),
        pd.Timestamp("2010-12-21 12:00:00"),
        pd.Timestamp("2010-12-21 23:00:00"),
    ]


def test_build_image_index_writes_output_csv(tmp_path):
    paths = ["/data/HMI.m2010.12.21_21.00.00.jpg"]
    output_path = tmp_path / "image_index.csv"

    result = build_image_index_from_filenames(paths, output_path=output_path)

    assert output_path.exists()
    on_disk = pd.read_csv(output_path)
    assert list(on_disk.columns) == ["timestamp", "image_path"]
    assert len(on_disk) == 1
    assert result["image_path"].iloc[0] == paths[0]


# ── strict vs. tolerant malformed-row handling ──────────────────────────


def test_build_image_index_strict_invalid_date_in_correctly_shaped_filename_raises():
    # Shape matches (2-digit groups) but month=13 is not a real calendar month.
    paths = ["HMI.m2010.13.21_21.00.00.jpg"]

    with pytest.raises(ValueError) as excinfo:
        build_image_index_from_filenames(paths)

    assert "HMI.m2010.13.21_21.00.00.jpg" in str(excinfo.value)


def test_build_image_index_tolerant_excludes_and_reports_malformed_rows():
    paths = [
        "/data/HMI.m2010.12.21_21.00.00.jpg",
        "/data/labels_store/C_full_dataset_cleaned_9_hours.csv",  # real-world garbage row
        "HMI.m2010.13.21_21.00.00.jpg",  # shape-valid, invalid month
        "/data/HMI.m2010.12.21_20.00.00.jpg",
    ]

    df, report = build_image_index_from_filenames(paths, strict=False, return_report=True)

    assert len(df) == 2
    assert isinstance(report, ImageIndexReport)
    assert report.total_input_rows == 4
    assert report.valid_image_rows_before_dedup == 2
    assert report.excluded_row_count == 2
    assert "/data/labels_store/C_full_dataset_cleaned_9_hours.csv" in report.excluded_rows
    assert "HMI.m2010.13.21_21.00.00.jpg" in report.excluded_rows
    assert report.final_row_count == 2


def test_build_image_index_strict_default_true_matches_original_behavior():
    paths = ["/data/not_an_image.csv"]

    with pytest.raises(ValueError):
        build_image_index_from_filenames(paths)  # strict not passed -> default True


# ── duplicate_policy ─────────────────────────────────────────────────────


_DUPLICATE_PATHS = [
    "HMI.m2010.12.21_20.00.00.jpg",
    "/other/dir/HMI.m2010.12.21_20.00.00.jpg",  # same timestamp, different path
    "HMI.m2010.12.21_21.00.00.jpg",
]


def test_build_image_index_duplicate_policy_error_raises():
    with pytest.raises(ValueError) as excinfo:
        build_image_index_from_filenames(_DUPLICATE_PATHS)  # default duplicate_policy="error"

    assert "duplicate" in str(excinfo.value).lower()


def test_build_image_index_duplicate_policy_first_keeps_first_encountered():
    result = build_image_index_from_filenames(_DUPLICATE_PATHS, duplicate_policy="first")

    assert len(result) == 2
    kept = result.loc[result["timestamp"] == pd.Timestamp("2010-12-21 20:00:00"), "image_path"].iloc[0]
    assert kept == "HMI.m2010.12.21_20.00.00.jpg"


def test_build_image_index_duplicate_policy_last_keeps_last_encountered():
    result = build_image_index_from_filenames(_DUPLICATE_PATHS, duplicate_policy="last")

    assert len(result) == 2
    kept = result.loc[result["timestamp"] == pd.Timestamp("2010-12-21 20:00:00"), "image_path"].iloc[0]
    assert kept == "/other/dir/HMI.m2010.12.21_20.00.00.jpg"


def test_build_image_index_duplicate_policy_report_fields():
    df, report = build_image_index_from_filenames(
        _DUPLICATE_PATHS, duplicate_policy="first", return_report=True
    )

    assert report.duplicate_timestamp_count == 1
    assert report.duplicate_policy == "first"
    assert report.final_row_count == 2
    assert len(report.duplicate_rows) == 2  # both rows sharing the duplicated timestamp


def test_build_image_index_invalid_duplicate_policy_raises():
    with pytest.raises(ValueError) as excinfo:
        build_image_index_from_filenames(["HMI.m2010.12.21_20.00.00.jpg"], duplicate_policy="skip")

    assert "duplicate_policy" in str(excinfo.value)


# ── adapt_goes_catalog ───────────────────────────────────────────────────


def _write_source_catalog(tmp_path, rows_csv: str):
    path = tmp_path / "goes_flares_integrated.csv"
    path.write_text(rows_csv)
    return path


def test_adapt_goes_catalog_basic_schema(tmp_path):
    source = _write_source_catalog(
        tmp_path,
        "start_time,peak_time,end_time,goes_class,noaa_active_region,other_col\n"
        "2010-05-01 01:39:00,2010-05-01 01:45:00,2010-05-01 01:51:00,C5.7,11067.0,ignored\n",
    )

    result = adapt_goes_catalog(source)

    assert list(result.columns) == ["start_time", "peak_time", "end_time", "class", "active_region"]
    row = result.iloc[0]
    assert row["start_time"] == pd.Timestamp("2010-05-01 01:39:00")
    assert row["peak_time"] == pd.Timestamp("2010-05-01 01:45:00")
    assert row["end_time"] == pd.Timestamp("2010-05-01 01:51:00")
    assert row["class"] == "C5.7"
    assert row["active_region"] == "11067"


def test_adapt_goes_catalog_preserves_dates_crossing_midnight(tmp_path):
    # This is exactly the bug the old date+HHMM re-encoding introduced:
    # start_time and peak_time/end_time on different calendar dates must
    # stay on their own correct dates, not get flattened onto peak's date.
    source = _write_source_catalog(
        tmp_path,
        "start_time,peak_time,end_time,goes_class,noaa_active_region\n"
        "2020-01-01 23:58:00,2020-01-02 00:04:00,2020-01-02 00:10:00,C1.0,4456.0\n",
    )

    result = adapt_goes_catalog(source)
    row = result.iloc[0]

    assert row["start_time"] == pd.Timestamp("2020-01-01 23:58:00")
    assert row["peak_time"] == pd.Timestamp("2020-01-02 00:04:00")
    assert row["end_time"] == pd.Timestamp("2020-01-02 00:10:00")
    assert row["start_time"].date() != row["peak_time"].date()


def test_adapt_goes_catalog_includes_fl_lon_fl_lat_when_present(tmp_path):
    source = _write_source_catalog(
        tmp_path,
        "start_time,peak_time,end_time,goes_class,noaa_active_region,fl_lon,fl_lat\n"
        "2010-05-01 01:39:00,2010-05-01 01:45:00,2010-05-01 01:51:00,C5.7,11067.0,-73.0,23.0\n",
    )

    result = adapt_goes_catalog(source)

    assert "fl_lon" in result.columns
    assert "fl_lat" in result.columns
    assert result.iloc[0]["fl_lon"] == -73.0
    assert result.iloc[0]["fl_lat"] == 23.0


def test_adapt_goes_catalog_omits_fl_lon_fl_lat_when_absent(tmp_path):
    source = _write_source_catalog(
        tmp_path,
        "start_time,peak_time,end_time,goes_class,noaa_active_region\n"
        "2010-05-01 01:39:00,2010-05-01 01:45:00,2010-05-01 01:51:00,C5.7,11067.0\n",
    )

    result = adapt_goes_catalog(source)

    assert "fl_lon" not in result.columns
    assert "fl_lat" not in result.columns


def test_adapt_goes_catalog_nan_active_region_becomes_none(tmp_path):
    source = _write_source_catalog(
        tmp_path,
        "start_time,peak_time,end_time,goes_class,noaa_active_region\n"
        "2010-05-01 01:39:00,2010-05-01 01:45:00,2010-05-01 01:51:00,C5.7,\n",
    )

    result = adapt_goes_catalog(source)

    assert result.iloc[0]["active_region"] is None


def test_adapt_goes_catalog_single_digit_vs_multi_digit_active_region(tmp_path):
    source = _write_source_catalog(
        tmp_path,
        "start_time,peak_time,end_time,goes_class,noaa_active_region\n"
        "2010-05-01 01:39:00,2010-05-01 01:45:00,2010-05-01 01:51:00,C5.7,7.0\n"
        "2010-05-02 01:39:00,2010-05-02 01:45:00,2010-05-02 01:51:00,M2.3,11067.0\n",
    )

    result = adapt_goes_catalog(source)

    assert result.iloc[0]["active_region"] == "7"
    assert result.iloc[1]["active_region"] == "11067"


def test_adapt_goes_catalog_missing_required_column_raises(tmp_path):
    # No noaa_active_region column at all.
    source = _write_source_catalog(
        tmp_path,
        "start_time,peak_time,end_time,goes_class\n"
        "2010-05-01 01:39:00,2010-05-01 01:45:00,2010-05-01 01:51:00,C5.7\n",
    )

    with pytest.raises(ValueError) as excinfo:
        adapt_goes_catalog(source)

    assert "noaa_active_region" in str(excinfo.value)


def test_adapt_goes_catalog_writes_output_csv(tmp_path):
    source = _write_source_catalog(
        tmp_path,
        "start_time,peak_time,end_time,goes_class,noaa_active_region\n"
        "2010-05-01 01:39:00,2010-05-01 01:45:00,2010-05-01 01:51:00,C5.7,11067.0\n",
    )
    output_path = tmp_path / "flare_catalog.csv"

    result = adapt_goes_catalog(source, output_path=output_path)

    assert output_path.exists()
    on_disk = pd.read_csv(output_path)
    assert list(on_disk.columns) == ["start_time", "peak_time", "end_time", "class", "active_region"]
    assert on_disk.iloc[0]["start_time"] == "2010-05-01 01:39:00"
    assert result.iloc[0]["start_time"] == pd.Timestamp("2010-05-01 01:39:00")


def test_adapt_goes_catalog_output_is_directly_loadable_by_event_matcher(tmp_path):
    # The whole point of the new schema: no adapter-side lossy round-trip,
    # and EventMatcher accepts the result as-is.
    from flare_indexer.events import EventMatcher

    source = _write_source_catalog(
        tmp_path,
        "start_time,peak_time,end_time,goes_class,noaa_active_region\n"
        "2020-01-01 23:58:00,2020-01-02 00:04:00,2020-01-02 00:10:00,M2.3,4456.0\n",
    )
    catalog_path = tmp_path / "flare_catalog.csv"
    adapt_goes_catalog(source, output_path=catalog_path)

    matcher = EventMatcher(catalog_path)
    results = matcher.query(
        pd.Timestamp("2020-01-01 23:00:00"), prediction_window_hours=2, event_time="start"
    )

    assert len(results) == 1
    assert results[0].peak_time == pd.Timestamp("2020-01-02 00:04:00")
