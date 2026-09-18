import math

import pandas as pd
import pytest

from flare_indexer.loaders import adapt_goes_catalog, build_image_index_from_filenames

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


def test_build_image_index_writes_output_csv(tmp_path):
    paths = ["/data/HMI.m2010.12.21_21.00.00.jpg"]
    output_path = tmp_path / "image_index.csv"

    result = build_image_index_from_filenames(paths, output_path=output_path)

    assert output_path.exists()
    on_disk = pd.read_csv(output_path)
    assert list(on_disk.columns) == ["timestamp", "image_path"]
    assert len(on_disk) == 1
    assert result["image_path"].iloc[0] == paths[0]


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

    assert list(result.columns) == ["date", "start", "peak", "end", "class", "active_region"]
    row = result.iloc[0]
    assert row["date"] == "2010-05-01"
    assert row["start"] == "0139"
    assert row["peak"] == "0145"
    assert row["end"] == "0151"
    assert row["class"] == "C5.7"
    assert row["active_region"] == "11067"


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


def test_adapt_goes_catalog_zero_pads_single_digit_hour_and_minute(tmp_path):
    source = _write_source_catalog(
        tmp_path,
        "start_time,peak_time,end_time,goes_class,noaa_active_region\n"
        "2010-05-01 01:05:00,2010-05-01 01:09:00,2010-05-01 01:11:00,C5.7,11067.0\n",
    )

    result = adapt_goes_catalog(source)

    assert result.iloc[0]["start"] == "0105"
    assert result.iloc[0]["peak"] == "0109"
    assert result.iloc[0]["end"] == "0111"


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
    assert list(on_disk.columns) == ["date", "start", "peak", "end", "class", "active_region"]
    assert on_disk.iloc[0]["start"] == 139  # read back as int by pandas; string "0139" round-trips as 139
    assert result.iloc[0]["start"] == "0139"
