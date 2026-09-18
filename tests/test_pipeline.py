from pathlib import Path

import pandas as pd
import pytest

from flare_indexer.builder import DatasetBuilder
from flare_indexer.events import EventMatcher, FlareEvent
from flare_indexer.labelers import BinaryThresholdLabeler, RegressionLabeler
from flare_indexer.pipeline import DatasetBuildingPipeline, PipelineResult
from flare_indexer.reducers import MaxFluxReducer
from flare_indexer.strategies import BinaryThresholdStrategy, MaxFlareStrategy

FIXTURES_DIR = Path(__file__).parent / "fixtures"
CATALOG_PATH = FIXTURES_DIR / "catalog.csv"
IMAGE_INDEX_PATH = FIXTURES_DIR / "image_index.csv"


def _flare(goes_class: str, peak="2024-01-01 00:30", start="2024-01-01 00:20", active_region=None) -> FlareEvent:
    return FlareEvent(
        peak_time=pd.Timestamp(peak),
        goes_class=goes_class,
        start_time=pd.Timestamp(start),
        active_region=active_region,
    )


def _matcher_with_flares(rows: list[dict]) -> EventMatcher:
    matcher = EventMatcher.__new__(EventMatcher)
    matcher._catalog = pd.DataFrame(rows)
    return matcher


def _default_pipeline() -> DatasetBuildingPipeline:
    return DatasetBuildingPipeline(reducer=MaxFluxReducer(), labeler=BinaryThresholdLabeler("M"))


# ── Component validation (Task 9) ─────────────────────────────────────────


def test_invalid_reducer_fails_clearly():
    with pytest.raises(TypeError) as excinfo:
        DatasetBuildingPipeline(reducer=object(), labeler=BinaryThresholdLabeler("M"))
    message = str(excinfo.value)
    assert "reducer" in message
    assert "reduce" in message


def test_invalid_labeler_fails_clearly():
    with pytest.raises(TypeError) as excinfo:
        DatasetBuildingPipeline(reducer=MaxFluxReducer(), labeler=object())
    message = str(excinfo.value)
    assert "labeler" in message
    assert "assign" in message


def test_reducer_with_non_callable_reduce_attribute_fails():
    class BadReducer:
        reduce = "not callable"

    with pytest.raises(TypeError):
        DatasetBuildingPipeline(reducer=BadReducer(), labeler=BinaryThresholdLabeler("M"))


def test_custom_duck_typed_reducer_and_labeler_work():
    class CustomReducer:
        def reduce(self, events):
            return len(events)

    class CustomLabeler:
        def assign(self, value):
            return "many" if value > 1 else "few"

    pipeline = DatasetBuildingPipeline(reducer=CustomReducer(), labeler=CustomLabeler())

    assert pipeline.reduce_events([_flare("M1.0"), _flare("M2.0")]) == 2
    assert pipeline.assign_label(2) == "many"
    assert pipeline.assign_label(1) == "few"
    assert pipeline.label([_flare("M1.0")]) == "few"


# ── extract_events ─────────────────────────────────────────────────────


def test_extract_events_delegates_to_event_matcher():
    matcher = _matcher_with_flares([
        {
            "peak_time": pd.Timestamp("2024-01-01 00:30"), "start_time": pd.Timestamp("2024-01-01 00:20"),
            "goes_class": "M2.3", "active_region": None,
        },
    ])
    pipeline = _default_pipeline()

    events = pipeline.extract_events(matcher, pd.Timestamp("2024-01-01 00:00"), prediction_window=1)

    assert len(events) == 1
    assert events[0].goes_class == "M2.3"


def test_extract_events_event_time_start():
    matcher = _matcher_with_flares([
        {
            "peak_time": pd.Timestamp("2024-01-01 01:00"), "start_time": pd.Timestamp("2024-01-01 00:30"),
            "goes_class": "M2.3", "active_region": None,
        },
    ])
    pipeline = _default_pipeline()

    events = pipeline.extract_events(
        matcher, pd.Timestamp("2024-01-01 00:00"), prediction_window=0.75, event_time="start",
    )
    assert len(events) == 1  # start (00:30) is inside [00:00, 00:45)


def test_extract_events_event_time_peak():
    matcher = _matcher_with_flares([
        {
            "peak_time": pd.Timestamp("2024-01-01 01:00"), "start_time": pd.Timestamp("2024-01-01 00:30"),
            "goes_class": "M2.3", "active_region": None,
        },
    ])
    pipeline = _default_pipeline()

    events = pipeline.extract_events(
        matcher, pd.Timestamp("2024-01-01 00:00"), prediction_window=0.75, event_time="peak",
    )
    assert len(events) == 0  # peak (01:00) is outside [00:00, 00:45)


def test_extract_events_interval_mode_left_closed():
    matcher = _matcher_with_flares([
        {
            "peak_time": pd.Timestamp("2024-01-01 01:00"), "start_time": pd.Timestamp("2024-01-01 00:50"),
            "goes_class": "X1.0", "active_region": None,
        },
    ])
    pipeline = _default_pipeline()

    events = pipeline.extract_events(
        matcher, pd.Timestamp("2024-01-01 00:00"), prediction_window=1, interval_mode="left_closed",
    )
    assert events == []  # peak exactly at t+1h excluded by [t, t+1h)


def test_extract_events_interval_mode_right_closed():
    matcher = _matcher_with_flares([
        {
            "peak_time": pd.Timestamp("2024-01-01 01:00"), "start_time": pd.Timestamp("2024-01-01 00:50"),
            "goes_class": "X1.0", "active_region": None,
        },
    ])
    pipeline = _default_pipeline()

    events = pipeline.extract_events(
        matcher, pd.Timestamp("2024-01-01 00:00"), prediction_window=1, interval_mode="right_closed",
    )
    assert len(events) == 1  # peak exactly at t+1h included by (t, t+1h]


def test_extract_events_active_region_forwarding():
    matcher = _matcher_with_flares([
        {
            "peak_time": pd.Timestamp("2024-01-01 00:30"), "start_time": pd.Timestamp("2024-01-01 00:20"),
            "goes_class": "M5.1", "active_region": "3559",
        },
        {
            "peak_time": pd.Timestamp("2024-01-01 00:31"), "start_time": pd.Timestamp("2024-01-01 00:21"),
            "goes_class": "X9.0", "active_region": "9999",
        },
    ])
    pipeline = _default_pipeline()

    events = pipeline.extract_events(
        matcher, pd.Timestamp("2024-01-01 00:00"), prediction_window=1, active_region="3559",
    )
    assert len(events) == 1
    assert events[0].goes_class == "M5.1"


# ── reduce_events / assign_label independence ─────────────────────────────


def test_reduce_events_returns_intermediate_result_independently():
    pipeline = _default_pipeline()
    reduced = pipeline.reduce_events([_flare("M2.3")])
    assert reduced == 2.3e-5


def test_assign_label_can_be_called_independently_of_extract_or_reduce():
    pipeline = _default_pipeline()
    # No extract_events()/reduce_events() call ever happened on this
    # pipeline instance before this -- assign_label must not depend on it.
    assert pipeline.assign_label(2.3e-5) == 1
    assert pipeline.assign_label(0.0) == 0


def test_pipeline_methods_are_stateless_across_calls():
    pipeline = _default_pipeline()
    pipeline.reduce_events([_flare("X1.0")])  # any prior call
    # A fresh, unrelated call must not be influenced by the call above.
    assert pipeline.reduce_events([]) == 0.0
    assert pipeline.assign_label(0.0) == 0


# ── run_one / PipelineResult ───────────────────────────────────────────


def test_run_one_returns_pipeline_result():
    matcher = _matcher_with_flares([
        {
            "peak_time": pd.Timestamp("2024-01-01 00:30"), "start_time": pd.Timestamp("2024-01-01 00:20"),
            "goes_class": "M2.3", "active_region": None,
        },
    ])
    pipeline = _default_pipeline()

    result = pipeline.run_one(matcher, pd.Timestamp("2024-01-01 00:00"), prediction_window=1)

    assert isinstance(result, PipelineResult)
    assert len(result.events) == 1
    assert result.events[0].goes_class == "M2.3"
    assert result.reduced_value == 2.3e-5
    assert result.label == 1


def test_run_one_no_flares_in_window():
    matcher = _matcher_with_flares([
        {
            "peak_time": pd.Timestamp("2024-01-02 00:30"), "start_time": pd.Timestamp("2024-01-02 00:20"),
            "goes_class": "M2.3", "active_region": None,
        },
    ])
    pipeline = _default_pipeline()

    result = pipeline.run_one(matcher, pd.Timestamp("2024-01-01 00:00"), prediction_window=1)

    assert result.events == []
    assert result.reduced_value == 0.0
    assert result.label == 0


# ── No mutation of caller's data ──────────────────────────────────────────


def test_reduce_events_does_not_mutate_caller_list_even_with_misbehaving_reducer():
    class MutatingReducer:
        def reduce(self, events):
            events.append(_flare("X1.0"))  # a badly-behaved custom reducer
            return len(events)

    pipeline = DatasetBuildingPipeline(reducer=MutatingReducer(), labeler=RegressionLabeler())
    original_events = [_flare("M1.0")]

    pipeline.reduce_events(original_events)

    assert len(original_events) == 1  # caller's own list is untouched


def test_extract_events_returns_fresh_list_not_shared_with_matcher():
    matcher = _matcher_with_flares([
        {
            "peak_time": pd.Timestamp("2024-01-01 00:30"), "start_time": pd.Timestamp("2024-01-01 00:20"),
            "goes_class": "M2.3", "active_region": None,
        },
    ])
    pipeline = _default_pipeline()

    events_a = pipeline.extract_events(matcher, pd.Timestamp("2024-01-01 00:00"), prediction_window=1)
    events_a.clear()  # mutate the returned list
    events_b = pipeline.extract_events(matcher, pd.Timestamp("2024-01-01 00:00"), prediction_window=1)

    assert len(events_b) == 1  # unaffected by mutating the earlier result


# ── DatasetBuilder compatibility via label() ──────────────────────────────


def test_pipeline_label_method_matches_manual_reduce_then_assign():
    pipeline = _default_pipeline()
    events = [_flare("C4.0"), _flare("M2.3")]
    assert pipeline.label(events) == pipeline.assign_label(pipeline.reduce_events(events))


def test_dataset_builder_accepts_pipeline_as_strategy():
    pipeline = DatasetBuildingPipeline(reducer=MaxFluxReducer(), labeler=BinaryThresholdLabeler())
    builder = DatasetBuilder(prediction_window=24, strategy=pipeline)
    result = builder.build(IMAGE_INDEX_PATH, CATALOG_PATH)

    assert list(result.columns) == ["timestamp", "label"]
    assert result["label"].tolist() == [0, 1, 0]  # matches test_end_to_end_binary_threshold_strategy


# ── Exception propagation (Task 10) ────────────────────────────────────────


def test_exceptions_from_custom_reducer_are_not_swallowed():
    class BrokenReducer:
        def reduce(self, events):
            raise RuntimeError("reducer exploded")

    pipeline = DatasetBuildingPipeline(reducer=BrokenReducer(), labeler=RegressionLabeler())
    with pytest.raises(RuntimeError, match="reducer exploded"):
        pipeline.reduce_events([_flare("M1.0")])


def test_exceptions_from_custom_labeler_are_not_swallowed():
    class BrokenLabeler:
        def assign(self, value):
            raise RuntimeError("labeler exploded")

    pipeline = DatasetBuildingPipeline(reducer=MaxFluxReducer(), labeler=BrokenLabeler())
    with pytest.raises(RuntimeError, match="labeler exploded"):
        pipeline.assign_label(1e-5)


def test_invalid_goes_class_from_event_matcher_propagates_through_pipeline():
    matcher = _matcher_with_flares([
        {
            "peak_time": pd.Timestamp("2024-01-01 00:30"), "start_time": pd.Timestamp("2024-01-01 00:20"),
            "goes_class": "Z9.9", "active_region": None,
        },
    ])
    pipeline = _default_pipeline()
    with pytest.raises(KeyError):
        pipeline.run_one(matcher, pd.Timestamp("2024-01-01 00:00"), prediction_window=1)


# ── Backward compatibility ─────────────────────────────────────────────


def test_binary_threshold_strategy_produces_same_labels_as_before():
    strategy = BinaryThresholdStrategy(threshold="M")
    assert strategy.label([_flare("M2.3")]) == 1
    assert strategy.label([_flare("C4.0")]) == 0
    assert strategy.label([]) == 0


def test_max_flare_strategy_produces_same_values_as_before():
    strategy = MaxFlareStrategy()
    assert strategy.label([_flare("C4.0"), _flare("X1.0"), _flare("M2.3")]) == 1e-4
    assert strategy.label([]) == 0.0


def test_dataset_builder_works_with_old_binary_threshold_strategy(tmp_path):
    builder = DatasetBuilder(prediction_window=24, strategy=BinaryThresholdStrategy())
    result = builder.build(IMAGE_INDEX_PATH, CATALOG_PATH)
    assert result["label"].tolist() == [0, 1, 0]


def test_dataset_builder_works_with_old_max_flare_strategy(tmp_path):
    builder = DatasetBuilder(prediction_window=24, strategy=MaxFlareStrategy())
    result = builder.build(IMAGE_INDEX_PATH, CATALOG_PATH)
    assert result["label"].tolist() == [3e-6, 4e-5, 0.0]


def test_full_disk_row_based_sequences_work_with_pipeline(tmp_path):
    index_path = tmp_path / "sequence_index.csv"
    pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2024-02-01T09:00:00", "2024-02-01T10:00:00",
            "2024-02-01T11:00:00", "2024-02-01T12:00:00",
        ]),
    }).to_csv(index_path, index=False)

    pipeline = _default_pipeline()
    builder = DatasetBuilder(
        prediction_window=1, strategy=pipeline, sequence_length=2, stride=1, cadence_minutes=60,
    )
    result = builder.build(index_path, CATALOG_PATH)

    assert list(result.columns) == ["sequence_start", "sequence_end", "timestamps", "n_images", "label"]
    assert result["label"].tolist() == [0, 0, 0]


def test_full_disk_time_based_sequences_work_with_pipeline(tmp_path):
    index_path = tmp_path / "index.csv"
    pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01T00:00:00", periods=10, freq="1h"),
    }).to_csv(index_path, index=False)
    empty_catalog = tmp_path / "catalog.csv"
    empty_catalog.write_text("date,start,peak,end,class,active_region\n")

    pipeline = _default_pipeline()
    builder = DatasetBuilder(
        prediction_window=24, strategy=pipeline,
        cadence="1h", observation_window="6h", sliding_window="3h",
    )
    result, report = builder.build(index_path, empty_catalog, return_report=True)

    assert list(result.columns) == ["sequence_start", "sequence_end", "timestamps", "n_images", "label"]
    assert len(result) == 2
    assert report.emitted_sequences == 2


def test_sequence_build_report_unaffected_by_strategy_vs_pipeline(tmp_path):
    # SequenceBuildReport is computed from candidate-timestamp generation
    # alone, independent of labeling -- confirm it's identical whether the
    # strategy is the legacy BinaryThresholdStrategy or the equivalent pipeline.
    index_path = tmp_path / "index.csv"
    pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01T00:00:00", periods=10, freq="1h"),
    }).to_csv(index_path, index=False)
    empty_catalog = tmp_path / "catalog.csv"
    empty_catalog.write_text("date,start,peak,end,class,active_region\n")

    _, legacy_report = DatasetBuilder(
        prediction_window=24, strategy=BinaryThresholdStrategy(),
        cadence="1h", observation_window="6h", sliding_window="3h",
    ).build(index_path, empty_catalog, return_report=True)

    _, pipeline_report = DatasetBuilder(
        prediction_window=24, strategy=_default_pipeline(),
        cadence="1h", observation_window="6h", sliding_window="3h",
    ).build(index_path, empty_catalog, return_report=True)

    assert legacy_report == pipeline_report


def test_active_region_single_image_works_with_pipeline(tmp_path):
    index_path = tmp_path / "ar_index.csv"
    pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-02-01T00:00:00", "2024-02-05T00:00:00", "2024-02-05T00:00:00"]),
        "active_region": ["11111", "11111", "22222"],
    }).to_csv(index_path, index=False)

    pipeline = DatasetBuildingPipeline(reducer=MaxFluxReducer(), labeler=BinaryThresholdLabeler("C"))
    builder = DatasetBuilder(prediction_window=24, strategy=pipeline, target="active_region")
    result = builder.build(index_path, CATALOG_PATH)

    assert list(result.columns) == ["timestamp", "label"]
    assert result["label"].tolist() == [1, 0, 1]


def test_active_region_sequences_work_with_pipeline(tmp_path):
    index_path = tmp_path / "ar_sequence_index.csv"
    pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2024-02-01T11:00:00", "2024-02-01T11:30:00", "2024-02-01T12:00:00",
            "2024-02-05T09:00:00", "2024-02-05T09:30:00", "2024-02-05T10:00:00",
        ]),
        "active_region": ["11111", "11111", "11111", "22222", "22222", "22222"],
    }).to_csv(index_path, index=False)

    pipeline = DatasetBuildingPipeline(reducer=MaxFluxReducer(), labeler=BinaryThresholdLabeler("C"))
    builder = DatasetBuilder(
        prediction_window=1, strategy=pipeline, target="active_region",
        sequence_length=3, stride=3, cadence_minutes=30,
    )
    result = builder.build(index_path, CATALOG_PATH)

    assert result["label"].tolist() == [1, 0]
    assert result["active_region"].tolist() == ["11111", "22222"]


def test_extra_columns_preserved_with_pipeline(tmp_path):
    index_path = tmp_path / "index.csv"
    pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2024-02-01T09:00:00", "2024-02-01T10:00:00", "2024-02-01T11:00:00",
        ]),
        "image_path": ["a.jpg", "b.jpg", "c.jpg"],
    }).to_csv(index_path, index=False)

    pipeline = _default_pipeline()
    builder = DatasetBuilder(prediction_window=1, strategy=pipeline, sequence_length=2, stride=1)
    result = builder.build(index_path, CATALOG_PATH)

    assert "image_path" in result.columns
    assert result.iloc[0]["image_path"] == ["a.jpg", "b.jpg"]


# ── End-to-end equivalence (strategy vs. pipeline) ────────────────────────


def test_binary_threshold_strategy_equivalent_to_pipeline(tmp_path):
    legacy_result = DatasetBuilder(
        prediction_window=24, strategy=BinaryThresholdStrategy(threshold="M"),
    ).build(IMAGE_INDEX_PATH, CATALOG_PATH)

    pipeline = DatasetBuildingPipeline(reducer=MaxFluxReducer(), labeler=BinaryThresholdLabeler("M"))
    pipeline_result = DatasetBuilder(
        prediction_window=24, strategy=pipeline,
    ).build(IMAGE_INDEX_PATH, CATALOG_PATH)

    pd.testing.assert_frame_equal(legacy_result, pipeline_result)


def test_max_flare_strategy_equivalent_to_pipeline(tmp_path):
    legacy_result = DatasetBuilder(
        prediction_window=24, strategy=MaxFlareStrategy(),
    ).build(IMAGE_INDEX_PATH, CATALOG_PATH)

    pipeline = DatasetBuildingPipeline(reducer=MaxFluxReducer(), labeler=RegressionLabeler())
    pipeline_result = DatasetBuilder(
        prediction_window=24, strategy=pipeline,
    ).build(IMAGE_INDEX_PATH, CATALOG_PATH)

    pd.testing.assert_frame_equal(legacy_result, pipeline_result)
