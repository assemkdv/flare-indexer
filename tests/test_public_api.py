import flare_indexer as fidx


def test_public_names_are_accessible():
    expected = {
        "FlareClassifier",
        "FluxConverter",
        "FlareEvent",
        "EventMatcher",
        "BinaryThresholdStrategy",
        "MaxFlareStrategy",
        "DatasetBuilder",
        "SequenceBuildReport",
        "build_image_index_from_filenames",
        "adapt_goes_catalog",
        "ImageIndexReport",
    }

    assert set(fidx.__all__) == expected
    for name in expected:
        assert hasattr(fidx, name)


def test_top_level_classes_are_usable():
    strategy = fidx.BinaryThresholdStrategy()
    assert strategy.threshold == "M"

    builder = fidx.DatasetBuilder(prediction_window=24, strategy=fidx.BinaryThresholdStrategy())
    assert builder.prediction_window == 24
