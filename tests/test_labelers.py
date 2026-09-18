import pytest

from flare_indexer.labelers import BinaryThresholdLabeler, RegressionLabeler


# ── BinaryThresholdLabeler ─────────────────────────────────────────────


def test_binary_threshold_labeler_below_threshold():
    labeler = BinaryThresholdLabeler("M")  # 1e-5
    assert labeler.assign(9e-6) == 0


def test_binary_threshold_labeler_exactly_at_threshold():
    labeler = BinaryThresholdLabeler("M")  # 1e-5
    assert labeler.assign(1e-5) == 1


def test_binary_threshold_labeler_above_threshold():
    labeler = BinaryThresholdLabeler("M")
    assert labeler.assign(1e-4) == 1


def test_binary_threshold_labeler_custom_numeric_goes_threshold():
    labeler = BinaryThresholdLabeler("C4.2")  # 4.2e-6
    assert labeler.assign(4.2e-6) == 1
    assert labeler.assign(4.1e-6) == 0
    assert labeler.assign(5e-6) == 1


def test_binary_threshold_labeler_zero_no_flare():
    labeler = BinaryThresholdLabeler("M")
    assert labeler.assign(0.0) == 0


def test_binary_threshold_labeler_invalid_threshold_raises():
    with pytest.raises(ValueError) as excinfo:
        BinaryThresholdLabeler("Z5.0")
    assert "Z5.0" in str(excinfo.value)


def test_binary_threshold_labeler_empty_threshold_raises():
    with pytest.raises(ValueError):
        BinaryThresholdLabeler("")


def test_binary_threshold_labeler_malformed_number_raises():
    with pytest.raises(ValueError):
        BinaryThresholdLabeler("M2.3.4")


# ── RegressionLabeler ────────────────────────────────────────────────────


def test_regression_labeler_positive_flux():
    assert RegressionLabeler().assign(2.3e-5) == 2.3e-5


def test_regression_labeler_zero_flux():
    assert RegressionLabeler().assign(0.0) == 0.0


def test_regression_labeler_returns_python_float():
    import numpy as np

    result = RegressionLabeler().assign(np.float64(1.5e-5))
    assert result == 1.5e-5
    assert type(result) is float


def test_regression_labeler_returns_python_int_for_integral_input():
    import numpy as np

    result = RegressionLabeler().assign(np.int64(3))
    assert result == 3
    assert type(result) is int
