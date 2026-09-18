import pandas as pd
from flare_indexer.strategies import BinaryThresholdStrategy, MaxFlareStrategy
from flare_indexer.events import FlareEvent


def _flare(goes_class: str) -> FlareEvent:
    return FlareEvent(
        peak_time=pd.Timestamp("2024-01-01 00:00"),
        goes_class=goes_class,
        start_time=pd.Timestamp("2024-01-01 00:00"),
        active_region=None,
    )


def test_binary_threshold_default_is_m_class():
    strategy = BinaryThresholdStrategy()
    assert strategy.threshold == "M"
    assert strategy.label([_flare("M2.3")]) == 1
    assert strategy.label([_flare("C4.0")]) == 0


def test_binary_threshold_custom_c_class():
    strategy = BinaryThresholdStrategy(threshold="C")
    assert strategy.label([_flare("C1.0")]) == 1
    assert strategy.label([_flare("B9.0")]) == 0


def test_binary_threshold_any_qualifying_flare_gives_one():
    strategy = BinaryThresholdStrategy()
    flares = [_flare("B1.0"), _flare("C4.0"), _flare("X1.0")]
    assert strategy.label(flares) == 1


def test_binary_threshold_no_qualifying_flares_gives_zero():
    strategy = BinaryThresholdStrategy()
    assert strategy.label([_flare("B1.0"), _flare("C4.0")]) == 0
    assert strategy.label([]) == 0


def test_max_flare_empty_list_returns_zero():
    strategy = MaxFlareStrategy()
    assert strategy.label([]) == 0.0


def test_max_flare_single_flare_returns_its_flux():
    strategy = MaxFlareStrategy()
    assert strategy.label([_flare("M2.3")]) == 2.3e-5


def test_max_flare_multiple_flares_returns_max_flux():
    strategy = MaxFlareStrategy()
    flares = [_flare("C4.0"), _flare("X1.0"), _flare("M2.3")]
    assert strategy.label(flares) == 1e-4
