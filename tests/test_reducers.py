import pandas as pd
import pytest

from flare_indexer.events import FlareEvent
from flare_indexer.reducers import MaxFluxReducer


def _flare(goes_class: str) -> FlareEvent:
    return FlareEvent(
        peak_time=pd.Timestamp("2024-01-01 00:00"),
        goes_class=goes_class,
        start_time=pd.Timestamp("2024-01-01 00:00"),
        active_region=None,
    )


def test_max_flux_reducer_empty_events():
    assert MaxFluxReducer().reduce([]) == 0.0


def test_max_flux_reducer_one_event():
    assert MaxFluxReducer().reduce([_flare("C4.2")]) == pytest.approx(4.2e-6)


def test_max_flux_reducer_multiple_events():
    events = [_flare("C4.0"), _flare("X1.0"), _flare("M2.3")]
    assert MaxFluxReducer().reduce(events) == 1e-4


def test_max_flux_reducer_numeric_not_lexical_ordering():
    # A lexical sort would rank "M9.0" above "M10.0" ('9' > '1' as chars).
    assert MaxFluxReducer().reduce([_flare("M9.0"), _flare("M10.0"), _flare("X1.0")]) == 1e-4
    assert MaxFluxReducer().reduce([_flare("M9.0"), _flare("M10.0")]) == 1e-4


def test_max_flux_reducer_invalid_goes_class_propagates():
    with pytest.raises(KeyError):
        MaxFluxReducer().reduce([_flare("Z5.0")])
