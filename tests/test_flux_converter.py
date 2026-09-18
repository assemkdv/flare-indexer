import pytest
from flare_indexer.classifier import FluxConverter

# ── FluxConverter tests ────────────────────────────────────────────────


def test_to_flux():
    converter = FluxConverter()
    assert converter.to_flux("M2.3") == 2.3e-5
    assert converter.to_flux("C4.0") == 4e-6
    assert converter.to_flux("X1.0") == 1e-4
    assert converter.to_flux("A1.0") == 1e-8


def test_to_flux_invalid_letter_raises():
    converter = FluxConverter()
    with pytest.raises(KeyError):
        converter.to_flux("Z5.0")


def test_to_flux_empty_string_raises():
    converter = FluxConverter()
    with pytest.raises(IndexError):
        converter.to_flux("")


def test_to_flux_non_numeric_suffix_raises():
    converter = FluxConverter()
    with pytest.raises(ValueError):
        converter.to_flux("M2.3.4")
