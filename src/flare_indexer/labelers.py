import numbers
from typing import Protocol

from .classifier import FluxConverter

_flux_converter = FluxConverter()


class LabelAssigner(Protocol):
    """
    Structural contract for turning a reducer's output into a final label.
    Any object with a matching `assign` method satisfies this -- no base
    class or registration required.
    """

    def assign(self, reduced_value):
        ...


class BinaryThresholdLabeler:
    """
    Returns 1 if a reduced flux value meets or exceeds a GOES class
    threshold, else 0.

    threshold may be a plain letter ("M") or a full GOES class with a
    number ("C4.2") -- both are converted via FluxConverter.to_flux(), so
    the comparison is always numeric.
    """

    def __init__(self, threshold: str = "M"):
        self.threshold = threshold
        try:
            self._threshold_flux = _flux_converter.to_flux(threshold)
        except (KeyError, IndexError, ValueError) as exc:
            raise ValueError(f"Invalid GOES threshold {threshold!r}: {exc}") from exc

    def assign(self, reduced_value) -> int:
        return int(reduced_value >= self._threshold_flux)


class RegressionLabeler:
    """
    Passes a reduced numeric flux value through unchanged (0.0 for no
    flare), normalizing scalar numeric types (e.g. a numpy float64) to
    plain Python int/float so callers always get a standard scalar back.
    """

    def assign(self, reduced_value):
        if isinstance(reduced_value, numbers.Integral):
            return int(reduced_value)
        if isinstance(reduced_value, numbers.Real):
            return float(reduced_value)
        return reduced_value
