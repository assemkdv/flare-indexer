from .classifier import FlareClassifier, FluxConverter
from .events import FlareEvent

_classifier = FlareClassifier()
_flux_converter = FluxConverter()


class BinaryThresholdStrategy:
    """Returns 1 if any flare in the window meets or exceeds the threshold class, else 0."""

    def __init__(self, threshold: str = "M"):
        self.threshold = threshold

    def label(self, flares: list[FlareEvent]) -> int:
        return int(any(_classifier.is_strong(f.goes_class, self.threshold) for f in flares))


class MaxFlareStrategy:
    """Returns the numeric flux of the strongest flare in the window, or 0.0."""

    def label(self, flares: list[FlareEvent]) -> float:
        if not flares:
            return 0.0
        return max(_flux_converter.to_flux(f.goes_class) for f in flares)
