from typing import Protocol

from .classifier import FluxConverter
from .events import FlareEvent

_flux_converter = FluxConverter()


class EventReducer(Protocol):
    """
    Structural contract for reducing a list of FlareEvents to a single
    numeric (or other) value. Any object with a matching `reduce` method
    satisfies this -- no base class or registration required.
    """

    def reduce(self, events: list[FlareEvent]):
        ...


class MaxFluxReducer:
    """
    Reduces a list of FlareEvents to the numeric maximum physical X-ray
    flux among them, via FluxConverter -- never by comparing goes_class
    strings lexically (see FluxConverter's docstring for why that matters:
    "M9.0" < "M10.0" numerically, despite sorting the other way as text).

    Returns 0.0 for an empty event list (no flare in the window).
    """

    def reduce(self, events: list[FlareEvent]) -> float:
        if not events:
            return 0.0
        return max(_flux_converter.to_flux(event.goes_class) for event in events)
