from dataclasses import dataclass
from typing import Any

from .events import EventMatcher, FlareEvent


@dataclass
class PipelineResult:
    """The full set of intermediate values a run_one() call produced."""

    events: list[FlareEvent]
    reduced_value: Any
    label: Any


def _require_method(component: object, method_name: str, role: str) -> None:
    method = getattr(component, method_name, None)
    if method is None or not callable(method):
        raise TypeError(
            f"{role}={component!r} must provide a callable {method_name!r} method "
            f"(e.g. `def {method_name}(self, ...): ...`); duck typing is fine, "
            "subclassing a base class is not required."
        )


class DatasetBuildingPipeline:
    """
    Modular, scikit-learn-inspired composition of the labeling process into
    three independently callable stages:

        image/sequence reference time
              |
              v
        [ EventMatcher ]   -> extract_events(): list[FlareEvent]
              |
              v
        [ EventReducer ]   -> reduce_events(): a single reduced value
              |
              v
        [ LabelAssigner ]  -> assign_label(): the final label
              |
              v
             label

    Each method takes its inputs explicitly and returns its output
    directly -- none of them read or write any mutable state on the
    pipeline instance, so extract_events(), reduce_events(), and
    assign_label() can be called in any combination, independently, and
    a pipeline instance is safe to reuse and share.

    reducer and labeler may be any object exposing a callable `reduce`
    (EventReducer) or `assign` (LabelAssigner) method respectively --
    project-provided or entirely custom, duck-typed classes both work.
    """

    def __init__(self, reducer, labeler):
        _require_method(reducer, "reduce", "reducer")
        _require_method(labeler, "assign", "labeler")
        self.reducer = reducer
        self.labeler = labeler

    def extract_events(
        self,
        matcher: EventMatcher,
        reference_time,
        prediction_window,
        active_region=None,
        event_time: str = "peak",
        interval_mode: str = "left_closed",
    ) -> list[FlareEvent]:
        """Stage 1: query EventMatcher and return the matched FlareEvents."""
        events = matcher.query(
            reference_time, prediction_window, active_region=active_region,
            event_time=event_time, interval_mode=interval_mode,
        )
        # Defensive copy: guarantees the list handed back is never a
        # reference a caller (or a later pipeline call) could share and
        # accidentally mutate, regardless of EventMatcher's own internals.
        return list(events)

    def reduce_events(self, events: list[FlareEvent]):
        """Stage 2: reduce a list of FlareEvents to a single value via self.reducer."""
        # Pass the reducer a copy so nothing it does to its argument can
        # ever mutate the caller's own events list.
        return self.reducer.reduce(list(events))

    def assign_label(self, reduced_value):
        """Stage 3: turn a reduced value into a final label via self.labeler."""
        return self.labeler.assign(reduced_value)

    def run_one(
        self,
        matcher: EventMatcher,
        reference_time,
        prediction_window,
        active_region=None,
        event_time: str = "peak",
        interval_mode: str = "left_closed",
    ) -> PipelineResult:
        """Run all three stages for one reference time and return every intermediate value."""
        events = self.extract_events(
            matcher, reference_time, prediction_window,
            active_region=active_region, event_time=event_time, interval_mode=interval_mode,
        )
        reduced_value = self.reduce_events(events)
        label = self.assign_label(reduced_value)
        return PipelineResult(events=events, reduced_value=reduced_value, label=label)

    def label(self, events: list[FlareEvent]):
        """
        DatasetBuilder-compatibility shim.

        DatasetBuilder only ever calls `strategy.label(flares)` on
        whatever object it was given as `strategy=`. Implementing that
        same method here lets a DatasetBuildingPipeline be passed directly
        as `DatasetBuilder(..., strategy=pipeline)`, with no changes to
        DatasetBuilder itself.
        """
        reduced_value = self.reduce_events(events)
        return self.assign_label(reduced_value)
