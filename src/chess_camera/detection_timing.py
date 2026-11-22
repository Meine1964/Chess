from __future__ import annotations

from collections import deque

_MOTION_STATES = frozenset({"activity", "waiting-for-clear"})
_ACCEPTED_STATES = frozenset({"accepted", "compound-accepted"})


class _RollingStatistic:
    def __init__(self, window_size: int) -> None:
        self._values: deque[float] = deque(maxlen=window_size)

    def add(self, seconds: float) -> None:
        self._values.append(seconds)

    def snapshot(self) -> dict[str, object]:
        if not self._values:
            return {"count": 0, "mean_ms": None, "max_ms": None}
        return {
            "count": len(self._values),
            "mean_ms": round(
                1000.0 * sum(self._values) / len(self._values),
                1,
            ),
            "max_ms": round(1000.0 * max(self._values), 1),
        }


class DetectionTimingMonitor:
    """Rolling occupancy-tick timings and hand-leave-to-acceptance latency."""

    def __init__(self, *, window_size: int = 50, latency_history: int = 10):
        if window_size < 1 or latency_history < 1:
            raise ValueError("Timing windows must be positive.")
        self._tick = _RollingStatistic(window_size)
        self._period = _RollingStatistic(window_size)
        self._edge = _RollingStatistic(window_size)
        self._tracker = _RollingStatistic(window_size)
        self._latencies: deque[float] = deque(maxlen=latency_history)
        self._previous_tick_start: float | None = None
        self._last_motion_at: float | None = None

    def record_tick(
        self,
        started_at: float,
        finished_at: float,
        *,
        edge_seconds: float | None,
        tracker_seconds: float | None,
        edge_active: bool,
        tracker_state: str | None,
    ) -> float | None:
        """Record one tick; return the move latency when it accepted a move."""
        if self._previous_tick_start is not None:
            self._period.add(max(0.0, started_at - self._previous_tick_start))
        self._previous_tick_start = started_at
        self._tick.add(max(0.0, finished_at - started_at))
        if edge_seconds is not None:
            self._edge.add(edge_seconds)
        if tracker_seconds is not None:
            self._tracker.add(tracker_seconds)
        if edge_active or tracker_state in _MOTION_STATES:
            self._last_motion_at = finished_at
            return None
        if tracker_state not in _ACCEPTED_STATES:
            return None
        if self._last_motion_at is None:
            return None
        latency = max(0.0, finished_at - self._last_motion_at)
        self._last_motion_at = None
        self._latencies.append(latency)
        return latency

    def snapshot(self) -> dict[str, object]:
        return {
            "tick": self._tick.snapshot(),
            "tick_period": self._period.snapshot(),
            "edge_detector": self._edge.snapshot(),
            "reference_tracker": self._tracker.snapshot(),
            "recent_move_latency_seconds": [
                round(latency, 3) for latency in self._latencies
            ],
        }
