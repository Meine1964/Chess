import pytest

from src.chess_camera.detection_timing import DetectionTimingMonitor


def _tick(
    monitor: DetectionTimingMonitor,
    at: float,
    *,
    edge_active: bool = False,
    tracker_state: str | None = "watching",
) -> float | None:
    return monitor.record_tick(
        at,
        at + 0.05,
        edge_seconds=0.03,
        tracker_seconds=0.02,
        edge_active=edge_active,
        tracker_state=tracker_state,
    )


def test_latency_runs_from_last_motion_tick_to_acceptance() -> None:
    monitor = DetectionTimingMonitor()

    assert _tick(monitor, 0.0, tracker_state="accepted") is None
    assert _tick(monitor, 0.2, edge_active=True) is None
    assert _tick(monitor, 0.4, tracker_state="activity") is None
    assert _tick(monitor, 0.6, tracker_state="settling") is None
    latency = _tick(monitor, 0.8, tracker_state="accepted")

    assert latency == pytest.approx(0.4)
    assert _tick(monitor, 1.0, tracker_state="compound-accepted") is None
    snapshot = monitor.snapshot()
    assert snapshot["recent_move_latency_seconds"] == [0.4]
    assert snapshot["tick_period"] == {
        "count": 5,
        "mean_ms": 200.0,
        "max_ms": 200.0,
    }
    assert snapshot["edge_detector"]["mean_ms"] == 30.0


def test_ticks_without_detector_work_only_update_tick_statistics() -> None:
    monitor = DetectionTimingMonitor(window_size=2)

    for at in (0.0, 0.1, 0.3):
        monitor.record_tick(
            at,
            at + 0.01,
            edge_seconds=None,
            tracker_seconds=None,
            edge_active=False,
            tracker_state=None,
        )

    snapshot = monitor.snapshot()
    assert snapshot["tick"]["count"] == 2
    assert snapshot["tick_period"]["max_ms"] == 200.0
    assert snapshot["edge_detector"] == {
        "count": 0,
        "mean_ms": None,
        "max_ms": None,
    }
