import os

import cv2
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from src.chess_camera.alignment_monitor import (
    RECOVERY_PATTERN,
    AlignmentMonitor,
    recovery_pattern_image,
)
from src.chess_camera.board_detector import BoardDetection, BoardDetector
from src.chess_camera.calibration_controller import CalibrationResult
from src.chess_camera.projector_calibration import MappingResult, ValidationResult


@pytest.fixture(scope="module", autouse=True)
def application() -> QApplication:
    existing = QApplication.instance()
    if isinstance(existing, QApplication):
        return existing
    return QApplication([])


def _detection(offset_x: float = 0, offset_y: float = 0) -> BoardDetection:
    coordinates = np.linspace(10, 90, 9, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(coordinates, coordinates)
    grid = np.stack(
        (x_grid + offset_x, y_grid + offset_y),
        axis=2,
    ).astype(np.float32)
    return BoardDetection(
        grid_points=grid,
        confidence=1.0,
        reprojection_error=0.0,
        method="test",
        grid_alignment=1.0,
    )


def _result() -> CalibrationResult:
    detection = _detection()
    point = np.asarray([[50, 50]], dtype=np.float32)
    error = np.asarray([0], dtype=np.float32)
    return CalibrationResult(
        board_detection=detection,
        mapping=MappingResult(
            camera_grid=detection.grid_points,
            projector_grid=detection.grid_points,
            camera_to_projector_homography=np.eye(3, dtype=np.float32),
            coverage_ratio=1.0,
            decoded_ratio=1.0,
            inlier_ratio=1.0,
            rms_error=0.0,
            maximum_error=0.0,
        ),
        validation=ValidationResult(
            measured_points=point,
            expected_points=point,
            errors=error,
            mean_error=0.0,
            rms_error=0.0,
            maximum_error=0.0,
            maximum_square_fraction=0.0,
        ),
    )


def test_monitor_remaps_after_board_movement() -> None:
    monitor = AlignmentMonitor(BoardDetector())
    lost_messages: list[str] = []
    updated_mappings: list[object] = []
    monitor.alignment_lost.connect(lost_messages.append)
    monitor.mapping_updated.connect(
        lambda mapping, _displacement: updated_mappings.append(mapping)
    )
    monitor.configure(_result(), (200, 200))
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    moved = _detection(offset_x=8)

    monitor._check_completed(monitor._generation, moved, frame)
    assert not lost_messages
    assert len(updated_mappings) == 1
    np.testing.assert_allclose(
        updated_mappings[0].camera_grid,
        moved.grid_points,
    )


def test_monitor_ignores_temporary_board_occlusion() -> None:
    monitor = AlignmentMonitor(BoardDetector())
    lost_messages: list[str] = []
    status_messages: list[str] = []
    updated_mappings: list[object] = []
    monitor.alignment_lost.connect(lost_messages.append)
    monitor.status_changed.connect(status_messages.append)
    monitor.mapping_updated.connect(
        lambda mapping, _displacement: updated_mappings.append(mapping)
    )
    monitor.configure(_result(), (200, 200))
    status_messages.clear()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    for _check in range(10):
        monitor._check_completed(monitor._generation, None, frame)

    assert not lost_messages
    assert not status_messages
    assert not updated_mappings


def test_monitor_waits_for_board_to_settle_after_remapping() -> None:
    monitor = AlignmentMonitor(
        BoardDetector(),
        stable_checks_after_remap=5,
    )
    lost_messages: list[str] = []
    calibration_errors: list[tuple[str, str]] = []
    monitor.alignment_lost.connect(lost_messages.append)
    monitor.calibration_error.connect(
        lambda source, message: calibration_errors.append((source, message))
    )
    monitor.configure(_result(), (200, 200))
    detection = _detection()
    black = np.full((100, 100, 3), 20, dtype=np.uint8)
    monitor.set_black()
    monitor._check_completed(monitor._generation, detection, black)
    monitor.set_selected_square(3, 3)

    moved = _detection(offset_x=8)
    monitor._check_completed(monitor._generation, moved, black)
    for _check in range(4):
        monitor._check_completed(monitor._generation, moved, black)

    assert not lost_messages

    monitor._check_completed(monitor._generation, moved, black)
    assert not lost_messages
    monitor._check_completed(monitor._generation, moved, black)
    assert len(lost_messages) == 1
    assert calibration_errors[0][0] == "camera"


def test_monitor_verifies_remapped_board_after_it_settles() -> None:
    monitor = AlignmentMonitor(
        BoardDetector(),
        stable_checks_after_remap=5,
    )
    verified_mappings: list[object] = []
    monitor.mapping_verified.connect(verified_mappings.append)
    monitor.configure(_result(), (200, 200))
    detection = _detection()
    black = np.full((100, 100, 3), 20, dtype=np.uint8)
    monitor.set_black()
    monitor._check_completed(monitor._generation, detection, black)
    monitor.set_selected_square(3, 3)

    moved = _detection(offset_x=8)
    monitor._check_completed(monitor._generation, moved, black)
    for _check in range(4):
        monitor._check_completed(monitor._generation, moved, black)

    aligned = black.copy()
    cv2.rectangle(aligned, (50, 42), (56, 48), (180, 180, 180), -1)
    monitor._check_completed(monitor._generation, moved, aligned)

    assert len(verified_mappings) == 1
    np.testing.assert_allclose(
        verified_mappings[0].camera_grid,
        moved.grid_points,
    )


def test_monitor_invalidates_shifted_projected_square() -> None:
    monitor = AlignmentMonitor(BoardDetector())
    lost_messages: list[str] = []
    calibration_errors: list[tuple[str, str]] = []
    monitor.alignment_lost.connect(lost_messages.append)
    monitor.calibration_error.connect(
        lambda source, message: calibration_errors.append((source, message))
    )
    monitor.configure(_result(), (200, 200))
    detection = _detection()
    black = np.full((100, 100, 3), 20, dtype=np.uint8)
    monitor.set_black()
    monitor._check_completed(monitor._generation, detection, black)

    monitor.set_selected_square(3, 3)
    shifted = black.copy()
    cv2.rectangle(shifted, (48, 42), (56, 48), (180, 180, 180), -1)
    monitor._check_completed(monitor._generation, detection, shifted)
    assert not lost_messages
    monitor._check_completed(monitor._generation, detection, shifted)

    assert len(lost_messages) == 1
    assert calibration_errors[0][0] == "projector"
    assert "projected square is misaligned" in lost_messages[0]


def test_monitor_skips_optical_check_for_occupied_square() -> None:
    monitor = AlignmentMonitor(BoardDetector())
    lost_messages: list[str] = []
    status_messages: list[str] = []
    monitor.alignment_lost.connect(lost_messages.append)
    monitor.status_changed.connect(status_messages.append)
    monitor.configure(_result(), (200, 200))
    detection = _detection()
    black = np.full((100, 100, 3), 20, dtype=np.uint8)
    monitor.set_black()
    monitor._check_completed(monitor._generation, detection, black)

    monitor.set_selected_square(3, 3, verify_projection=False)
    shifted = black.copy()
    cv2.rectangle(shifted, (48, 42), (56, 48), (180, 180, 180), -1)
    monitor._check_completed(monitor._generation, detection, shifted)
    monitor._check_completed(monitor._generation, detection, shifted)

    assert not lost_messages
    assert "selected square is occupied" in status_messages[-1]

def _scene(
    board_grid: np.ndarray,
    light_grid: np.ndarray | None,
    inverted: bool = False,
) -> np.ndarray:
    reflectance = np.full((100, 100), 0.5, dtype=np.float32)
    for row in range(8):
        for column in range(8):
            quad = np.rint(
                [
                    board_grid[row, column],
                    board_grid[row, column + 1],
                    board_grid[row + 1, column + 1],
                    board_grid[row + 1, column],
                ]
            ).astype(np.int32)
            cv2.fillConvexPoly(
                reflectance,
                quad,
                0.9 if (row + column) % 2 == 0 else 0.3,
            )
    light = np.zeros((100, 100), dtype=np.float32)
    if light_grid is not None:
        light = (
            recovery_pattern_image(light_grid, (100, 100), inverted=inverted)
            .astype(np.float32)
            * 0.8
        )
    gray = np.clip((40 + light) * reflectance, 0, 255).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _recovering_monitor(
    live: BoardDetection,
    *,
    camera_changed: bool = False,
    occupied: frozenset[tuple[int, int]] = frozenset(),
) -> tuple[AlignmentMonitor, dict[str, list[object]], _Clock]:
    clock = _Clock()
    monitor = AlignmentMonitor(BoardDetector(), clock=clock)
    events: dict[str, list[object]] = {
        "requested": [],
        "completed": [],
        "failed": [],
        "checked": [],
    }
    monitor.recovery_projection_requested.connect(
        lambda pattern, _message, details: events["requested"].append(
            (pattern, details)
        )
    )
    monitor.recovery_completed.connect(
        lambda result, message, details, images: events["completed"].append(
            (result, message, details, images)
        )
    )
    monitor.recovery_failed.connect(
        lambda message, details, images: events["failed"].append(
            (message, details, images)
        )
    )
    monitor.camera_grid_checked.connect(lambda: events["checked"].append(True))
    monitor.configure(
        _result(),
        (200, 200),
        startup_verification=True,
        camera_changed=camera_changed,
        recovery_occupied_squares=occupied,
    )
    if camera_changed:
        monitor._tracker.reset(live)
    monitor._check_completed(
        monitor._generation,
        live,
        _scene(live.grid_points, None),
    )
    return monitor, events, clock


def _check_at(
    monitor: AlignmentMonitor,
    clock: _Clock,
    seconds: float,
    detection: BoardDetection,
    frame: np.ndarray,
) -> None:
    clock.now = seconds
    monitor._check_started_at = seconds
    monitor._check_completed(monitor._generation, detection, frame)


def _drive_recovery(
    monitor: AlignmentMonitor,
    clock: _Clock,
    detection: BoardDetection,
    board_grid: np.ndarray,
    light_grid: np.ndarray | None,
) -> None:
    for _step in range(8):
        state = monitor._recovery
        if state is None:
            return
        clock.now += 0.1
        monitor.recovery_projection_shown()
        _check_at(
            monitor,
            clock,
            clock.now + 0.6,
            detection,
            _scene(board_grid, light_grid, state.inverted),
        )


def test_recovery_pattern_is_distinct_under_symmetries_and_shifts() -> None:
    signs = np.where(RECOVERY_PATTERN, 1.0, -1.0)
    assert int(np.sum(RECOVERY_PATTERN)) == 32
    flipped = np.flip(signs, axis=1)
    for source in (signs, flipped):
        for turns in range(4):
            if source is signs and turns == 0:
                continue
            assert abs(np.mean(signs * np.rot90(source, turns))) <= 0.1
    for row_shift in (-1, 0, 1):
        for column_shift in (-1, 0, 1):
            if row_shift == column_shift == 0:
                continue
            rows = slice(max(row_shift, 0), 8 + min(row_shift, 0))
            columns = slice(max(column_shift, 0), 8 + min(column_shift, 0))
            shifted_rows = slice(max(-row_shift, 0), 8 + min(-row_shift, 0))
            shifted_columns = slice(
                max(-column_shift, 0),
                8 + min(-column_shift, 0),
            )
            assert abs(
                np.mean(
                    signs[rows, columns] * signs[shifted_rows, shifted_columns]
                )
            ) <= 0.1


@pytest.mark.parametrize(
    ("hypothesis", "camera_grid_offset", "projector_grid_offset"),
    [
        ("camera_moved", 6.0, 0.0),
        ("board_moved", 6.0, 6.0),
    ],
)
def test_startup_recovery_selects_the_hypothesis_matching_the_pattern(
    hypothesis: str,
    camera_grid_offset: float,
    projector_grid_offset: float,
) -> None:
    moved = _detection(offset_x=6)
    light_grid = (
        moved.grid_points
        if hypothesis == "camera_moved"
        else _detection().grid_points
    )
    monitor, events, clock = _recovering_monitor(moved)

    pattern, details = events["requested"][0]
    assert details["pattern"] == "pattern"
    assert pattern.shape == (200, 200)
    assert monitor.output_mode == "recovery"

    _drive_recovery(monitor, clock, moved, moved.grid_points, light_grid)

    assert not events["failed"]
    assert [details["pattern"] for _pattern, details in events["requested"]] == [
        "pattern",
        "inverse",
    ]
    result, _message, details, images = events["completed"][0]
    assert details["selected"]["hypothesis"] == hypothesis
    assert details["selected"]["score"] >= 0.6
    assert len(details["hypotheses"]) == 9
    assert set(images) == {
        "recovery-black",
        "recovery-pattern",
        "recovery-inverse",
        "recovery-response",
    }
    original = _detection().grid_points
    np.testing.assert_allclose(
        result.mapping.camera_grid[..., 0],
        original[..., 0] + camera_grid_offset,
    )
    np.testing.assert_allclose(
        result.mapping.projector_grid[..., 0],
        original[..., 0] + projector_grid_offset,
        atol=1e-3,
    )
    assert monitor.output_mode == "black"
    assert not monitor.startup_verification_pending


def test_recovery_ignores_frames_before_the_pattern_is_visible() -> None:
    moved = _detection(offset_x=6)
    saved = _detection().grid_points
    monitor, events, clock = _recovering_monitor(moved)
    dark = _scene(moved.grid_points, None)

    _check_at(monitor, clock, 0.2, moved, dark)
    clock.now = 0.3
    monitor.recovery_projection_shown()
    _check_at(monitor, clock, 0.6, moved, dark)

    assert len(events["requested"]) == 1
    assert not events["completed"]
    assert not events["failed"]

    _drive_recovery(monitor, clock, moved, moved.grid_points, saved)

    _result, _message, details, _images = events["completed"][0]
    assert details["selected"]["hypothesis"] == "board_moved"
    assert details["attempts"] == 1


def test_camera_change_recovery_restores_grid_indexing() -> None:
    moved = _detection(offset_x=6)
    reindexed = BoardDetection(
        grid_points=np.ascontiguousarray(
            np.rot90(moved.grid_points, 1, axes=(0, 1))
        ),
        confidence=1.0,
        reprojection_error=0.0,
        method="test",
        grid_alignment=1.0,
    )
    monitor, events, clock = _recovering_monitor(reindexed, camera_changed=True)

    _drive_recovery(
        monitor,
        clock,
        reindexed,
        moved.grid_points,
        moved.grid_points,
    )

    result, _message, details, _images = events["completed"][0]
    assert details["selected"]["hypothesis"] == "camera_moved"
    assert len(details["hypotheses"]) == 8
    np.testing.assert_allclose(result.mapping.camera_grid, moved.grid_points)
    np.testing.assert_allclose(
        result.mapping.projector_grid,
        _detection().grid_points,
    )


def test_missing_pattern_fails_after_three_attempts_without_alignment_loss() -> (
    None
):
    moved = _detection(offset_x=6)
    monitor, events, clock = _recovering_monitor(moved)
    lost: list[str] = []
    monitor.alignment_lost.connect(lost.append)

    _drive_recovery(monitor, clock, moved, moved.grid_points, None)

    assert not events["completed"]
    _message, details, _images = events["failed"][0]
    assert details["attempts"] == 3
    assert "best score" in details["last_error"]
    assert not lost
    assert monitor.output_mode == "paused"


def test_recovery_fails_when_the_pattern_is_never_displayed() -> None:
    moved = _detection(offset_x=6)
    monitor, events, clock = _recovering_monitor(moved)

    _check_at(monitor, clock, 3.5, moved, _scene(moved.grid_points, None))

    message, _details, _images = events["failed"][0]
    assert "not displayed" in message


def test_recovery_scores_only_empty_squares_unless_too_few() -> None:
    moved = _detection(offset_x=6)
    few = frozenset((row, column) for row in range(2) for column in range(8))
    _monitor, events, _clock = _recovering_monitor(moved, occupied=few)
    assert events["requested"][0][1]["scored_squares"] == 48

    many = frozenset((row, column) for row in range(7) for column in range(8))
    _monitor, events, _clock = _recovering_monitor(moved, occupied=many)
    assert events["requested"][0][1]["scored_squares"] == 64


def test_small_startup_drift_is_accepted_without_recovery() -> None:
    _monitor, events, _clock = _recovering_monitor(_detection(offset_x=1))

    assert not events["requested"]
    assert events["checked"] == [True]
