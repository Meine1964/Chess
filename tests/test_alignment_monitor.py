import os

import cv2
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from src.chess_camera.alignment_monitor import AlignmentMonitor
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


def test_camera_only_monitor_updates_camera_grid_without_projector() -> None:
    monitor = AlignmentMonitor(BoardDetector())
    updated_mappings: list[object] = []
    checked: list[bool] = []
    monitor.mapping_updated.connect(
        lambda mapping, _displacement: updated_mappings.append(mapping)
    )
    monitor.camera_grid_checked.connect(lambda: checked.append(True))
    result = _result()
    monitor.configure(result, None)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    moved = _detection(offset_x=8)

    monitor._check_completed(monitor._generation, moved, frame)

    assert len(updated_mappings) == 1
    assert checked == [True]
    np.testing.assert_allclose(
        updated_mappings[0].camera_grid,
        moved.grid_points,
    )
    np.testing.assert_allclose(
        updated_mappings[0].projector_grid,
        result.mapping.projector_grid,
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
