from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from src.chess_camera.camera_app import CameraWindow
from src.chess_camera.guidance_activity import GuidanceActivityDetector


def _camera_grid() -> np.ndarray:
    coordinates = np.linspace(200, 1000, 9, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(coordinates, coordinates)
    return np.stack((x_grid, y_grid), axis=2)


def _frame() -> np.ndarray:
    return np.full((1200, 1200, 3), 110, dtype=np.uint8)


def _armed_detector(frame: np.ndarray) -> GuidanceActivityDetector:
    detector = GuidanceActivityDetector()
    for _ in range(10):
        assert not detector.observe(frame, _camera_grid())
    return detector


def test_static_guidance_does_not_trigger_activity() -> None:
    frame = _frame()
    detector = _armed_detector(frame)

    assert not detector.observe(frame, _camera_grid())
    assert not detector.observe(frame, _camera_grid())


def test_projector_transition_during_one_second_arming_is_ignored() -> None:
    frame = _frame()
    projected = frame.copy()
    cv2.rectangle(
        projected,
        (400, 400),
        (600, 600),
        (220, 220, 220),
        -1,
    )
    detector = GuidanceActivityDetector()

    for _ in range(3):
        assert not detector.observe(frame, _camera_grid())
    for _ in range(7):
        assert not detector.observe(projected, _camera_grid())

    assert not detector.observe(projected, _camera_grid())
    assert not detector.observe(projected, _camera_grid())


def test_activity_on_board_requires_consecutive_observations() -> None:
    frame = _frame()
    changed = frame.copy()
    cv2.rectangle(changed, (430, 430), (500, 560), (20, 20, 20), -1)
    detector = _armed_detector(frame)

    assert not detector.observe(changed, _camera_grid())
    assert detector.observe(changed, _camera_grid())


def test_activity_in_near_edge_halo_triggers_blackout_intent() -> None:
    frame = _frame()
    changed = frame.copy()
    cv2.rectangle(changed, (430, 115), (510, 185), (20, 20, 20), -1)
    detector = _armed_detector(frame)

    assert not detector.observe(changed, _camera_grid())
    assert detector.observe(changed, _camera_grid())


def test_activity_outside_board_halo_is_ignored() -> None:
    frame = _frame()
    changed = frame.copy()
    cv2.rectangle(changed, (10, 10), (80, 90), (20, 20, 20), -1)
    detector = _armed_detector(frame)

    assert not detector.observe(changed, _camera_grid())
    assert not detector.observe(changed, _camera_grid())


def test_uniform_brightness_change_is_normalized() -> None:
    frame = _frame()
    brighter = cv2.convertScaleAbs(frame, alpha=1.0, beta=24)
    detector = _armed_detector(frame)

    assert not detector.observe(brighter, _camera_grid())
    assert not detector.observe(brighter, _camera_grid())


def test_pending_lichess_guidance_has_no_fixed_timeout() -> None:
    window = SimpleNamespace(
        _camera_only_calibration=False,
        _calibration_result=object(),
        _open_projector=lambda: True,
        _physical_board_square=lambda row, column: (row, column),
        _projector_window=MagicMock(),
        _board_widget=MagicMock(),
        _alignment_monitor=MagicMock(),
        _guidance_activity_detector=MagicMock(),
        _projector_status=MagicMock(),
        _lichess_guidance_generation=0,
        _lichess_guidance_projection_active=False,
        _lichess_guidance_blackout_pending=False,
    )

    with patch(
        "src.chess_camera.camera_app.QTimer.singleShot"
    ) as single_shot:
        CameraWindow._project_lichess_move_guidance(
            window,
            ((1, 4), (3, 4)),
        )

    single_shot.assert_not_called()
    assert window._lichess_guidance_projection_active is True
    window._board_widget.set_illegal_source.assert_called_once_with(None)
    window._board_widget.set_reconciliation_squares.assert_called_once_with(())
    window._board_widget.set_guidance_move.assert_called_once_with(
        ((1, 4), (3, 4))
    )
    window._guidance_activity_detector.reset.assert_called_once_with()
    window._projector_status.setText.assert_called_once_with(
        "Projecting pending Lichess move e7-e5 until hand movement is "
        "detected."
    )


def test_hand_activity_blacks_projector_and_waits_for_settling() -> None:
    window = SimpleNamespace(
        _lichess_guidance_projection_active=True,
        _lichess_guidance_blackout_pending=False,
        _lichess_guidance_generation=4,
        _guidance_activity_detector=MagicMock(),
        _projector_window=MagicMock(),
        _alignment_monitor=MagicMock(),
        _reference_move_tracker=MagicMock(),
        _projector_status=MagicMock(),
        _finish_lichess_guidance_blackout=MagicMock(),
    )

    with patch(
        "src.chess_camera.camera_app.QTimer.singleShot"
    ) as single_shot:
        CameraWindow._begin_lichess_guidance_observation(window)

    assert window._lichess_guidance_projection_active is False
    assert window._lichess_guidance_blackout_pending is True
    window._projector_window.show_black.assert_called_once_with()
    window._alignment_monitor.set_black.assert_called_once_with()
    window._reference_move_tracker.reset_activity.assert_called_once_with()
    assert single_shot.call_args.args[0] == 1000
    single_shot.call_args.args[1]()
    window._finish_lichess_guidance_blackout.assert_called_once_with(4)


def test_guidance_activity_precedes_black_output_gate() -> None:
    window = SimpleNamespace(
        _latest_frame=object(),
        _calibration_result=object(),
        _calibrator=SimpleNamespace(active=False),
        _camera_only_calibration=False,
        _camera_geometry_ready=True,
        _lichess_guidance_projection_active=True,
        _lichess_guidance_blackout_pending=False,
        _observe_lichess_guidance_activity=MagicMock(),
        _alignment_monitor=SimpleNamespace(output_mode="move"),
    )

    CameraWindow._update_occupancy(window)

    window._observe_lichess_guidance_activity.assert_called_once_with()


def test_black_button_starts_pending_move_observation() -> None:
    window = SimpleNamespace(
        _lichess_guidance_projection_active=True,
        _calibrator=SimpleNamespace(active=False),
        _open_projector=MagicMock(return_value=True),
        _suspend_selected_square=MagicMock(),
        _begin_lichess_guidance_observation=MagicMock(),
        _projector_status=MagicMock(),
    )

    CameraWindow._black_projector(window)

    window._open_projector.assert_called_once_with()
    window._suspend_selected_square.assert_called_once_with()
    window._begin_lichess_guidance_observation.assert_called_once_with()
