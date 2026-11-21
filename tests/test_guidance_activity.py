from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from src.chess_camera.camera_app import CameraWindow
from src.chess_camera.guidance_activity import GuidanceActivityDetector
from src.chess_camera.reference_move_tracker import ReferenceTrackerUpdate


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


def test_activity_on_board_keeps_guidance_until_scene_is_quiet() -> None:
    frame = _frame()
    changed = frame.copy()
    cv2.rectangle(changed, (430, 430), (500, 560), (20, 20, 20), -1)
    detector = _armed_detector(frame)

    assert not detector.observe(changed, _camera_grid())
    assert not detector.observe(changed, _camera_grid())
    assert detector.manipulation_active
    for _ in range(4):
        assert not detector.observe(changed, _camera_grid())
    assert detector.observe(changed, _camera_grid())


def test_reset_clears_completed_manipulation_lifecycle() -> None:
    frame = _frame()
    changed = frame.copy()
    cv2.rectangle(changed, (430, 430), (500, 560), (20, 20, 20), -1)
    detector = _armed_detector(frame)

    assert not detector.observe(changed, _camera_grid())
    assert not detector.observe(changed, _camera_grid())
    assert detector.manipulation_active

    detector.reset()

    assert detector.manipulation_active is False
    for _index in range(10):
        assert not detector.observe(frame, _camera_grid())
    for _index in range(5):
        assert not detector.observe(frame, _camera_grid())
    assert detector.manipulation_active is False


def test_activity_in_near_edge_halo_waits_for_quiet_before_blackout() -> None:
    frame = _frame()
    changed = frame.copy()
    cv2.rectangle(changed, (430, 115), (510, 185), (20, 20, 20), -1)
    detector = _armed_detector(frame)

    assert not detector.observe(changed, _camera_grid())
    assert not detector.observe(changed, _camera_grid())
    for _ in range(4):
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


def test_ignored_correction_squares_do_not_trigger_activity() -> None:
    frame = _frame()
    projected = frame.copy()
    cv2.rectangle(projected, (400, 400), (500, 500), (220, 20, 20), -1)
    cv2.rectangle(projected, (500, 400), (600, 500), (220, 20, 20), -1)
    detector = _armed_detector(frame)

    assert not detector.observe(
        projected,
        _camera_grid(),
        ((2, 2), (2, 3)),
    )
    assert not detector.observe(
        projected,
        _camera_grid(),
        ((2, 2), (2, 3)),
    )
    assert detector.manipulation_active is False


def test_correction_detector_ignores_edge_spill_but_detects_hand() -> None:
    frame = _frame()
    projected = frame.copy()
    cv2.rectangle(projected, (397, 397), (503, 503), (220, 20, 20), -1)
    cv2.rectangle(projected, (497, 397), (603, 503), (220, 20, 20), -1)
    detector = GuidanceActivityDetector(
        confirmation_observations=1,
        minimum_square_fraction=0.5,
        rolling_baseline_before_activity=True,
        require_halo_activity=True,
    )
    for _index in range(10):
        assert not detector.observe(frame, _camera_grid())

    for _index in range(3):
        assert not detector.observe(
            projected,
            _camera_grid(),
            ((2, 2), (2, 3)),
        )
    assert detector.manipulation_active is False

    hand = projected.copy()
    cv2.rectangle(hand, (950, 500), (1080, 610), (20, 20, 20), -1)
    assert not detector.observe(
        hand,
        _camera_grid(),
        ((2, 2), (2, 3)),
    )
    assert detector.manipulation_active is True


def test_rolling_correction_baseline_does_not_accumulate_exposure_drift() -> None:
    frame = _frame()
    detector = GuidanceActivityDetector(
        confirmation_observations=1,
        minimum_square_fraction=0.5,
        rolling_baseline_before_activity=True,
        require_halo_activity=True,
    )
    for _index in range(10):
        assert not detector.observe(frame, _camera_grid())

    for offset in (8, 16, 24, 32):
        drifted = frame.copy()
        drifted[500:610, 700:800] = 110 + offset
        assert not detector.observe(drifted, _camera_grid())

    assert detector.manipulation_active is False


def test_correction_detector_rejects_board_confined_light_change() -> None:
    frame = _frame()
    changed = frame.copy()
    cv2.rectangle(changed, (700, 500), (820, 620), (20, 20, 20), -1)
    detector = GuidanceActivityDetector(
        confirmation_observations=1,
        minimum_square_fraction=0.5,
        rolling_baseline_before_activity=True,
        require_halo_activity=True,
    )
    for _index in range(10):
        assert not detector.observe(frame, _camera_grid())

    assert not detector.observe(changed, _camera_grid())
    assert detector.manipulation_active is False


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
        "Projecting pending Lichess move e7-e5 until the hand leaves the "
        "board."
    )


def test_castling_guidance_projects_all_four_squares() -> None:
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
    squares = ((7, 4), (7, 6), (7, 5), (7, 7))

    CameraWindow._project_lichess_move_guidance(
        window,
        (squares[0], squares[1]),
        squares[2:],
    )

    window._board_widget.set_guidance_squares.assert_called_once_with(squares)
    window._projector_window.show_move_squares.assert_called_once_with(
        (
            (7, 4, False),
            (7, 6, False),
            (7, 5, True),
            (7, 7, True),
        )
    )


def test_en_passant_guidance_projects_all_three_squares() -> None:
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
    squares = ((3, 4), (2, 3), (3, 3))

    CameraWindow._project_lichess_move_guidance(
        window,
        (squares[0], squares[1]),
        squares[2:],
    )

    window._board_widget.set_guidance_squares.assert_called_once_with(squares)
    window._projector_window.show_move_squares.assert_called_once_with(
        (
            (3, 4, False),
            (2, 3, False),
            (3, 3, True),
        )
    )


def test_settled_manipulation_blacks_projector_before_observation() -> None:
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
    assert window._lichess_guidance_reprojection_pending is True
    window._projector_window.show_black.assert_called_once_with()
    window._alignment_monitor.set_black.assert_called_once_with()
    window._reference_move_tracker.reset_activity.assert_called_once_with()
    assert single_shot.call_args.args[0] == 1000
    single_shot.call_args.args[1]()
    window._finish_lichess_guidance_blackout.assert_called_once_with(4)


def test_active_manipulation_keeps_opponent_guidance_projected() -> None:
    detector = MagicMock()
    detector.observe.return_value = False
    detector.manipulation_active = True
    window = SimpleNamespace(
        _latest_frame=object(),
        _calibration_result=SimpleNamespace(
            mapping=SimpleNamespace(camera_grid=object())
        ),
        _lichess_guidance_projection_active=True,
        _guidance_activity_detector=detector,
        _projector_status=MagicMock(),
        _begin_lichess_guidance_observation=MagicMock(),
    )

    CameraWindow._observe_lichess_guidance_activity(window)

    window._begin_lichess_guidance_observation.assert_not_called()
    window._projector_status.setText.assert_called_once_with(
        "Opponent move in progress; guidance remains projected until the hand "
        "leaves the board."
    )


def test_offline_accepted_projection_returns_when_hand_moves_nothing() -> None:
    move = ((6, 4), (4, 4))
    window = SimpleNamespace(
        _offline_accepted_reprojection_pending=True,
        _offline_accepted_no_change_count=0,
        _offline_accepted_move=move,
        _game_mode="offline-game",
        _project_offline_accepted_move=MagicMock(),
        _occupancy_status=MagicMock(),
    )

    CameraWindow._restore_offline_accepted_projection_after_no_change(
        window,
        ReferenceTrackerUpdate(
            "watching",
            None,
            "Board matches the accepted reference.",
        ),
    )

    assert window._offline_accepted_reprojection_pending is True
    window._project_offline_accepted_move.assert_not_called()

    CameraWindow._restore_offline_accepted_projection_after_no_change(
        window,
        ReferenceTrackerUpdate(
            "watching",
            None,
            "Board matches the accepted reference.",
        ),
    )

    assert window._offline_accepted_reprojection_pending is False
    window._project_offline_accepted_move.assert_called_once_with(move)
    window._occupancy_status.setText.assert_called_once_with(
        "No physical move detected; accepted-move illumination restored."
    )


def test_offline_hand_entry_requests_accepted_move_reprojection() -> None:
    detector = MagicMock()
    detector.manipulation_active = True
    window = SimpleNamespace(
        _latest_frame=object(),
        _calibration_result=SimpleNamespace(
            mapping=SimpleNamespace(camera_grid=object())
        ),
        _offline_accepted_projection_active=True,
        _offline_accepted_reprojection_pending=False,
        _offline_accepted_no_change_count=1,
        _offline_guidance_activity_detector=detector,
        _projector_window=SimpleNamespace(
            output_mode="move",
            show_black=MagicMock(),
        ),
        _alignment_monitor=MagicMock(),
        _reference_move_tracker=MagicMock(),
        _projector_status=MagicMock(),
    )

    CameraWindow._observe_offline_accepted_projection(window)

    assert window._offline_accepted_projection_active is False
    assert window._offline_accepted_reprojection_pending is True
    assert window._offline_accepted_no_change_count == 0
    window._projector_window.show_black.assert_called_once_with()
    window._reference_move_tracker.reset_activity.assert_called_once_with()


def test_unchanged_board_restores_pending_lichess_guidance() -> None:
    window = SimpleNamespace(
        _lichess_guidance_reprojection_pending=True,
        _lichess_guidance_projection_active=False,
        _lichess_game_controller=SimpleNamespace(
            expected_signature=object()
        ),
        _occupancy_status=MagicMock(),
    )

    def restore_projection() -> None:
        window._lichess_guidance_projection_active = True

    window._update_lichess_synchronization_projection = MagicMock(
        side_effect=restore_projection
    )

    CameraWindow._restore_pending_lichess_guidance_after_no_change(
        window,
        ReferenceTrackerUpdate(
            "watching",
            None,
            "Board matches the accepted reference.",
        ),
    )

    window._update_lichess_synchronization_projection.assert_called_once_with()
    window._occupancy_status.setText.assert_called_once_with(
        "No physical move detected; pending Lichess guidance restored."
    )


def test_guidance_activity_precedes_black_output_gate() -> None:
    window = SimpleNamespace(
        _latest_frame=object(),
        _calibration_result=object(),
        _calibrator=SimpleNamespace(active=False),
        _camera_only_calibration=False,
        _camera_geometry_ready=True,
        _activate_offline_after_reset=False,
        _reset_to_start_requested=False,
        _pending_promotion_selection=None,
        _illegal_correction_projection_active=False,
        _illegal_correction_blackout_pending=False,
        _offline_accepted_projection_active=False,
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
        _offline_accepted_projection_active=False,
        _offline_accepted_reprojection_pending=False,
        _offline_guidance_activity_detector=MagicMock(),
        _illegal_correction_activity_detector=MagicMock(),
        _illegal_correction_generation=0,
        _illegal_correction_physical_squares=(),
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


def test_illegal_correction_projects_both_squares_without_timeout() -> None:
    window = SimpleNamespace(
        _illegal_correction_projection_active=False,
        _illegal_correction_blackout_pending=False,
        _illegal_correction_generation=0,
        _illegal_correction_physical_squares=(),
        _illegal_correction_activity_logged=False,
        _illegal_correction_last_projected_move=None,
        _illegal_correction_activity_detector=MagicMock(),
        _camera_only_calibration=False,
        _calibration_result=object(),
        _open_projector=MagicMock(return_value=True),
        _physical_board_square=lambda row, column: (row, column),
        _projector_window=MagicMock(),
        _alignment_monitor=MagicMock(),
        _projector_status=MagicMock(),
        _record_runtime_diagnostic=MagicMock(),
    )

    with patch(
        "src.chess_camera.camera_app.QTimer.singleShot"
    ) as single_shot:
        CameraWindow._project_illegal_move_correction(
            window,
            ((6, 2), (5, 2)),
        )

    window._projector_window.show_correction.assert_called_once_with(
        (6, 2),
        (5, 2),
    )
    assert window._illegal_correction_projection_active is True
    single_shot.assert_not_called()
    window._record_runtime_diagnostic.assert_called_once()
    assert (
        window._record_runtime_diagnostic.call_args.args[0]
        == "illegal-correction-projected"
    )
    assert window._record_runtime_diagnostic.call_args.kwargs["details"] == {
        "generation": 1,
        "logical_move": ["c2", "c3"],
        "physical_squares": [[6, 2], [5, 2]],
    }

    window._illegal_correction_projection_active = False
    CameraWindow._project_illegal_move_correction(
        window,
        ((6, 2), (5, 2)),
    )

    assert window._record_runtime_diagnostic.call_count == 2
    assert (
        window._record_runtime_diagnostic.call_args.args[0]
        == "illegal-correction-reprojected"
    )


def test_illegal_correction_remains_while_hand_is_present() -> None:
    detector = MagicMock()
    detector.observe.return_value = False
    detector.manipulation_active = True
    window = SimpleNamespace(
        _latest_frame=object(),
        _calibration_result=SimpleNamespace(
            mapping=SimpleNamespace(camera_grid=object())
        ),
        _illegal_correction_projection_active=True,
        _illegal_correction_activity_detector=detector,
        _illegal_correction_physical_squares=((6, 2), (5, 2)),
        _illegal_correction_activity_logged=False,
        _illegal_correction_generation=3,
        _begin_illegal_correction_observation=MagicMock(),
        _projector_status=MagicMock(),
        _record_runtime_diagnostic=MagicMock(),
    )

    CameraWindow._observe_illegal_correction_activity(window)
    CameraWindow._observe_illegal_correction_activity(window)

    window._begin_illegal_correction_observation.assert_not_called()
    window._projector_status.setText.assert_called_with(
        "Correction in progress; both red squares remain projected until "
        "the hand leaves the board."
    )
    window._record_runtime_diagnostic.assert_called_once()
    assert (
        window._record_runtime_diagnostic.call_args.args[0]
        == "illegal-correction-hand-entered"
    )


def test_illegal_correction_blackout_starts_after_hand_leaves() -> None:
    callback = None

    def capture_timer(milliseconds: int, pending_callback: object) -> None:
        nonlocal callback
        assert milliseconds == 1000
        callback = pending_callback

    window = SimpleNamespace(
        _illegal_correction_projection_active=True,
        _illegal_correction_blackout_pending=False,
        _illegal_correction_generation=4,
        _illegal_correction_activity_detector=MagicMock(),
        _projector_window=MagicMock(),
        _alignment_monitor=MagicMock(),
        _reference_move_tracker=MagicMock(),
        _projector_status=MagicMock(),
        _finish_illegal_correction_blackout=MagicMock(),
        _record_runtime_diagnostic=MagicMock(),
    )

    with patch(
        "src.chess_camera.camera_app.QTimer.singleShot",
        side_effect=capture_timer,
    ):
        CameraWindow._begin_illegal_correction_observation(window)

    assert window._illegal_correction_projection_active is False
    assert window._illegal_correction_blackout_pending is True
    window._projector_window.show_black.assert_called_once_with()
    assert (
        window._record_runtime_diagnostic.call_args.args[0]
        == "illegal-correction-blackout-started"
    )
    assert callback is not None
