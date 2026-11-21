from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from src.chess_camera.camera_app import CameraWindow
from src.chess_camera.guidance_activity import (
    BlackoutSettlingDetector,
    BlackoutSettlingUpdate,
    BoardEdgeActivityDetector,
    GuidanceActivityDetector,
)
from src.chess_camera.legal_game import LegalGame
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


def test_board_edge_indicator_requires_activity_across_boundary() -> None:
    frame = _frame()

    def armed_detector() -> BoardEdgeActivityDetector:
        detector = BoardEdgeActivityDetector(
            arming_observations=2,
            confirmation_observations=1,
            quiet_observations=1,
        )
        assert not detector.observe(frame, _camera_grid())
        assert not detector.observe(frame, _camera_grid())
        return detector

    halo_only = frame.copy()
    cv2.rectangle(halo_only, (350, 100), (550, 160), (20, 20, 20), -1)
    detector = armed_detector()
    assert not detector.observe(halo_only, _camera_grid())
    assert not detector.observe(halo_only, _camera_grid())

    board_only = frame.copy()
    cv2.rectangle(board_only, (350, 240), (550, 340), (20, 20, 20), -1)
    detector = armed_detector()
    assert not detector.observe(board_only, _camera_grid())
    assert not detector.observe(board_only, _camera_grid())

    crossing = frame.copy()
    cv2.rectangle(crossing, (350, 100), (550, 260), (20, 20, 20), -1)
    detector = armed_detector()
    assert detector.observe(crossing, _camera_grid())
    assert detector.active
    assert detector.crossing_detected
    assert detector.largest_component_pixels > 0
    assert detector.largest_component_inside_pixels >= 600
    assert detector.largest_component_outside_pixels >= 600
    assert detector.largest_component_outer_halo_pixels >= 60
    assert detector.largest_component_bounds is not None

    completed_edge_change = frame.copy()
    cv2.rectangle(
        completed_edge_change,
        (350, 175),
        (550, 260),
        (20, 20, 20),
        -1,
    )
    assert not detector.observe(completed_edge_change, _camera_grid())
    assert not detector.active
    assert not detector.crossing_detected
    assert detector.largest_component_inside_pixels >= 600
    assert detector.largest_component_outside_pixels >= 600
    assert detector.largest_component_outer_halo_pixels == 0
    assert detector.largest_component_bounds is not None

    assert not detector.observe(completed_edge_change, _camera_grid())
    assert detector.largest_component_pixels == 0


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


def test_activity_keeps_guidance_until_obstruction_leaves() -> None:
    frame = _frame()
    changed = frame.copy()
    cv2.rectangle(changed, (430, 430), (500, 560), (20, 20, 20), -1)
    detector = _armed_detector(frame)

    assert not detector.observe(changed, _camera_grid())
    assert not detector.observe(changed, _camera_grid())
    assert detector.manipulation_active
    for _ in range(4):
        assert not detector.observe(changed, _camera_grid())
    for _ in range(4):
        assert not detector.observe(frame, _camera_grid())
    assert detector.observe(frame, _camera_grid())


def test_lichess_guidance_ignores_projection_drift_and_waits_for_hand_exit() -> None:
    frame = _frame()
    detector = GuidanceActivityDetector(
        rolling_baseline_before_activity=True,
        require_halo_activity=True,
    )
    for _index in range(10):
        assert not detector.observe(frame, _camera_grid(), ((2, 2), (2, 3)))

    drifted = frame.copy()
    cv2.rectangle(drifted, (400, 400), (600, 500), (190, 190, 190), -1)
    for _index in range(3):
        assert not detector.observe(
            drifted,
            _camera_grid(),
            ((2, 2), (2, 3)),
        )
    assert detector.manipulation_active is False

    hand_entering = drifted.copy()
    cv2.rectangle(
        hand_entering,
        (350, 100),
        (550, 220),
        (20, 20, 20),
        -1,
    )
    hand_over_board = drifted.copy()
    cv2.rectangle(
        hand_over_board,
        (450, 100),
        (650, 220),
        (20, 20, 20),
        -1,
    )
    assert not detector.observe(
        hand_entering,
        _camera_grid(),
        ((2, 2), (2, 3)),
    )
    assert not detector.observe(
        hand_over_board,
        _camera_grid(),
        ((2, 2), (2, 3)),
    )
    assert detector.manipulation_active

    for _index in range(5):
        assert not detector.observe(
            hand_over_board,
            _camera_grid(),
            ((2, 2), (2, 3)),
        )
    for _index in range(4):
        assert not detector.observe(
            drifted,
            _camera_grid(),
            ((2, 2), (2, 3)),
        )
    assert detector.observe(
        drifted,
        _camera_grid(),
        ((2, 2), (2, 3)),
    )


def test_lichess_guidance_ignores_moved_piece_spill_after_hand_exit() -> None:
    frame = _frame()
    moved_piece = frame.copy()
    cv2.rectangle(
        moved_piece,
        (290, 190),
        (410, 310),
        (35, 35, 35),
        -1,
    )
    detector = GuidanceActivityDetector(
        rolling_baseline_before_activity=True,
        require_halo_activity=True,
    )
    ignored_squares = ((0, 1),)
    for _index in range(10):
        assert not detector.observe(
            frame,
            _camera_grid(),
            ignored_squares,
        )

    hand = moved_piece.copy()
    cv2.rectangle(hand, (650, 100), (850, 320), (20, 20, 20), -1)
    assert not detector.observe(hand, _camera_grid(), ignored_squares)
    assert not detector.observe(hand, _camera_grid(), ignored_squares)
    assert detector.manipulation_active

    for _index in range(4):
        assert not detector.observe(
            moved_piece,
            _camera_grid(),
            ignored_squares,
        )
    assert detector.observe(
        moved_piece,
        _camera_grid(),
        ignored_squares,
    )


def test_lichess_guidance_clears_adjacent_edge_square_spill() -> None:
    frame = _frame()
    moved_piece = frame.copy()
    cv2.rectangle(
        moved_piece,
        (570, 170),
        (830, 330),
        (35, 35, 35),
        -1,
    )
    detector = GuidanceActivityDetector(
        rolling_baseline_before_activity=True,
        require_halo_activity=True,
    )
    ignored_squares = ((0, 4), (0, 5))
    for _index in range(10):
        assert not detector.observe(
            frame,
            _camera_grid(),
            ignored_squares,
        )
    assert detector.lifecycle_state == "arming"

    hand = moved_piece.copy()
    cv2.rectangle(hand, (300, 100), (520, 340), (20, 20, 20), -1)
    assert not detector.observe(hand, _camera_grid(), ignored_squares)
    assert detector.lifecycle_state == "entering"
    assert not detector.observe(hand, _camera_grid(), ignored_squares)
    assert detector.lifecycle_state == "manipulating"

    for _index in range(4):
        assert not detector.observe(
            moved_piece,
            _camera_grid(),
            ignored_squares,
        )
    assert detector.lifecycle_state == "exiting"
    assert detector.observe(
        moved_piece,
        _camera_grid(),
        ignored_squares,
    )
    assert detector.lifecycle_state == "departed"


def test_lichess_guidance_clears_bounded_edge_piece_spill() -> None:
    frame = _frame()
    moved_piece = frame.copy()
    cv2.rectangle(
        moved_piece,
        (755, 155),
        (945, 345),
        (35, 35, 35),
        -1,
    )
    cv2.rectangle(
        moved_piece,
        (670, 370),
        (830, 530),
        (35, 35, 35),
        -1,
    )
    detector = GuidanceActivityDetector(
        rolling_baseline_before_activity=True,
        require_halo_activity=True,
    )
    ignored_squares = ((0, 6), (2, 5))
    for _index in range(10):
        assert not detector.observe(
            frame,
            _camera_grid(),
            ignored_squares,
        )

    hand = moved_piece.copy()
    cv2.rectangle(hand, (300, 100), (520, 340), (20, 20, 20), -1)
    assert not detector.observe(hand, _camera_grid(), ignored_squares)
    assert not detector.observe(hand, _camera_grid(), ignored_squares)
    assert detector.lifecycle_state == "manipulating"

    for _index in range(4):
        assert not detector.observe(
            moved_piece,
            _camera_grid(),
            ignored_squares,
        )
    assert detector.observe(
        moved_piece,
        _camera_grid(),
        ignored_squares,
    )
    assert detector.lifecycle_state == "departed"


def test_lichess_guidance_ignores_external_halo_after_entry() -> None:
    frame = _frame()
    detector = GuidanceActivityDetector(
        rolling_baseline_before_activity=True,
        require_halo_activity=True,
    )
    ignored_squares = ((0, 4), (0, 5))
    for _index in range(10):
        assert not detector.observe(
            frame,
            _camera_grid(),
            ignored_squares,
        )

    hand = frame.copy()
    cv2.rectangle(hand, (300, 100), (520, 340), (20, 20, 20), -1)
    assert not detector.observe(hand, _camera_grid(), ignored_squares)
    assert not detector.observe(hand, _camera_grid(), ignored_squares)

    unexpected = frame.copy()
    cv2.rectangle(
        unexpected,
        (880, 110),
        (1030, 190),
        (20, 20, 20),
        -1,
    )
    for _index in range(4):
        assert not detector.observe(
            unexpected,
            _camera_grid(),
            ignored_squares,
        )
    assert detector.observe(
        unexpected,
        _camera_grid(),
        ignored_squares,
    )
    assert detector.lifecycle_state == "departed"


def test_lichess_guidance_requires_board_boundary_crossing() -> None:
    frame = _frame()
    detector = GuidanceActivityDetector(
        rolling_baseline_before_activity=True,
        require_halo_activity=True,
    )
    for _index in range(10):
        assert not detector.observe(frame, _camera_grid())

    outside_only = frame.copy()
    cv2.rectangle(
        outside_only,
        (420, 110),
        (620, 190),
        (20, 20, 20),
        -1,
    )
    for _index in range(4):
        assert not detector.observe(outside_only, _camera_grid())
    assert detector.lifecycle_state == "arming"
    assert detector.manipulation_active is False

    crossing = frame.copy()
    cv2.rectangle(
        crossing,
        (420, 110),
        (620, 240),
        (20, 20, 20),
        -1,
    )
    assert not detector.observe(crossing, _camera_grid())
    assert not detector.observe(crossing, _camera_grid())
    assert detector.lifecycle_state == "manipulating"


def test_lichess_guidance_clears_after_smooth_exposure_drift() -> None:
    frame = _frame()
    height, width = frame.shape[:2]
    horizontal = np.linspace(-45.0, 45.0, width, dtype=np.float32)
    exposure_offset = np.broadcast_to(horizontal, (height, width))
    exposed = np.clip(
        frame.astype(np.float32) + exposure_offset[:, :, None],
        0,
        255,
    ).astype(np.uint8)
    detector = GuidanceActivityDetector(
        rolling_baseline_before_activity=True,
        require_halo_activity=True,
    )
    ignored_squares = ((1, 3), (3, 3))
    for _index in range(10):
        assert not detector.observe(
            frame,
            _camera_grid(),
            ignored_squares,
        )

    hand = exposed.copy()
    cv2.rectangle(hand, (300, 100), (520, 340), (20, 20, 20), -1)
    assert not detector.observe(hand, _camera_grid(), ignored_squares)
    assert not detector.observe(hand, _camera_grid(), ignored_squares)
    assert detector.lifecycle_state == "manipulating"

    for _index in range(4):
        assert not detector.observe(
            exposed,
            _camera_grid(),
            ignored_squares,
        )
    assert detector.observe(
        exposed,
        _camera_grid(),
        ignored_squares,
    )
    assert detector.lifecycle_state == "departed"


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
    for _ in range(4):
        assert not detector.observe(frame, _camera_grid())
    assert detector.observe(frame, _camera_grid())


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


def test_blackout_settling_finishes_after_minimum_and_three_stable_frames() -> None:
    now = 0.0
    detector = BlackoutSettlingDetector(clock=lambda: now)
    frame = _frame()
    detector.reset()

    now = 0.3
    assert detector.observe(frame, _camera_grid()).state == "waiting"
    now = 0.4
    first = detector.observe(frame, _camera_grid())
    now = 0.5
    second = detector.observe(frame, _camera_grid())
    now = 0.6
    third = detector.observe(frame, _camera_grid())

    assert first.stable_observations == 1
    assert second.stable_observations == 2
    assert third.state == "stable"
    assert third.stable_observations == 3


def test_blackout_instability_resets_stable_observations() -> None:
    now = 0.0
    detector = BlackoutSettlingDetector(clock=lambda: now)
    frame = _frame()
    changed = frame.copy()
    changed[400:600, 400:600] = 180
    detector.reset()

    now = 0.4
    detector.observe(frame, _camera_grid())
    now = 0.5
    stable = detector.observe(frame, _camera_grid())
    now = 0.6
    unstable = detector.observe(changed, _camera_grid())

    assert stable.stable_observations == 1
    assert unstable.state == "waiting"
    assert unstable.stable_observations == 0


def test_blackout_settling_uses_one_second_timeout() -> None:
    now = 0.0
    detector = BlackoutSettlingDetector(clock=lambda: now)
    detector.reset()

    now = 1.0
    update = detector.observe(_frame(), _camera_grid())

    assert update.state == "timeout"
    assert update.elapsed_milliseconds == 1000


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
        _board_orientation_needs_detection=False,
        _piece_side_orientation_needs_detection=False,
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
        _lichess_guidance_reprojection_pending=False,
        _record_runtime_diagnostic=MagicMock(),
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
    assert window._lichess_guidance_squares == ((1, 4), (3, 4))
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
    assert (
        window._record_runtime_diagnostic.call_args.args[0]
        == "lichess-guidance-projected"
    )


def test_pending_lichess_guidance_waits_for_complete_orientation() -> None:
    window = SimpleNamespace(
        _board_orientation_needs_detection=False,
        _piece_side_orientation_needs_detection=True,
        _camera_only_calibration=False,
        _calibration_result=object(),
        _open_projector=MagicMock(return_value=True),
        _projector_window=MagicMock(),
        _board_widget=MagicMock(),
        _alignment_monitor=MagicMock(),
        _cancel_lichess_guidance_activity=MagicMock(),
        _projector_status=MagicMock(),
    )

    projected = CameraWindow._project_lichess_move_guidance(
        window,
        ((0, 5), (0, 3)),
    )

    assert projected is False
    window._board_widget.set_guidance_move.assert_called_once_with(
        ((0, 5), (0, 3))
    )
    window._open_projector.assert_not_called()
    window._cancel_lichess_guidance_activity.assert_called_once_with()
    window._projector_window.show_black.assert_called_once_with()
    window._alignment_monitor.set_black.assert_called_once_with()
    assert (
        "White-side orientation"
        in window._projector_status.setText.call_args.args[0]
    )


def test_pending_lichess_guidance_uses_complete_two_turn_orientation() -> None:
    window = SimpleNamespace(
        _board_orientation_needs_detection=False,
        _piece_side_orientation_needs_detection=False,
        _board_quarter_turns=2,
        _camera_only_calibration=False,
        _calibration_result=object(),
        _open_projector=lambda: True,
        _projector_window=MagicMock(),
        _board_widget=MagicMock(),
        _alignment_monitor=MagicMock(),
        _guidance_activity_detector=MagicMock(),
        _projector_status=MagicMock(),
        _lichess_guidance_generation=0,
        _lichess_guidance_projection_active=False,
        _lichess_guidance_blackout_pending=False,
        _lichess_guidance_reprojection_pending=False,
        _record_runtime_diagnostic=MagicMock(),
    )
    window._physical_board_square = lambda row, column: (
        CameraWindow._physical_board_square(window, row, column)
    )

    projected = CameraWindow._project_lichess_move_guidance(
        window,
        ((0, 5), (0, 3)),
    )

    assert projected is True
    window._projector_window.show_move.assert_called_once_with(
        (7, 2),
        (7, 4),
        source_is_light=False,
        destination_is_light=False,
    )


def test_castling_guidance_projects_only_king_move() -> None:
    signature = LegalGame(
        "r3k2r/8/8/8/8/8/8/4K3 b kq - 0 1"
    ).legal_move_signature("e8g8")
    window = SimpleNamespace(
        _lichess_game_controller=SimpleNamespace(
            expected_signature=signature
        ),
        _last_projected_lichess_uci=None,
        _board_widget=MagicMock(),
        _project_lichess_move_guidance=MagicMock(),
    )

    CameraWindow._update_lichess_synchronization_projection(window)

    window._project_lichess_move_guidance.assert_called_once_with(
        (signature.source, signature.destination),
        (),
    )


def test_failed_guidance_projection_remains_eligible_for_refresh() -> None:
    signature = LegalGame().legal_move_signature("e2e4")
    window = SimpleNamespace(
        _lichess_game_controller=SimpleNamespace(
            expected_signature=signature
        ),
        _last_projected_lichess_uci="stale",
        _board_widget=MagicMock(),
        _cancel_lichess_guidance_activity=MagicMock(),
        _project_lichess_move_guidance=MagicMock(return_value=False),
    )

    CameraWindow._update_lichess_synchronization_projection(window)

    assert window._last_projected_lichess_uci is None


def test_en_passant_guidance_projects_all_three_squares() -> None:
    window = SimpleNamespace(
        _board_orientation_needs_detection=False,
        _piece_side_orientation_needs_detection=False,
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
        _lichess_guidance_reprojection_pending=False,
        _record_runtime_diagnostic=MagicMock(),
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
        _lichess_blackout_settling_detector=MagicMock(),
        _reference_move_tracker=MagicMock(),
        _projector_status=MagicMock(),
        _record_runtime_diagnostic=MagicMock(),
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
    window._lichess_blackout_settling_detector.reset.assert_called_once_with()
    window._reference_move_tracker.reset_activity.assert_called_once_with()
    assert single_shot.call_args.args[0] == 1000
    single_shot.call_args.args[1]()
    window._finish_lichess_guidance_blackout.assert_called_once_with(
        4,
        "fallback",
    )


def test_stable_blackout_observations_finish_guidance_early() -> None:
    detector = MagicMock()
    detector.observe.return_value = BlackoutSettlingUpdate(
        "stable",
        3,
        3,
        620,
    )
    window = SimpleNamespace(
        _latest_frame=object(),
        _calibration_result=SimpleNamespace(
            mapping=SimpleNamespace(camera_grid=object())
        ),
        _lichess_guidance_blackout_pending=True,
        _lichess_guidance_generation=7,
        _lichess_blackout_settling_detector=detector,
        _finish_lichess_guidance_blackout=MagicMock(),
        _projector_status=MagicMock(),
        _occupancy_status=MagicMock(),
    )

    CameraWindow._observe_lichess_guidance_blackout(window)

    window._finish_lichess_guidance_blackout.assert_called_once_with(
        7,
        "stable",
    )


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
        _lichess_guidance_squares=((1, 4), (3, 4)),
        _lichess_guidance_activity_logged=False,
        _lichess_guidance_generation=3,
        _guidance_activity_detector=detector,
        _projector_status=MagicMock(),
        _record_runtime_diagnostic=MagicMock(),
        _begin_lichess_guidance_observation=MagicMock(),
    )

    CameraWindow._observe_lichess_guidance_activity(window)
    CameraWindow._observe_lichess_guidance_activity(window)

    window._begin_lichess_guidance_observation.assert_not_called()
    detector.observe.assert_called_with(
        window._latest_frame,
        window._calibration_result.mapping.camera_grid,
        ((1, 4), (3, 4)),
    )
    assert detector.observe.call_count == 2
    window._projector_status.setText.assert_called_with(
        "Opponent move in progress; guidance remains projected until the hand "
        "leaves the board."
    )
    window._record_runtime_diagnostic.assert_called_once()


def test_offline_move_marking_uses_fixed_one_second_intervals() -> None:
    move = ((6, 4), (4, 4))
    window = SimpleNamespace(
        _board_orientation_needs_detection=False,
        _piece_side_orientation_needs_detection=False,
        _camera_only_calibration=False,
        _calibration_result=object(),
        _open_projector=MagicMock(return_value=True),
        _physical_board_square=lambda row, column: (row, column),
        _projector_window=MagicMock(),
        _offline_move_marking_generation=0,
        _offline_move_marking_active=False,
        _offline_move_marking_blackout_pending=False,
        _alignment_monitor=MagicMock(),
        _reference_move_tracker=MagicMock(),
        _projector_status=MagicMock(),
    )
    window._finish_offline_move_marking = lambda generation: (
        CameraWindow._finish_offline_move_marking(window, generation)
    )
    window._finish_offline_move_marking_blackout = lambda generation: (
        CameraWindow._finish_offline_move_marking_blackout(
            window,
            generation,
        )
    )

    with patch(
        "src.chess_camera.camera_app.QTimer.singleShot"
    ) as single_shot:
        CameraWindow._project_offline_accepted_move(window, move)
        assert single_shot.call_args.args[0] == 1000
        finish_marking = single_shot.call_args.args[1]
        finish_marking()
        assert single_shot.call_args.args[0] == 1000
        finish_blackout = single_shot.call_args.args[1]
        finish_blackout()

    window._projector_window.show_move.assert_called_once_with(
        (6, 4),
        (4, 4),
        source_is_light=True,
        destination_is_light=True,
        include_turn_indicator=True,
    )
    window._projector_window.show_black.assert_called_once_with()
    assert window._offline_move_marking_active is False
    assert window._offline_move_marking_blackout_pending is False
    assert window._reference_move_tracker.reset_activity.call_count == 2


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
        _offline_move_marking_active=False,
        _offline_move_marking_blackout_pending=False,
        _lichess_guidance_projection_active=True,
        _lichess_guidance_blackout_pending=False,
        _update_board_edge_activity_indicator=MagicMock(),
        _observe_lichess_guidance_activity=MagicMock(),
        _alignment_monitor=SimpleNamespace(output_mode="move"),
    )

    CameraWindow._update_occupancy(window)

    window._update_board_edge_activity_indicator.assert_called_once_with()
    window._observe_lichess_guidance_activity.assert_called_once_with()


def test_board_edge_indicator_records_only_state_transitions() -> None:
    detector = SimpleNamespace(
        observe=MagicMock(side_effect=(True, True, False)),
        crossing_detected=True,
        largest_component_pixels=1400,
        largest_component_inside_pixels=700,
        largest_component_outside_pixels=700,
        largest_component_outer_halo_pixels=100,
        largest_component_bounds=(100, 200, 30, 40),
    )
    window = SimpleNamespace(
        _latest_frame=object(),
        _calibration_result=SimpleNamespace(
            mapping=SimpleNamespace(camera_grid=object())
        ),
        _board_edge_activity_detector=detector,
        _board_edge_indicator_active=False,
        _projector_window=SimpleNamespace(
            set_edge_activity_indicator=MagicMock()
        ),
        _record_runtime_diagnostic=MagicMock(),
    )

    CameraWindow._update_board_edge_activity_indicator(window)
    CameraWindow._update_board_edge_activity_indicator(window)
    detector.crossing_detected = False
    detector.largest_component_pixels = 0
    detector.largest_component_inside_pixels = 0
    detector.largest_component_outside_pixels = 0
    detector.largest_component_outer_halo_pixels = 0
    detector.largest_component_bounds = None
    CameraWindow._update_board_edge_activity_indicator(window)

    assert [
        call.args[0]
        for call in window._record_runtime_diagnostic.call_args_list
    ] == [
        "board-edge-indicator-activated",
        "board-edge-indicator-cleared",
    ]
    assert all(
        call.kwargs["include_images"]
        for call in window._record_runtime_diagnostic.call_args_list
    )


def test_black_button_starts_pending_move_observation() -> None:
    window = SimpleNamespace(
        _lichess_guidance_projection_active=True,
        _offline_move_marking_active=False,
        _offline_move_marking_blackout_pending=False,
        _offline_move_marking_generation=0,
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
