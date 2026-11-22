from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from src.chess_camera.camera_app import CameraWindow
from src.chess_camera.guidance_activity import (
    BlackoutSettlingDetector,
    BlackoutSettlingUpdate,
    BoardEdgeHandGate,
    BoardEdgeActivityDetector,
)
from src.chess_camera.legal_game import LegalGame
from src.chess_camera.reference_move_tracker import ReferenceTrackerUpdate


def _camera_grid() -> np.ndarray:
    coordinates = np.linspace(200, 1000, 9, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(coordinates, coordinates)
    return np.stack((x_grid, y_grid), axis=2)


def _frame() -> np.ndarray:
    return np.full((1200, 1200, 3), 110, dtype=np.uint8)






def test_board_edge_indicator_records_observation_timing() -> None:
    detector = BoardEdgeActivityDetector()

    detector.observe(_frame(), _camera_grid())
    try:
        detector.observe(np.zeros((10, 10), dtype=np.uint8), _camera_grid())
    except ValueError:
        pass

    assert detector.observation_count == 2
    assert detector.last_observation_seconds >= 0.0


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


def test_hand_gate_blocks_resting_arm_against_clean_baseline() -> None:
    clean = _frame()
    gate = BoardEdgeHandGate()
    assert not gate.observe(clean, _camera_grid())

    gate.capture_baseline(clean, _camera_grid())
    reaching = clean.copy()
    cv2.rectangle(reaching, (350, 100), (550, 260), (20, 20, 20), -1)
    resting = clean.copy()
    cv2.rectangle(resting, (360, 100), (560, 270), (20, 20, 20), -1)
    for frame in (reaching, resting, resting, resting):
        assert gate.observe(frame, _camera_grid())

    edge_square_change = clean.copy()
    cv2.rectangle(
        edge_square_change, (350, 175), (550, 260), (20, 20, 20), -1
    )
    assert not gate.observe(edge_square_change, _camera_grid())
    assert gate.has_baseline


def test_hand_gate_releases_crossing_still_for_five_seconds() -> None:
    now = 0.0
    clean = _frame()
    gate = BoardEdgeHandGate(clock=lambda: now)
    gate.capture_baseline(clean, _camera_grid())
    crossing = clean.copy()
    cv2.rectangle(crossing, (350, 100), (550, 260), (20, 20, 20), -1)
    moved = clean.copy()
    cv2.rectangle(moved, (380, 100), (580, 260), (20, 20, 20), -1)

    assert gate.observe(crossing, _camera_grid())
    now = 4.0
    assert gate.observe(moved, _camera_grid())
    now = 8.9
    assert gate.observe(moved, _camera_grid())
    assert not gate.released
    now = 9.0
    assert not gate.observe(moved, _camera_grid())
    assert gate.released
    now = 9.2
    assert not gate.observe(moved, _camera_grid())
    assert not gate.released


def test_hand_gate_ignores_sleeve_overhanging_only_edge_square_margins() -> None:
    clean = _frame()
    gate = BoardEdgeHandGate()
    gate.capture_baseline(clean, _camera_grid())
    sleeve = clean.copy()
    cv2.rectangle(sleeve, (350, 90), (550, 207), (20, 20, 20), -1)

    assert not gate.observe(sleeve, _camera_grid())
    assert gate.largest_component_inside_pixels >= 600
    assert gate.largest_square_coverage < 0.1

    reaching = clean.copy()
    cv2.rectangle(reaching, (350, 90), (550, 240), (20, 20, 20), -1)
    assert gate.observe(reaching, _camera_grid())
    assert gate.largest_square_coverage >= 0.1


def test_hand_gate_release_ignores_distant_motion() -> None:
    now = 0.0
    clean = _frame()
    gate = BoardEdgeHandGate(clock=lambda: now)
    gate.capture_baseline(clean, _camera_grid())
    crossing = clean.copy()
    cv2.rectangle(crossing, (350, 100), (550, 260), (20, 20, 20), -1)
    ticking = crossing.copy()
    cv2.rectangle(ticking, (850, 1020), (950, 1080), (200, 200, 200), -1)

    for now, frame in ((0.0, crossing), (2.5, ticking), (4.9, crossing)):
        assert gate.observe(frame, _camera_grid())
    now = 5.0
    assert not gate.observe(ticking, _camera_grid())
    assert gate.released


def test_hand_gate_drops_baseline_when_camera_grid_changes() -> None:
    clean = _frame()
    gate = BoardEdgeHandGate()
    gate.capture_baseline(clean, _camera_grid())
    crossing = clean.copy()
    cv2.rectangle(crossing, (350, 100), (550, 260), (20, 20, 20), -1)

    assert not gate.observe(crossing, _camera_grid() + 2.0)
    assert not gate.has_baseline


def test_board_edge_indicator_is_independent_of_board_position_in_frame() -> None:
    crossing = _frame()
    cv2.rectangle(crossing, (350, 100), (550, 260), (20, 20, 20), -1)

    def observe(offset_x: int, offset_y: int) -> tuple[object, ...]:
        frame = np.full((1600, 1700, 3), 110, dtype=np.uint8)
        quiet = frame.copy()
        frame[offset_y : offset_y + 1200, offset_x : offset_x + 1200] = crossing
        grid = _camera_grid() + np.float32((offset_x, offset_y))
        detector = BoardEdgeActivityDetector(
            arming_observations=2,
            confirmation_observations=1,
            quiet_observations=1,
        )
        detector.observe(quiet, grid)
        detector.observe(quiet, grid)
        active = detector.observe(frame, grid)
        bounds = detector.largest_component_bounds
        assert bounds is not None
        return (
            active,
            detector.crossing_detected,
            detector.largest_component_pixels,
            detector.largest_component_inside_pixels,
            detector.largest_component_outside_pixels,
            detector.largest_component_outer_halo_pixels,
            (bounds[0] - offset_x, bounds[1] - offset_y, *bounds[2:]),
        )

    reference = observe(0, 0)
    assert reference[0] and reference[1]
    assert observe(437, 291) == reference


def test_board_edge_indicator_rearms_when_board_leaves_cached_crop() -> None:
    frame = _frame()
    crossing = frame.copy()
    cv2.rectangle(crossing, (350, 100), (550, 260), (20, 20, 20), -1)
    detector = BoardEdgeActivityDetector(
        arming_observations=2,
        confirmation_observations=1,
        quiet_observations=1,
    )
    detector.observe(frame, _camera_grid())
    detector.observe(frame, _camera_grid())

    slightly_moved = _camera_grid() + np.float32((2, 1))
    assert detector.observe(crossing, slightly_moved)

    detector.reset()
    detector.observe(frame, _camera_grid())
    detector.observe(frame, _camera_grid())
    far_moved = _camera_grid() + np.float32((60, 0))
    assert not detector.observe(crossing, far_moved)
    assert not detector.active




























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










def test_pending_lichess_guidance_stays_on_without_board_entry() -> None:
    window = SimpleNamespace(
        _board_orientation_needs_detection=False,
        _piece_side_orientation_needs_detection=False,
        _calibration_result=object(),
        _open_projector=lambda: True,
        _physical_board_square=lambda row, column: (row, column),
        _projector_window=MagicMock(),
        _board_widget=MagicMock(),
        _alignment_monitor=MagicMock(),
        _projector_status=MagicMock(),
        _lichess_guidance_generation=0,
        _lichess_guidance_projection_active=False,
        _lichess_guidance_blackout_pending=False,
        _lichess_guidance_reprojection_pending=False,
        _resume_board_edge_activity_indicator=MagicMock(),
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
    window._projector_status.setText.assert_called_once_with(
        "Projecting pending Lichess move e7-e5 until board entry."
    )
    assert (
        window._record_runtime_diagnostic.call_args.args[0]
        == "lichess-guidance-projected"
    )
def test_pending_lichess_guidance_waits_for_complete_orientation() -> None:
    window = SimpleNamespace(
        _board_orientation_needs_detection=False,
        _piece_side_orientation_needs_detection=True,
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
        _resume_board_edge_activity_indicator=MagicMock(),
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
        _resume_board_edge_activity_indicator=MagicMock(),
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
        _projector_window=MagicMock(),
        _alignment_monitor=MagicMock(),
        _lichess_blackout_settling_detector=MagicMock(),
        _reference_move_tracker=MagicMock(),
        _suspend_board_edge_activity_indicator=MagicMock(),
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
    window._suspend_board_edge_activity_indicator.assert_called_once_with()
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


def test_boundary_entry_starts_lichess_guidance_blackout() -> None:
    window = SimpleNamespace(
        _lichess_guidance_projection_active=True,
        _board_edge_indicator_active=False,
        _lichess_guidance_activity_logged=False,
        _lichess_guidance_generation=3,
        _projector_status=MagicMock(),
        _record_runtime_diagnostic=MagicMock(),
        _begin_lichess_guidance_observation=MagicMock(),
    )

    CameraWindow._observe_lichess_guidance_activity(window)
    window._begin_lichess_guidance_observation.assert_not_called()

    window._board_edge_indicator_active = True
    CameraWindow._observe_lichess_guidance_activity(window)
    window._begin_lichess_guidance_observation.assert_called_once_with(
        "boundary-entry"
    )
    window._projector_status.setText.assert_called_with(
        "Hand entered the board; guidance is returning to black before move "
        "verification."
    )
    window._record_runtime_diagnostic.assert_called_once()


def _offline_marking_window() -> SimpleNamespace:
    window = SimpleNamespace(
        _board_orientation_needs_detection=False,
        _piece_side_orientation_needs_detection=False,
        _calibration_result=SimpleNamespace(
            mapping=SimpleNamespace(camera_grid=object())
        ),
        _latest_frame=object(),
        _open_projector=MagicMock(return_value=True),
        _physical_board_square=lambda row, column: (row, column),
        _projector_window=MagicMock(),
        _offline_move_marking_generation=0,
        _offline_move_marking_active=False,
        _offline_move_marking_blackout_pending=False,
        _offline_blackout_settling_detector=MagicMock(),
        _alignment_monitor=MagicMock(),
        _reference_move_tracker=MagicMock(),
        _projector_status=MagicMock(),
        _occupancy_status=MagicMock(),
    )
    window._finish_offline_move_marking = lambda generation: (
        CameraWindow._finish_offline_move_marking(window, generation)
    )
    window._finish_offline_move_marking_blackout = (
        lambda generation, reason="fallback": (
            CameraWindow._finish_offline_move_marking_blackout(
                window,
                generation,
                reason,
            )
        )
    )
    return window


def test_offline_move_marking_shows_half_second_with_one_second_fallback(
) -> None:
    move = ((6, 4), (4, 4))
    window = _offline_marking_window()

    with patch(
        "src.chess_camera.camera_app.QTimer.singleShot"
    ) as single_shot:
        CameraWindow._project_offline_accepted_move(window, move)
        assert single_shot.call_args.args[0] == 500
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
    )
    window._projector_window.show_black.assert_called_once_with()
    window._offline_blackout_settling_detector.reset.assert_called_once_with()
    assert window._offline_move_marking_active is False
    assert window._offline_move_marking_blackout_pending is False
    assert window._reference_move_tracker.reset_activity.call_count == 2


def test_offline_blackout_ends_early_when_frames_confirm_black() -> None:
    window = _offline_marking_window()
    with patch("src.chess_camera.camera_app.QTimer.singleShot") as single_shot:
        CameraWindow._project_offline_accepted_move(
            window,
            ((6, 4), (4, 4)),
        )
        single_shot.call_args.args[1]()
        stale_fallback = single_shot.call_args.args[1]
    detector = window._offline_blackout_settling_detector

    detector.observe.return_value = BlackoutSettlingUpdate("waiting", 2, 3, 650)
    CameraWindow._observe_offline_move_marking_blackout(window)
    assert window._offline_move_marking_blackout_pending is True

    detector.observe.return_value = BlackoutSettlingUpdate("stable", 3, 3, 700)
    CameraWindow._observe_offline_move_marking_blackout(window)

    assert window._offline_move_marking_blackout_pending is False
    assert window._reference_move_tracker.reset_activity.call_count == 2
    window._projector_status.setText.assert_called_with(
        "Projector black confirmed by the camera; watching for the next "
        "Offline move."
    )
    stale_fallback()
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
        _camera_geometry_ready=True,
        _startup_offline_reference_refresh_pending=False,
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


def test_lichess_blackout_rearms_edge_indicator_after_synchronization() -> None:
    frame = _frame()
    crossing = frame.copy()
    cv2.rectangle(crossing, (350, 100), (550, 260), (20, 20, 20), -1)
    detector = BoardEdgeActivityDetector(
        arming_observations=2,
        confirmation_observations=1,
        quiet_observations=1,
    )
    assert not detector.observe(frame, _camera_grid())
    assert not detector.observe(frame, _camera_grid())
    assert detector.observe(crossing, _camera_grid())

    projector_window = SimpleNamespace(
        set_edge_activity_indicator=MagicMock()
    )
    window = SimpleNamespace(
        _latest_frame=frame,
        _calibration_result=SimpleNamespace(
            mapping=SimpleNamespace(camera_grid=_camera_grid())
        ),
        _board_edge_activity_detector=detector,
        _board_edge_indicator_active=True,
        _board_edge_activity_suspended=False,
        _board_edge_indicator_active_since=0.0,
        _board_edge_indicator_last_stuck_diagnostic_at=None,
        _projector_window=projector_window,
        _record_runtime_diagnostic=MagicMock(),
    )
    window._reset_board_edge_activity_indicator = lambda: (
        CameraWindow._reset_board_edge_activity_indicator(window)
    )

    CameraWindow._suspend_board_edge_activity_indicator(window)
    CameraWindow._update_board_edge_activity_indicator(window)

    assert window._board_edge_activity_suspended is True
    assert window._board_edge_indicator_active is False
    assert detector.active is False

    CameraWindow._resume_board_edge_activity_indicator(window)
    CameraWindow._update_board_edge_activity_indicator(window)
    CameraWindow._update_board_edge_activity_indicator(window)
    CameraWindow._update_board_edge_activity_indicator(window)

    assert window._board_edge_activity_suspended is False
    assert window._board_edge_indicator_active is False
    assert detector.active is False
    projector_window.set_edge_activity_indicator.assert_not_called()


def _hand_indicator_window(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "_latest_frame": object(),
        "_calibration_result": object(),
        "_lichess_guidance_projection_active": False,
        "_illegal_correction_projection_active": False,
        "_board_edge_indicator_active": False,
        "_board_edge_gate_blocking": False,
        "_hand_indicator_active": False,
        "_projector_window": SimpleNamespace(
            set_edge_activity_indicator=MagicMock()
        ),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_hand_indicator_follows_tracker_gate_on_black_board() -> None:
    window = _hand_indicator_window(_board_edge_indicator_active=True)

    for state, expected in (
        ("activity", True),
        ("waiting-for-clear", True),
        ("settling", False),
        ("candidate", False),
        ("accepted", False),
        (None, False),
    ):
        CameraWindow._update_hand_indicator(
            window,
            state,
            tracker_ran=True,
        )
        assert window._hand_indicator_active is expected
        window._projector_window.set_edge_activity_indicator.assert_called_with(
            expected
        )


def test_hand_indicator_uses_edge_detector_only_during_projection() -> None:
    window = _hand_indicator_window(
        _lichess_guidance_projection_active=True,
        _board_edge_indicator_active=True,
    )

    CameraWindow._update_hand_indicator(window, None, tracker_ran=False)
    assert window._hand_indicator_active is True

    window._lichess_guidance_projection_active = False
    window._board_edge_indicator_active = False
    CameraWindow._update_hand_indicator(window, None, tracker_ran=False)
    assert window._hand_indicator_active is True

    window._latest_frame = None
    CameraWindow._update_hand_indicator(window, None, tracker_ran=False)
    assert window._hand_indicator_active is False


def test_hand_indicator_shows_board_edge_gate_on_black_board() -> None:
    window = _hand_indicator_window(_board_edge_gate_blocking=True)

    CameraWindow._update_hand_indicator(window, None, tracker_ran=False)

    assert window._hand_indicator_active is True


def _edge_gate_window(gate: MagicMock) -> SimpleNamespace:
    return SimpleNamespace(
        _latest_frame=_frame(),
        _calibration_result=SimpleNamespace(
            mapping=SimpleNamespace(camera_grid=_camera_grid())
        ),
        _board_edge_hand_gate=gate,
        _board_edge_gate_blocking=False,
        _reference_move_tracker=MagicMock(),
        _occupancy_status=MagicMock(),
        _projector_status=MagicMock(),
        _record_runtime_diagnostic=MagicMock(),
    )


def test_board_edge_gate_blocks_tracker_and_records_transitions() -> None:
    gate = MagicMock(
        has_baseline=True,
        released=False,
        largest_component_bounds=(1, 2, 3, 4),
    )
    gate.observe.return_value = True
    window = _edge_gate_window(gate)

    assert CameraWindow._board_edge_gate_blocks(window)
    assert CameraWindow._board_edge_gate_blocks(window)
    gate.observe.return_value = False
    assert not CameraWindow._board_edge_gate_blocks(window)

    assert window._reference_move_tracker.reset_activity.call_count == 2
    assert [
        call.args[0] for call in window._record_runtime_diagnostic.mock_calls
    ] == ["board-edge-gate-blocking", "board-edge-gate-cleared"]


def test_board_edge_gate_without_baseline_does_not_block() -> None:
    gate = MagicMock(has_baseline=False)
    window = _edge_gate_window(gate)

    assert not CameraWindow._board_edge_gate_blocks(window)
    gate.observe.assert_not_called()


def test_board_edge_gate_skips_ticks_that_cannot_register() -> None:
    gate = MagicMock(has_baseline=True, released=False)
    gate.observe.return_value = True
    window = _edge_gate_window(gate)

    for state in ("watching", "activity"):
        window._reference_move_tracker.last_state = state
        assert not CameraWindow._board_edge_gate_blocks(window)
    gate.observe.assert_not_called()

    window._reference_move_tracker.last_state = "settling"
    assert CameraWindow._board_edge_gate_blocks(window)
    window._reference_move_tracker.last_state = "activity"
    assert CameraWindow._board_edge_gate_blocks(window)
    assert gate.observe.call_count == 2


def test_blocking_edge_gate_skips_reference_tracker_observation() -> None:
    window = SimpleNamespace(
        _latest_frame=object(),
        _calibration_result=object(),
        _calibrator=SimpleNamespace(active=False),
        _camera_geometry_ready=True,
        _lichess_guidance_projection_active=False,
        _illegal_correction_projection_active=False,
        _startup_offline_reference_refresh_pending=False,
        _activate_offline_after_reset=False,
        _pending_promotion_selection=None,
        _illegal_correction_blackout_pending=False,
        _offline_move_marking_active=False,
        _offline_move_marking_blackout_pending=False,
        _lichess_guidance_blackout_pending=False,
        _reconciliation_blackout_pending=False,
        _alignment_monitor=SimpleNamespace(output_mode="black"),
        _detect_current_board_orientation=MagicMock(),
        _board_orientation_needs_detection=False,
        _detect_automatic_starting_position=MagicMock(return_value=False),
        _reference_move_tracker=SimpleNamespace(initialized=True),
        _piece_side_orientation_needs_detection=False,
        _reset_to_start_requested=False,
        _reconciliation_requested=False,
        _offline_restore_requested=False,
        _board_edge_gate_blocks=MagicMock(return_value=True),
        _update_reference_move_tracking=MagicMock(),
    )

    CameraWindow._update_occupancy(window)
    window._update_reference_move_tracking.assert_not_called()

    window._board_edge_gate_blocks.return_value = False
    CameraWindow._update_occupancy(window)
    window._update_reference_move_tracking.assert_called_once_with()


def test_edge_detector_does_not_run_on_black_board() -> None:
    window = SimpleNamespace(
        _latest_frame=object(),
        _calibration_result=object(),
        _calibrator=SimpleNamespace(active=False),
        _camera_geometry_ready=True,
        _startup_offline_reference_refresh_pending=True,
        _lichess_guidance_projection_active=False,
        _illegal_correction_projection_active=False,
        _update_board_edge_activity_indicator=MagicMock(),
        _start_pending_offline_reference_refresh=MagicMock(),
    )

    CameraWindow._update_occupancy(window)

    window._update_board_edge_activity_indicator.assert_not_called()


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
        _board_edge_activity_suspended=False,
        _board_edge_indicator_active_since=None,
        _board_edge_indicator_last_stuck_diagnostic_at=None,
        _projector_window=SimpleNamespace(
            set_edge_activity_indicator=MagicMock()
        ),
        _record_runtime_diagnostic=MagicMock(),
    )

    with patch(
        "src.chess_camera.camera_app.monotonic",
        side_effect=(0.0, 1.0, 2.0),
    ):
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


def test_board_edge_indicator_records_periodic_stuck_images() -> None:
    detector = SimpleNamespace(
        observe=MagicMock(return_value=True),
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
        _board_edge_activity_suspended=False,
        _board_edge_indicator_active_since=None,
        _board_edge_indicator_last_stuck_diagnostic_at=None,
        _projector_window=SimpleNamespace(
            set_edge_activity_indicator=MagicMock()
        ),
        _record_runtime_diagnostic=MagicMock(),
    )

    with patch(
        "src.chess_camera.camera_app.monotonic",
        side_effect=(0.0, 2.9, 3.1, 7.9, 8.2),
    ):
        for _index in range(5):
            CameraWindow._update_board_edge_activity_indicator(window)

    assert [
        call.args[0]
        for call in window._record_runtime_diagnostic.call_args_list
    ] == [
        "board-edge-indicator-activated",
        "board-edge-indicator-still-active",
        "board-edge-indicator-still-active",
    ]
    assert [
        call.kwargs["details"]["active_seconds"]
        for call in window._record_runtime_diagnostic.call_args_list
    ] == [0.0, 3.1, 8.2]
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
        _illegal_correction_generation=0,
        _illegal_correction_physical_squares=(),
        _calibrator=SimpleNamespace(active=False),
        _open_projector=MagicMock(return_value=True),
        _resume_board_edge_activity_indicator=MagicMock(),
        _suspend_selected_square=MagicMock(),
        _begin_lichess_guidance_observation=MagicMock(),
        _projector_status=MagicMock(),
    )

    CameraWindow._black_projector(window)

    window._open_projector.assert_called_once_with()
    window._suspend_selected_square.assert_called_once_with()
    window._begin_lichess_guidance_observation.assert_called_once_with()


def test_illegal_correction_schedules_bounded_fallback() -> None:
    window = SimpleNamespace(
        _illegal_correction_projection_active=False,
        _illegal_correction_blackout_pending=False,
        _illegal_correction_generation=0,
        _illegal_correction_physical_squares=(),
        _illegal_correction_last_projected_move=None,
        _calibration_result=object(),
        _open_projector=MagicMock(return_value=True),
        _physical_board_square=lambda row, column: (row, column),
        _projector_window=MagicMock(),
        _resume_board_edge_activity_indicator=MagicMock(),
        _alignment_monitor=MagicMock(),
        _projector_status=MagicMock(),
        _record_runtime_diagnostic=MagicMock(),
        _begin_illegal_correction_observation_if_current=MagicMock(),
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
    single_shot.assert_called_once()
    assert single_shot.call_args.args[0] == 2000
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
    single_shot.call_args.args[1]()
    (
        window._begin_illegal_correction_observation_if_current
        .assert_called_once_with(1, "fallback")
    )

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


def test_compound_correction_projects_all_squares_red() -> None:
    window = SimpleNamespace(
        _illegal_correction_projection_active=False,
        _illegal_correction_blackout_pending=False,
        _illegal_correction_generation=0,
        _illegal_correction_physical_squares=(),
        _illegal_correction_last_projected_move=None,
        _calibration_result=object(),
        _open_projector=MagicMock(return_value=True),
        _physical_board_square=lambda row, column: (row, column),
        _projector_window=MagicMock(),
        _resume_board_edge_activity_indicator=MagicMock(),
        _alignment_monitor=MagicMock(),
        _projector_status=MagicMock(),
        _record_runtime_diagnostic=MagicMock(),
        _begin_illegal_correction_observation_if_current=MagicMock(),
    )
    squares = ((1, 4), (2, 4), (6, 5), (5, 5))

    with patch("src.chess_camera.camera_app.QTimer.singleShot"):
        CameraWindow._project_illegal_correction_squares(
            window,
            squares,
        )

    window._projector_window.show_reconciliation.assert_called_once_with(
        squares
    )
    assert window._illegal_correction_physical_squares == squares
    assert (
        window._record_runtime_diagnostic.call_args.kwargs["details"][
            "logical_squares"
        ]
        == ["e7", "e6", "f2", "f3"]
    )


def test_boundary_entry_starts_illegal_correction_blackout() -> None:
    window = SimpleNamespace(
        _illegal_correction_projection_active=True,
        _board_edge_indicator_active=False,
        _illegal_correction_physical_squares=((6, 2), (5, 2)),
        _illegal_correction_generation=3,
        _begin_illegal_correction_observation=MagicMock(),
        _projector_status=MagicMock(),
        _record_runtime_diagnostic=MagicMock(),
    )

    CameraWindow._observe_illegal_correction_activity(window)
    window._begin_illegal_correction_observation.assert_not_called()

    window._board_edge_indicator_active = True
    CameraWindow._observe_illegal_correction_activity(window)

    window._begin_illegal_correction_observation.assert_called_once_with(
        "boundary-entry"
    )
    window._projector_status.setText.assert_called_with(
        "Hand entered the board; red correction is returning to black before "
        "restoration verification."
    )
    window._record_runtime_diagnostic.assert_called_once()
    assert (
        window._record_runtime_diagnostic.call_args.args[0]
        == "illegal-correction-hand-entered"
    )


def test_illegal_correction_blackout_starts_after_boundary_entry() -> None:
    callback = None

    def capture_timer(milliseconds: int, pending_callback: object) -> None:
        nonlocal callback
        assert milliseconds == 1000
        callback = pending_callback

    window = SimpleNamespace(
        _illegal_correction_projection_active=True,
        _illegal_correction_blackout_pending=False,
        _illegal_correction_generation=4,
        _suspend_board_edge_activity_indicator=MagicMock(),
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
        CameraWindow._begin_illegal_correction_observation(
            window,
            "boundary-entry",
        )

    assert window._illegal_correction_projection_active is False
    assert window._illegal_correction_blackout_pending is True
    window._suspend_board_edge_activity_indicator.assert_called_once_with()
    window._projector_window.show_black.assert_called_once_with()
    assert (
        window._record_runtime_diagnostic.call_args.args[0]
        == "illegal-correction-blackout-started"
    )
    assert callback is not None


def test_stale_illegal_correction_fallback_cannot_black_new_projection() -> None:
    window = SimpleNamespace(
        _illegal_correction_generation=4,
        _begin_illegal_correction_observation=MagicMock(),
    )

    CameraWindow._begin_illegal_correction_observation_if_current(
        window,
        3,
        "fallback",
    )
    window._begin_illegal_correction_observation.assert_not_called()

    CameraWindow._begin_illegal_correction_observation_if_current(
        window,
        4,
        "fallback",
    )
    window._begin_illegal_correction_observation.assert_called_once_with(
        "fallback"
    )
