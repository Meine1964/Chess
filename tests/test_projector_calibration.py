import json
from collections.abc import Callable

import cv2
import numpy as np
import pytest

from src.chess_camera.board_detector import BoardDetection
from src.chess_camera.calibration_controller import (
    CalibrationResult,
    ProjectorCalibrationController,
)
from src.chess_camera.calibration_store import CalibrationStore
from src.chess_camera.projector_calibration import (
    CalibrationError,
    MappingResult,
    ValidationResult,
    coded_marker_pattern,
    decode_gray_code,
    detect_coded_marker_correspondences,
    detect_board_orientation,
    fit_coded_pattern_mapping,
    fit_projector_mapping,
    fit_visible_middle_lattice,
    gray_code_bit_count,
    gray_code_pattern,
    interpolate_grid_point,
    locate_validation_target,
    measure_coverage,
    quick_realign_mapping,
    remap_projector_mapping,
    refine_projector_mesh,
    square_polygons,
    validation_result,
)


def _detection(
    left: float = 10,
    top: float = 10,
    right: float = 90,
    bottom: float = 90,
) -> BoardDetection:
    x_values = np.linspace(left, right, 9, dtype=np.float32)
    y_values = np.linspace(top, bottom, 9, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    grid = np.stack((x_grid, y_grid), axis=2).astype(np.float32)
    return BoardDetection(
        grid_points=grid,
        confidence=1.0,
        reprojection_error=0.0,
        method="test",
        grid_alignment=1.0,
    )


def test_coded_marker_pattern_decodes_absolute_projector_coordinates() -> None:
    pattern = coded_marker_pattern(1280, 720)

    camera_points, projector_points, marker_ids = (
        detect_coded_marker_correspondences(
            np.zeros_like(pattern.image),
            pattern.image,
            pattern,
        )
    )

    assert pattern.image.shape == (720, 1280)
    assert len(pattern.marker_corners) == 240
    assert len(np.unique(marker_ids)) == 240
    first_marker = pattern.marker_corners[0]
    assert first_marker[1, 0] - first_marker[0, 0] == 39
    assert pattern.marker_corners[1][0, 0] - first_marker[0, 0] == 52
    assert np.max(np.abs(camera_points - projector_points)) < 0.5


def test_inverse_coded_marker_pattern_decodes_same_coordinates() -> None:
    pattern = coded_marker_pattern(1280, 720)
    inverse_pattern = coded_marker_pattern(
        1280,
        720,
        inverse=True,
        staggered=True,
    )

    camera_points, projector_points, marker_ids = (
        detect_coded_marker_correspondences(
            np.zeros_like(pattern.image),
            inverse_pattern.image,
            inverse_pattern,
            inverse=True,
        )
    )

    assert inverse_pattern.marker_corners.keys() == pattern.marker_corners.keys()
    np.testing.assert_allclose(
        inverse_pattern.marker_corners[0] - pattern.marker_corners[0],
        26,
    )
    assert len(np.unique(marker_ids)) == 240
    assert np.max(np.abs(camera_points - projector_points)) < 0.5


def test_coded_marker_error_reports_measured_response() -> None:
    pattern = coded_marker_pattern(1280, 720)

    with pytest.raises(
        CalibrationError,
        match=r"insufficient camera response \(0\.0 grayscale levels",
    ):
        detect_coded_marker_correspondences(
            np.zeros_like(pattern.image),
            np.zeros_like(pattern.image),
            pattern,
        )


def test_optional_coded_capture_allows_no_detected_markers() -> None:
    pattern = coded_marker_pattern(1280, 720)
    illuminated = np.full_like(pattern.image, 10)

    camera_points, projector_points, marker_ids = (
        detect_coded_marker_correspondences(
            np.zeros_like(pattern.image),
            illuminated,
            pattern,
            minimum_markers=0,
        )
    )

    assert camera_points.shape == (0, 2)
    assert projector_points.shape == (0, 2)
    assert marker_ids.shape == (0,)


def test_inverse_coded_capture_uses_valid_diagnostic_name() -> None:
    controller = ProjectorCalibrationController(object(), object())
    controller._captures["coded_inverse"] = np.zeros((10, 10), dtype=np.uint8)

    assert set(controller.diagnostic_captures) == {"coded-inverse"}


def test_coded_mapping_uses_measured_visible_lattice() -> None:
    width = 1280
    height = 720
    canonical = np.full((800, 800), 105, dtype=np.uint8)
    for row in range(6):
        for column in range(8):
            canonical[
                row * 100 : (row + 1) * 100,
                column * 100 : (column + 1) * 100,
            ] = 30 if (row + column) % 2 else 180
    canonical_corners = np.asarray(
        [[0, 0], [799, 0], [799, 799], [0, 799]],
        dtype=np.float32,
    )
    camera_corners = np.asarray(
        [[260, 45], [1010, 70], [1080, 680], [210, 650]],
        dtype=np.float32,
    )
    board_homography = cv2.getPerspectiveTransform(
        canonical_corners,
        camera_corners,
    )
    black = cv2.warpPerspective(
        canonical,
        board_homography,
        (width, height),
        borderValue=5,
    )
    model_grid = np.asarray(
        [
            (column * 100, row * 100)
            for row in range(9)
            for column in range(9)
        ],
        dtype=np.float32,
    )
    expected_grid = cv2.perspectiveTransform(
        model_grid.reshape(-1, 1, 2),
        board_homography,
    ).reshape(9, 9, 2)
    rough_detection = BoardDetection(
        grid_points=expected_grid
        + np.asarray([3.0, -2.0], dtype=np.float32),
        confidence=0.9,
        reprojection_error=1.0,
        method="rough test raster",
        grid_alignment=0.9,
    )
    pattern = coded_marker_pattern(width, height)
    illuminated = np.clip(
        black.astype(np.int16) + pattern.image.astype(np.int16),
        0,
        255,
    ).astype(np.uint8)

    detection, mapping = fit_coded_pattern_mapping(
        black,
        illuminated,
        rough_detection,
        pattern,
        (width, height),
    )

    assert detection.method == "visible 7x5 middle lattice"
    assert detection.reprojection_error is not None
    assert detection.reprojection_error < 0.1
    assert mapping.inlier_ratio == pytest.approx(1.0)
    assert mapping.rms_error < 1.0
    assert (
        np.max(
            np.linalg.norm(
                mapping.projector_grid - expected_grid,
                axis=2,
            )
        )
        < 1.5
    )

    shifted_grid = expected_grid.copy()
    shifted_grid[[0, 0, 8, 8], [0, 8, 8, 0]] += np.asarray(
        [[-80, -80], [80, -80], [80, 80], [-80, 80]],
        dtype=np.float32,
    )
    shifted_outer_detection = BoardDetection(
        grid_points=shifted_grid,
        confidence=0.9,
        reprojection_error=1.0,
        method="decorative-border test raster",
        grid_alignment=0.9,
    )

    unanchored = fit_visible_middle_lattice(
        black,
        shifted_outer_detection,
    )

    assert unanchored.method == "visible 7x5 middle lattice"
    assert (
        np.max(
            np.linalg.norm(
                unanchored.grid_points - expected_grid,
                axis=2,
            )
        )
        < 1.5
    )


def test_middle_lattice_accepts_verified_internal_corners_on_populated_board(
) -> None:
    image = np.full((1080, 1920), 125, dtype=np.uint8)
    canonical = np.empty((800, 800), dtype=np.uint8)
    for row in range(8):
        for column in range(8):
            canonical[
                row * 100 : (row + 1) * 100,
                column * 100 : (column + 1) * 100,
            ] = 190 if (row + column) % 2 == 0 else 45
    for row in (0, 1, 6, 7):
        for column in range(8):
            cv2.circle(
                canonical,
                (column * 100 + 50, row * 100 + 50),
                28,
                90 if row < 2 else 15,
                -1,
            )

    canonical_outer = np.asarray(
        [[0, 0], [799, 0], [799, 799], [0, 799]],
        dtype=np.float32,
    )
    camera_outer = np.asarray(
        [[625, 306], [1039, 310], [1063, 679], [559, 675]],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(
        canonical_outer,
        camera_outer,
    )
    board = cv2.warpPerspective(
        canonical,
        homography,
        (1920, 1080),
        borderValue=125,
    )
    board_mask = cv2.warpPerspective(
        np.full((800, 800), 255, dtype=np.uint8),
        homography,
        (1920, 1080),
    )
    image[board_mask > 0] = board[board_mask > 0]
    model_grid = np.asarray(
        [
            (column * 100, row * 100)
            for row in range(9)
            for column in range(9)
        ],
        dtype=np.float32,
    )
    expected_grid = cv2.perspectiveTransform(
        model_grid.reshape(-1, 1, 2),
        homography,
    ).reshape(9, 9, 2)
    rough_detection = BoardDetection(
        grid_points=expected_grid,
        confidence=0.909,
        reprojection_error=1.05,
        method="internal corners",
        grid_alignment=1.0,
    )

    refined = fit_visible_middle_lattice(image, rough_detection)

    assert refined.method == "verified internal corners"
    np.testing.assert_allclose(refined.grid_points, expected_grid)

    with pytest.raises(
        CalibrationError,
        match="middle-board intersections",
    ):
        fit_visible_middle_lattice(
            image,
            BoardDetection(
                grid_points=expected_grid,
                confidence=0.909,
                reprojection_error=1.11,
                method="internal corners",
                grid_alignment=1.0,
            ),
        )


def test_gray_code_patterns_decode_projector_coordinates() -> None:
    width = 13
    height = 9
    horizontal_pairs = [
        (
            gray_code_pattern(width, height, "x", bit),
            gray_code_pattern(width, height, "x", bit, inverse=True),
        )
        for bit in range(gray_code_bit_count(width))
    ]
    vertical_pairs = [
        (
            gray_code_pattern(width, height, "y", bit),
            gray_code_pattern(width, height, "y", bit, inverse=True),
        )
        for bit in range(gray_code_bit_count(height))
    ]

    decoded_x, decoded_y, valid = decode_gray_code(
        horizontal_pairs,
        vertical_pairs,
        np.ones((height, width), dtype=bool),
    )

    expected_x = np.broadcast_to(np.arange(width), (height, width))
    expected_y = np.broadcast_to(
        np.arange(height)[:, np.newaxis],
        (height, width),
    )
    assert np.array_equal(decoded_x, expected_x)
    assert np.array_equal(decoded_y, expected_y)
    assert np.all(valid)

    coarse_x, coarse_y, coarse_valid = decode_gray_code(
        horizontal_pairs[:-1],
        vertical_pairs[:-1],
        np.ones((height, width), dtype=bool),
        horizontal_omitted_bits=1,
        vertical_omitted_bits=1,
    )
    assert np.array_equal(coarse_x, (expected_x // 2) * 2 + 1)
    assert np.array_equal(coarse_y, (expected_y // 2) * 2 + 1)
    assert np.all(coarse_valid)


def test_coverage_allows_obstructed_square_but_requires_every_quadrant() -> None:
    detection = _detection()
    black = np.full((100, 100), 20, dtype=np.uint8)
    illuminated = np.full((100, 100), 100, dtype=np.uint8)

    result = measure_coverage(black, illuminated, detection)

    assert result.coverage_ratio == pytest.approx(1.0)
    illuminated[10:20, 10:20] = black[10:20, 10:20]
    obstructed = measure_coverage(black, illuminated, detection)
    assert obstructed.coverage_ratio < 1.0

    illuminated = np.full((100, 100), 100, dtype=np.uint8)
    illuminated[10:50, 10:50] = black[10:50, 10:50]
    with pytest.raises(CalibrationError, match="quadrant 1"):
        measure_coverage(black, illuminated, detection)


def test_detects_board_orientation_from_square_color_parity() -> None:
    detection = _detection()
    standard = np.zeros((100, 100), dtype=np.uint8)
    rotated = np.zeros((100, 100), dtype=np.uint8)
    for row in range(8):
        for column in range(8):
            corners = np.rint(
                np.asarray(
                    [
                        detection.grid_points[row, column],
                        detection.grid_points[row, column + 1],
                        detection.grid_points[row + 1, column + 1],
                        detection.grid_points[row + 1, column],
                    ]
                )
            ).astype(np.int32)
            standard_value = 200 if (row + column) % 2 == 0 else 40
            cv2.fillConvexPoly(standard, corners, standard_value)
            cv2.fillConvexPoly(rotated, corners, 240 - standard_value)

    assert detect_board_orientation(standard, detection.grid_points) == 0
    assert detect_board_orientation(rotated, detection.grid_points) == 1


def test_new_calibration_requires_orientation_redetection() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from src.chess_camera.camera_app import CameraWindow

    geometry = SimpleNamespace(
        width=lambda: 1280,
        height=lambda: 720,
        x=lambda: 0,
        y=lambda: 0,
    )
    screen = SimpleNamespace(geometry=lambda: geometry)
    result = SimpleNamespace(
        mapping=SimpleNamespace(projector_grid=np.zeros((9, 9, 2)))
    )
    window = SimpleNamespace(
        _piece_synchronizer=SimpleNamespace(initialized=True),
        _suspend_selected_square=MagicMock(),
        _reset_occupancy=MagicMock(),
        _projector_window=MagicMock(),
        _board_widget=MagicMock(),
        _reset_pieces_button=MagicMock(),
        _reconcile_button=MagicMock(),
        _show_grid_button=MagicMock(),
        _show_squares_button=MagicMock(),
        _quick_verify_button=MagicMock(),
        _validate_calibration_button=MagicMock(),
        _screens=[screen],
        _projector_selector=SimpleNamespace(currentIndex=lambda: 0),
        _alignment_monitor=MagicMock(),
        _set_calibration_indicator=MagicMock(),
        _camera_calibration_indicator=object(),
        _projector_calibration_indicator=object(),
    )

    CameraWindow._activate_calibration_result(window, result)

    assert window._board_orientation_needs_detection is True
    assert window._piece_side_orientation_needs_detection is True


def test_loaded_camera_geometry_requires_orientation_redetection() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from src.chess_camera.camera_app import CameraWindow

    result = SimpleNamespace()
    window = SimpleNamespace(
        _piece_synchronizer=SimpleNamespace(initialized=True),
        _suspend_selected_square=MagicMock(),
        _reset_occupancy=MagicMock(),
        _projector_window=MagicMock(),
        _board_widget=MagicMock(),
        _reset_pieces_button=MagicMock(),
        _reconcile_button=MagicMock(),
        _show_grid_button=MagicMock(),
        _show_squares_button=MagicMock(),
        _quick_verify_button=MagicMock(),
        _validate_calibration_button=MagicMock(),
        _alignment_monitor=MagicMock(),
        _set_calibration_indicator=MagicMock(),
        _camera_calibration_indicator=object(),
        _projector_calibration_indicator=object(),
    )

    CameraWindow._activate_camera_only_calibration(window, result)

    assert window._board_orientation_needs_detection is True
    assert window._piece_side_orientation_needs_detection is True


def test_parity_detection_does_not_persist_incomplete_orientation() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock, patch

    from src.chess_camera.camera_app import CameraWindow

    window = SimpleNamespace(
        _board_orientation_needs_detection=True,
        _latest_frame=np.zeros((20, 20, 3), dtype=np.uint8),
        _calibration_result=SimpleNamespace(
            mapping=SimpleNamespace(
                camera_grid=np.zeros((9, 9, 2), dtype=np.float32)
            )
        ),
        _board_quarter_turns=2,
        _settings=MagicMock(),
    )

    with patch(
        "src.chess_camera.camera_app.detect_board_orientation",
        return_value=1,
    ):
        CameraWindow._detect_current_board_orientation(window)

    assert window._board_quarter_turns == 1
    assert window._board_orientation_needs_detection is False
    window._settings.setValue.assert_not_called()
    window._settings.sync.assert_not_called()


def test_parity_detection_preserves_matching_complete_orientation() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock, patch

    from src.chess_camera.camera_app import CameraWindow

    window = SimpleNamespace(
        _board_orientation_needs_detection=True,
        _latest_frame=np.zeros((20, 20, 3), dtype=np.uint8),
        _calibration_result=SimpleNamespace(
            mapping=SimpleNamespace(
                camera_grid=np.zeros((9, 9, 2), dtype=np.float32)
            )
        ),
        _board_quarter_turns=2,
        _settings=MagicMock(),
    )

    with patch(
        "src.chess_camera.camera_app.detect_board_orientation",
        return_value=0,
    ):
        CameraWindow._detect_current_board_orientation(window)

    assert window._board_quarter_turns == 2
    assert window._board_orientation_needs_detection is False
    window._settings.setValue.assert_not_called()
    window._settings.sync.assert_not_called()


def test_square_projection_waits_for_white_side_orientation() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from src.chess_camera.camera_app import CameraWindow

    window = SimpleNamespace(
        _calibration_result=object(),
        _camera_only_calibration=False,
        _piece_side_orientation_needs_detection=True,
        _projector_status=MagicMock(),
        _record_runtime_diagnostic=MagicMock(),
        _open_projector=MagicMock(),
    )

    CameraWindow._board_square_clicked(window, 4, 2)

    window._open_projector.assert_not_called()
    assert (
        "White-side orientation"
        in window._projector_status.setText.call_args.args[0]
    )


def test_occupied_square_skips_runtime_optical_alignment_check() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock, patch

    from src.chess_camera.camera_app import CameraWindow

    window = SimpleNamespace(
        _pending_board_square=(0, 2),
        _calibration_result=object(),
        _latest_frame=None,
        _physical_board_square=MagicMock(return_value=(0, 2)),
        _projector_window=MagicMock(),
        _selected_board_square=None,
        _remembered_board_square=None,
        _board_widget=MagicMock(),
        _piece_synchronizer=SimpleNamespace(position={(0, 2): "b"}),
        _alignment_monitor=MagicMock(),
        _projector_status=MagicMock(),
        _record_runtime_diagnostic=MagicMock(),
    )

    with patch(
        "src.chess_camera.camera_app.QTimer.singleShot"
    ) as single_shot:
        CameraWindow._project_board_square_after_black(window, 0, 2)

    window._alignment_monitor.set_selected_square.assert_called_once_with(
        0,
        2,
        verify_projection=False,
    )
    assert single_shot.call_args.args[0] == 1000


def test_suspending_projection_clears_marker_but_remembers_square() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from src.chess_camera.camera_app import CameraWindow

    remembered_square = (4, 2)
    window = SimpleNamespace(
        _selected_board_square=remembered_square,
        _pending_board_square=(3, 2),
        _remembered_board_square=remembered_square,
        _board_widget=MagicMock(),
    )

    CameraWindow._suspend_selected_square(window)

    assert window._selected_board_square is None
    assert window._pending_board_square is None
    assert window._remembered_board_square == remembered_square
    window._board_widget.set_selected_square.assert_called_once_with(None)
    window._board_widget.set_last_move.assert_not_called()


def test_selected_square_timeout_returns_projector_to_black() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from src.chess_camera.camera_app import CameraWindow

    selected_square = (4, 2)
    window = SimpleNamespace(
        _selected_board_square=selected_square,
        _projector_window=SimpleNamespace(output_mode="selected_square"),
        _black_projector=MagicMock(),
        _projector_status=MagicMock(),
    )

    CameraWindow._finish_selected_square_projection(
        window,
        *selected_square,
    )

    window._black_projector.assert_called_once_with()
    window._projector_status.setText.assert_called_once_with(
        "Square illumination finished; projector returned to black."
    )


def test_stale_selected_square_timeout_does_not_clear_newer_square() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from src.chess_camera.camera_app import CameraWindow

    window = SimpleNamespace(
        _selected_board_square=(3, 2),
        _projector_window=SimpleNamespace(output_mode="selected_square"),
        _black_projector=MagicMock(),
        _projector_status=MagicMock(),
    )

    CameraWindow._finish_selected_square_projection(window, 4, 2)

    window._black_projector.assert_not_called()
    window._projector_status.setText.assert_not_called()


def test_failed_populated_calibration_retains_previous_mapping() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from src.chess_camera.camera_app import CameraWindow

    window = SimpleNamespace(
        _projector_window=MagicMock(),
        _calibration_result=object(),
        _alignment_monitor=MagicMock(),
        _calibrator=SimpleNamespace(
            quick_verification=False,
            validation_only=False,
            coded_pattern_mode=True,
            diagnostic_captures={},
        ),
        _invalidate_saved_calibration=MagicMock(return_value=None),
        _invalidate_calibration=MagicMock(),
        _set_calibration_indicator=MagicMock(),
        _projector_calibration_indicator=object(),
        _projector_status=MagicMock(),
        _record_runtime_diagnostic=MagicMock(),
    )

    CameraWindow._projector_calibration_failed(window, "failed")

    window._invalidate_saved_calibration.assert_not_called()
    window._invalidate_calibration.assert_not_called()
    window._alignment_monitor.set_black.assert_called_once_with()
    assert (
        "remains active"
        in window._projector_status.setText.call_args.args[0]
    )
    details = window._record_runtime_diagnostic.call_args.kwargs["details"]
    assert details["previous_calibration_retained"] is True
    assert details["previous_calibration_invalidated"] is False


def test_alignment_loss_invalidates_projection_mapping() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from src.chess_camera.camera_app import CameraWindow

    window = SimpleNamespace(
        _projector_window=MagicMock(),
        _invalidate_saved_calibration=MagicMock(return_value=None),
        _invalidate_calibration=MagicMock(),
        _set_calibration_indicator=MagicMock(),
        _camera_calibration_indicator=object(),
        _projector_calibration_indicator=object(),
        _alignment_status=MagicMock(),
        _projector_status=MagicMock(),
        _record_runtime_diagnostic=MagicMock(),
    )

    CameraWindow._calibration_error(window, "camera", "moved")

    window._invalidate_saved_calibration.assert_called_once_with()
    window._invalidate_calibration.assert_called_once_with()
    assert (
        "invalidated"
        in window._alignment_status.setText.call_args.args[0]
    )


def test_open_projector_marks_black_as_authoritative_output() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from src.chess_camera.camera_app import CameraWindow

    geometry = SimpleNamespace(
        width=lambda: 1280,
        height=lambda: 720,
        x=lambda: 0,
        y=lambda: 0,
    )
    screen = SimpleNamespace(
        geometry=lambda: geometry,
        name=lambda: "projector",
        manufacturer=lambda: "",
        model=lambda: "",
    )
    window = SimpleNamespace(
        _screens=[screen],
        _projector_selector=SimpleNamespace(currentIndex=lambda: 0),
        _projector_window=MagicMock(),
        _settings=MagicMock(),
        _projector_status=MagicMock(),
        _alignment_monitor=MagicMock(),
        _record_runtime_diagnostic=MagicMock(),
    )

    assert CameraWindow._open_projector(window) is True
    window._alignment_monitor.set_black.assert_called_once_with()


def test_black_action_opens_and_raises_projector_before_clearing_marker() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from src.chess_camera.camera_app import CameraWindow

    window = SimpleNamespace(
        _calibrator=SimpleNamespace(active=False),
        _open_projector=MagicMock(return_value=True),
        _suspend_selected_square=MagicMock(),
        _projector_status=MagicMock(),
        _lichess_guidance_projection_active=False,
        _offline_guidance_activity_detector=MagicMock(),
        _illegal_correction_activity_detector=MagicMock(),
        _illegal_correction_generation=0,
    )

    CameraWindow._black_projector(window)

    window._open_projector.assert_called_once_with()
    window._suspend_selected_square.assert_called_once_with()


def test_reconciliation_adopts_current_lichess_position_and_history() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from src.chess_camera.camera_app import CameraWindow
    from src.chess_camera.chess_position import (
        PiecePositionSynchronizer,
        standard_position,
    )
    from src.chess_camera.legal_game import LegalGame
    from src.chess_camera.lichess_client import (
        LichessGameFull,
        LichessGameState,
        LichessPlayer,
    )

    server_game = LegalGame()
    moved = server_game.apply_move((6, 4), (4, 4))
    state = LichessGameState(
        ("e2e4",),
        moved.fen,
        "black",
        0,
        0,
        0,
        0,
        "started",
        None,
        "*",
    )
    game = LichessGameFull(
        "game1234",
        "startpos",
        LichessPlayer("white", "Local", 1500),
        LichessPlayer("black", "Remote", 1500),
        state,
    )
    synchronizer = PiecePositionSynchronizer()
    synchronizer.restore(standard_position())
    tracker = MagicMock()
    tracker.reference = None
    board_widget = MagicMock()
    window = SimpleNamespace(
        _latest_lichess_game=game,
        _lichess_game_id=game.game_id,
        _physical_reconciliation_target=MagicMock(
            return_value=(server_game.position, game)
        ),
        _lichess_player_color="white",
        _lichess_opponent_name="Remote",
        _lichess_game_controller=None,
        _piece_synchronizer=synchronizer,
        _reference_move_tracker=tracker,
        _latest_frame=np.zeros((20, 20, 3), dtype=np.uint8),
        _calibration_result=SimpleNamespace(
            mapping=SimpleNamespace(camera_grid=np.zeros((9, 9, 2)))
        ),
        _board_quarter_turns=0,
        _board_widget=board_widget,
        _piece_status=MagicMock(),
        _occupancy_status=MagicMock(),
        _lichess_sync_status=MagicMock(),
        _reconcile_button=MagicMock(),
        _reconciliation_requested=True,
        _reconciliation_blackout_pending=False,
        _reconciliation_blackout_generation=0,
        _game_mode="diagnostic",
        _legal_game=LegalGame(),
        _persist_piece_position=MagicMock(),
        _persist_move_reference=MagicMock(),
        _black_projector=MagicMock(),
        _occupancy_detector=MagicMock(),
        _cancel_offline_recovery_for_lichess=MagicMock(),
    )
    window._complete_physical_reconciliation = (
        lambda target_position, lichess_game, completion_message: (
            CameraWindow._complete_physical_reconciliation(
                window,
                target_position,
                lichess_game,
                completion_message,
            )
        )
    )

    CameraWindow._update_physical_reconciliation(
        window,
        frozenset(server_game.position),
    )

    assert synchronizer.position == server_game.position
    assert window._lichess_game_controller is not None
    assert window._lichess_game_controller.server_moves == ("e2e4",)
    assert window._lichess_game_controller.physical_fen == moved.fen
    assert window._lichess_game_controller.state == "synchronized"
    board_widget.set_pieces.assert_called_once_with(server_game.position)
    board_widget.set_reconciliation_squares.assert_called_with(())
    window._persist_piece_position.assert_called_once_with(server_game.position)
    window._persist_move_reference.assert_called_once_with()
    assert window._reconciliation_requested is False


def test_stop_reconciliation_accepts_current_authoritative_position() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from src.chess_camera.camera_app import CameraWindow

    target_position = {(0, 4): "k", (7, 4): "K"}
    lichess_game = object()
    complete = MagicMock()
    window = SimpleNamespace(
        _physical_reconciliation_target=MagicMock(
            return_value=(target_position, lichess_game)
        ),
        _latest_frame=np.zeros((20, 20, 3), dtype=np.uint8),
        _calibration_result=object(),
        _lichess_player_color="white",
        _complete_physical_reconciliation=complete,
    )

    CameraWindow._accept_current_physical_reconciliation(window)

    complete.assert_called_once_with(
        target_position,
        lichess_game,
        "Reconciliation stopped by user; current position accepted.",
    )


def test_active_reconciliation_button_stops_instead_of_restarting() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from src.chess_camera.camera_app import CameraWindow

    accept = MagicMock()
    window = SimpleNamespace(
        _reconciliation_requested=True,
        _accept_current_physical_reconciliation=accept,
    )

    CameraWindow._toggle_physical_reconciliation(window)

    accept.assert_called_once_with()


def test_reconciliation_button_changes_to_stop_action() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from src.chess_camera.camera_app import CameraWindow

    window = SimpleNamespace(
        _reconciliation_requested=False,
        _reconciliation_blackout_pending=False,
        _reconciliation_blackout_generation=0,
        _reconcile_button=MagicMock(),
        _board_widget=MagicMock(),
        _last_rejected_move=object(),
        _black_projector=MagicMock(),
        _occupancy_detector=MagicMock(),
        _reference_move_tracker=MagicMock(),
        _occupancy_status=MagicMock(),
        _piece_status=MagicMock(),
    )

    CameraWindow._begin_physical_reconciliation(window, "Starting.")

    window._reconcile_button.setText.assert_called_once_with(
        "Stop reconciliation"
    )
    assert window._reconciliation_requested is True


def test_reconciliation_projects_and_draws_every_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from src.chess_camera.camera_app import CameraWindow, QTimer
    from src.chess_camera.chess_position import standard_position

    position = standard_position()
    physical = set(position)
    physical.remove((6, 1))
    physical.add((4, 1))
    board_widget = MagicMock()
    projector = MagicMock()
    window = SimpleNamespace(
        _physical_reconciliation_target=MagicMock(
            return_value=(position, None)
        ),
        _reference_move_tracker=SimpleNamespace(reference=None),
        _latest_frame=None,
        _calibration_result=object(),
        _piece_synchronizer=SimpleNamespace(position=position),
        _board_widget=board_widget,
        _piece_status=MagicMock(),
        _occupancy_status=MagicMock(),
        _camera_only_calibration=False,
        _open_projector=MagicMock(return_value=True),
        _physical_board_square=lambda row, column: (row, column),
        _projector_window=projector,
        _alignment_monitor=MagicMock(),
        _projector_status=MagicMock(),
        _finish_reconciliation_projection=MagicMock(),
    )
    monkeypatch.setattr(QTimer, "singleShot", MagicMock())

    CameraWindow._update_physical_reconciliation(
        window,
        frozenset(physical),
    )

    board_widget.set_reconciliation_squares.assert_called_once_with(
        ((6, 1), (4, 1))
    )
    projector.show_reconciliation.assert_called_once_with(
        ((6, 1), (4, 1))
    )


def test_reconciliation_waits_after_red_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from src.chess_camera.camera_app import CameraWindow, QTimer

    callback: Callable[[], None] | None = None

    def capture_timer(
        milliseconds: int,
        pending_callback: Callable[[], None],
    ) -> None:
        nonlocal callback
        assert milliseconds == 1000
        callback = pending_callback

    monkeypatch.setattr(QTimer, "singleShot", capture_timer)
    detector = MagicMock()
    window = SimpleNamespace(
        _projector_window=SimpleNamespace(
            output_mode="reconciliation",
            show_black=MagicMock(),
        ),
        _alignment_monitor=MagicMock(),
        _occupancy_detector=detector,
        _projector_status=MagicMock(),
        _reconciliation_requested=True,
        _reconciliation_blackout_pending=False,
        _reconciliation_blackout_generation=0,
        _finish_reconciliation_blackout=lambda generation: (
            CameraWindow._finish_reconciliation_blackout(
                window,
                generation,
            )
        ),
    )

    CameraWindow._finish_reconciliation_projection(window)

    assert window._reconciliation_blackout_pending
    assert callback is not None
    callback()
    assert not window._reconciliation_blackout_pending
    assert detector.reset.call_count == 2


def test_lichess_attachment_mismatch_starts_live_reconciliation() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from src.chess_camera.camera_app import CameraWindow
    from src.chess_camera.chess_position import PiecePositionSynchronizer
    from src.chess_camera.legal_game import LegalGame
    from src.chess_camera.lichess_client import (
        LichessGameFull,
        LichessGameState,
        LichessPlayer,
    )

    server_game = LegalGame()
    moves = ("e2e4", "e7e6", "g1f3", "d7d5", "b1c3", "c7c6")
    stale_game = LegalGame()
    for uci in moves[:-1]:
        stale_game.apply_move(
            (8 - int(uci[1]), ord(uci[0]) - ord("a")),
            (8 - int(uci[3]), ord(uci[2]) - ord("a")),
        )
    for uci in moves:
        server_game.apply_move(
            (8 - int(uci[1]), ord(uci[0]) - ord("a")),
            (8 - int(uci[3]), ord(uci[2]) - ord("a")),
        )
    state = LichessGameState(
        moves,
        server_game.fen,
        "white",
        0,
        0,
        0,
        0,
        "started",
        None,
        "*",
    )
    game = LichessGameFull(
        "jgWZLWs6",
        LegalGame().fen,
        LichessPlayer("white", "Local", 1500),
        LichessPlayer("black", "Stockfish level 1", None),
        state,
    )
    synchronizer = PiecePositionSynchronizer()
    synchronizer.restore(stale_game.position)
    start_reconciliation = MagicMock()
    window = SimpleNamespace(
        _latest_lichess_game=None,
        _lichess_game_id=game.game_id,
        _lichess_player_color="white",
        _lichess_game_controller=None,
        _reference_move_tracker=SimpleNamespace(initialized=True),
        _piece_synchronizer=synchronizer,
        _lichess_sync_status=MagicMock(),
        _record_runtime_diagnostic=MagicMock(),
        _cancel_offline_recovery_for_lichess=MagicMock(),
        _start_automatic_lichess_reconciliation=start_reconciliation,
    )

    CameraWindow._update_lichess_synchronization(window, game)

    assert window._latest_lichess_game == game
    start_reconciliation.assert_called_once_with()


def test_lichess_attachment_accepts_matching_current_server_position() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from src.chess_camera.camera_app import CameraWindow
    from src.chess_camera.chess_position import PiecePositionSynchronizer
    from src.chess_camera.legal_game import LegalGame
    from src.chess_camera.lichess_client import (
        LichessGameFull,
        LichessGameState,
        LichessPlayer,
    )

    server_game = LegalGame()
    moves = ("e2e4", "e7e5", "g1f3")
    for uci in moves:
        server_game.apply_move(
            (8 - int(uci[1]), ord(uci[0]) - ord("a")),
            (8 - int(uci[3]), ord(uci[2]) - ord("a")),
        )
    state = LichessGameState(
        moves,
        server_game.fen,
        "black",
        0,
        0,
        0,
        0,
        "started",
        None,
        "*",
    )
    game = LichessGameFull(
        "current1",
        "startpos",
        LichessPlayer("white", "Local", 1500),
        LichessPlayer("black", "Remote", 1500),
        state,
    )
    synchronizer = PiecePositionSynchronizer()
    synchronizer.restore(server_game.position)
    start_reconciliation = MagicMock()
    window = SimpleNamespace(
        _latest_lichess_game=None,
        _lichess_game_id=game.game_id,
        _lichess_player_color="white",
        _lichess_opponent_name="Remote",
        _lichess_game_controller=None,
        _reference_move_tracker=SimpleNamespace(initialized=True),
        _piece_synchronizer=synchronizer,
        _lichess_sync_status=MagicMock(),
        _record_runtime_diagnostic=MagicMock(),
        _cancel_offline_recovery_for_lichess=MagicMock(),
        _start_automatic_lichess_reconciliation=start_reconciliation,
        _pending_lichess_local_uci=None,
        _last_projected_lichess_uci=None,
    )
    window._update_lichess_synchronization_state = lambda current: (
        CameraWindow._update_lichess_synchronization_state(window, current)
    )

    CameraWindow._update_lichess_synchronization(window, game)

    start_reconciliation.assert_not_called()
    assert window._lichess_game_controller is not None
    assert window._lichess_game_controller.server_moves == moves
    assert window._lichess_game_controller.state == "synchronized"
    assert window._lichess_game_controller.physical_fen == server_game.fen


def test_automatic_reconciliation_waits_for_safe_live_inputs() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from src.chess_camera.camera_app import CameraWindow

    begin = MagicMock()
    window = SimpleNamespace(
        _reconciliation_requested=False,
        _reset_to_start_requested=False,
        _calibration_result=object(),
        _piece_synchronizer=SimpleNamespace(initialized=True),
        _reference_move_tracker=SimpleNamespace(initialized=True),
        _begin_physical_reconciliation=begin,
    )

    CameraWindow._start_automatic_lichess_reconciliation(window)

    begin.assert_called_once_with(
        "Lichess reconciliation started automatically; checking stable "
        "physical occupancy."
    )
    begin.reset_mock()
    window._reference_move_tracker.initialized = False

    CameraWindow._start_automatic_lichess_reconciliation(window)

    begin.assert_not_called()


def test_polling_rebaseline_restarts_physical_reconciliation() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from src.chess_camera.camera_app import CameraWindow
    from src.chess_camera.legal_game import LegalGame
    from src.chess_camera.lichess_client import LichessGameState
    from src.chess_camera.lichess_game_controller import LichessGameController

    moved_game = LegalGame()
    moved = moved_game.apply_move((6, 4), (4, 4))
    controller = LichessGameController(
        "jgWZLWs6",
        LegalGame().fen,
        "white",
        "Stockfish level 1",
        physical_fen=moved.fen,
    )
    controller.accept_server_history(("e2e4",))
    start_reconciliation = MagicMock()
    window = SimpleNamespace(
        _lichess_game_controller=controller,
        _pending_lichess_local_uci=None,
        _lichess_sync_status=MagicMock(),
        _record_runtime_diagnostic=MagicMock(),
        _start_automatic_lichess_reconciliation=start_reconciliation,
        _last_projected_lichess_uci=None,
    )
    rebased = LichessGameState(
        (),
        moved.fen,
        "black",
        0,
        0,
        0,
        0,
        "started",
        None,
        "*",
    )

    CameraWindow._update_lichess_synchronization_state(window, rebased)

    assert controller.state == "reconciliation-required"
    start_reconciliation.assert_called_once_with()


def test_missing_remembered_projector_does_not_select_replacement() -> None:
    from types import SimpleNamespace

    from src.chess_camera.camera_app import CameraWindow

    primary = SimpleNamespace(name=lambda: "primary")
    secondary = SimpleNamespace(name=lambda: "secondary")

    selected = CameraWindow._default_projector_index(
        [primary, secondary],
        "missing-projector",
    )

    assert selected == -1


def test_explicit_projector_selection_is_persisted() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock, call

    from src.chess_camera.camera_app import CameraWindow

    settings = MagicMock()
    geometry = SimpleNamespace(
        width=lambda: 1280,
        height=lambda: 720,
        x=lambda: 0,
        y=lambda: 0,
    )
    window = SimpleNamespace(
        _calibrator=SimpleNamespace(active=False),
        _projector_selector=SimpleNamespace(currentIndex=lambda: 1),
        _screens=[
            SimpleNamespace(
                name=lambda: "primary",
                geometry=lambda: geometry,
                manufacturer=lambda: "",
                model=lambda: "",
            ),
            SimpleNamespace(
                name=lambda: "remembered-projector",
                geometry=lambda: geometry,
                manufacturer=lambda: "",
                model=lambda: "",
            ),
        ],
        _settings=settings,
        _record_runtime_diagnostic=MagicMock(),
        _restore_saved_calibration=MagicMock(),
    )

    CameraWindow._projector_selection_changed(window)

    assert settings.setValue.call_args_list == [
        call("projector/screen_name", "remembered-projector"),
        call(
            "projector/screen_label",
            "remembered-projector - 1280x720 at (0, 0)",
        ),
    ]
    settings.sync.assert_called_once_with()


def test_projector_screen_refresh_uses_current_enumeration() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from src.chess_camera.camera_app import CameraWindow

    geometry = SimpleNamespace(
        width=lambda: 1280,
        height=lambda: 720,
        x=lambda: 0,
        y=lambda: 0,
    )
    primary = SimpleNamespace(
        name=lambda: "primary",
        geometry=lambda: geometry,
        manufacturer=lambda: "",
        model=lambda: "",
    )
    mini = SimpleNamespace(
        name=lambda: "ABC TV",
        geometry=lambda: geometry,
        manufacturer=lambda: "",
        model=lambda: "",
    )
    selector = MagicMock()
    settings = MagicMock()
    settings.value.return_value = "ABC TV"
    window = SimpleNamespace(
        _settings=settings,
        _projector_selector=selector,
        _screens=[],
        _restore_saved_calibration=MagicMock(),
    )

    CameraWindow._refresh_projector_screens(
        window,
        [primary, mini],
    )

    assert window._screens == [primary, mini]
    assert selector.addItem.call_args_list[-1].args == (
        "ABC TV - 1280x720 at (0, 0)",
    )
    selector.setCurrentIndex.assert_called_once_with(1)
    window._restore_saved_calibration.assert_called_once_with()


def test_reset_to_start_blacks_projector_before_verification() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from src.chess_camera.camera_app import CameraWindow

    window = SimpleNamespace(
        _latest_frame=object(),
        _calibration_result=object(),
        _black_projector=MagicMock(),
        _reset_to_start_requested=False,
        _piece_side_orientation_needs_detection=False,
        _occupancy_detector=MagicMock(),
        _latest_stable_occupied_ui=frozenset(),
        _piece_status=MagicMock(),
        _occupancy_status=MagicMock(),
    )

    CameraWindow._reset_pieces_to_start(window)

    window._black_projector.assert_called_once_with()
    assert window._reset_to_start_requested is True


def test_mapping_recovers_projector_grid_from_dense_measurements() -> None:
    detection = _detection(20, 20, 180, 180)
    height = 200
    width = 200
    camera_y, camera_x = np.indices((height, width), dtype=np.float32)
    camera_points = np.stack((camera_x, camera_y), axis=2)
    expected_homography = np.asarray(
        [
            [1.5, 0.08, 12.0],
            [0.03, 1.2, 8.0],
            [0.0002, 0.0001, 1.0],
        ],
        dtype=np.float32,
    )
    projector_points = cv2.perspectiveTransform(
        camera_points.reshape(-1, 1, 2),
        expected_homography,
    ).reshape(height, width, 2)
    projector_x = np.rint(projector_points[:, :, 0]).astype(np.int32)
    projector_y = np.rint(projector_points[:, :, 1]).astype(np.int32)
    valid = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(
        valid,
        np.rint(detection.outer_corners).astype(np.int32),
        1,
    )

    result = fit_projector_mapping(
        detection,
        projector_x,
        projector_y,
        valid.astype(bool),
        (400, 300),
        coverage_ratio=1.0,
    )
    expected_grid = cv2.perspectiveTransform(
        detection.grid_points.reshape(-1, 1, 2),
        expected_homography,
    ).reshape(9, 9, 2)

    assert result.inlier_ratio > 0.99
    assert result.rms_error < 0.6
    assert np.max(np.abs(result.projector_grid - expected_grid)) < 0.6


def test_mapping_accepts_sparse_distributed_board_plane_support() -> None:
    detection = _detection(20, 20, 180, 180)
    height = 200
    width = 200
    camera_y, camera_x = np.indices((height, width), dtype=np.float32)
    projector_x = np.rint(camera_x * 1.5 + 12).astype(np.int32)
    projector_y = np.rint(camera_y * 1.2 + 8).astype(np.int32)
    board = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(
        board,
        np.rint(detection.outer_corners).astype(np.int32),
        1,
    )
    random_values = np.random.default_rng(7).random((height, width))
    valid = np.logical_and(board > 0, random_values < 0.11)

    result = fit_projector_mapping(
        detection,
        projector_x,
        projector_y,
        valid,
        (400, 300),
        coverage_ratio=0.8,
    )

    assert result.decoded_ratio == pytest.approx(0.11, abs=0.015)
    assert result.inlier_ratio > 0.99


def test_mapping_rejects_concentrated_board_plane_support() -> None:
    detection = _detection(20, 20, 180, 180)
    height = 200
    width = 200
    camera_y, camera_x = np.indices((height, width), dtype=np.float32)
    projector_x = np.rint(camera_x * 1.5 + 12).astype(np.int32)
    projector_y = np.rint(camera_y * 1.2 + 8).astype(np.int32)
    valid = np.zeros((height, width), dtype=bool)
    valid[20:180, 20:45] = True

    with pytest.raises(CalibrationError, match="too concentrated"):
        fit_projector_mapping(
            detection,
            projector_x,
            projector_y,
            valid,
            (400, 300),
            coverage_ratio=0.8,
        )


def test_square_polygons_build_64_inset_quadrilaterals() -> None:
    projector_grid = _detection(0, 0, 800, 800).grid_points

    polygons = square_polygons(projector_grid, inset_fraction=0.05)

    assert len(polygons) == 64
    assert polygons[0] == pytest.approx(
        np.asarray(
            [[5, 5], [95, 5], [95, 95], [5, 95]],
            dtype=np.float32,
        )
    )
    assert interpolate_grid_point(projector_grid, 4.5, 4.5) == pytest.approx(
        np.asarray([450, 450], dtype=np.float32)
    )


def test_refinement_applies_local_validation_corrections() -> None:
    camera_grid = _detection(20, 20, 180, 180).grid_points
    board_points = np.asarray(
        [
            [0.5, 0.5],
            [4.0, 0.5],
            [7.5, 0.5],
            [0.5, 4.0],
            [4.0, 4.0],
            [7.5, 4.0],
            [0.5, 7.5],
            [4.0, 7.5],
            [7.5, 7.5],
        ],
        dtype=np.float32,
    )
    expected_camera = np.asarray(
        [
            interpolate_grid_point(camera_grid, float(x), float(y))
            for x, y in board_points
        ],
        dtype=np.float32,
    )
    measured_camera = expected_camera + np.asarray([10, -5], dtype=np.float32)
    homography = np.asarray(
        [[1.5, 0, 12], [0, 1.2, 8], [0, 0, 1]],
        dtype=np.float32,
    )
    projector_grid = cv2.perspectiveTransform(
        camera_grid.reshape(-1, 1, 2),
        homography,
    ).reshape(9, 9, 2)

    refined = refine_projector_mesh(
        projector_grid,
        homography,
        board_points,
        expected_camera,
        measured_camera,
        (400, 300),
    )

    expected = projector_grid + np.asarray([-15, 6], dtype=np.float32)
    np.testing.assert_allclose(refined, expected, atol=0.01)


def test_mapping_remaps_board_motion_with_local_correction() -> None:
    old_grid = _detection(20, 20, 180, 180).grid_points
    homography = np.asarray(
        [[1.5, 0.05, 12], [0.02, 1.2, 8], [0, 0, 1]],
        dtype=np.float32,
    )
    old_projector_grid = cv2.perspectiveTransform(
        old_grid.reshape(-1, 1, 2),
        homography,
    ).reshape(9, 9, 2)
    local_correction = np.asarray([2, -1], dtype=np.float32)
    old_projector_grid += local_correction
    mapping = MappingResult(
        camera_grid=old_grid,
        projector_grid=old_projector_grid,
        camera_to_projector_homography=homography,
        coverage_ratio=1.0,
        decoded_ratio=1.0,
        inlier_ratio=1.0,
        rms_error=0.0,
        maximum_error=0.0,
    )
    moved_grid = old_grid + np.asarray([6, 4], dtype=np.float32)

    remapped = remap_projector_mapping(
        mapping,
        moved_grid,
        (400, 300),
    )
    expected = cv2.perspectiveTransform(
        moved_grid.reshape(-1, 1, 2),
        homography,
    ).reshape(9, 9, 2)
    expected += local_correction

    np.testing.assert_allclose(remapped.projector_grid, expected, atol=0.01)
    np.testing.assert_allclose(remapped.camera_grid, moved_grid)


def test_quick_realign_recovers_minor_geometry_change() -> None:
    camera_grid = _detection(20, 20, 180, 180).grid_points
    old_homography = np.asarray(
        [[1.5, 0.02, 12], [0.01, 1.2, 8], [0, 0, 1]],
        dtype=np.float32,
    )
    local_correction = np.asarray([2, -1], dtype=np.float32)
    old_projector_grid = cv2.perspectiveTransform(
        camera_grid.reshape(-1, 1, 2),
        old_homography,
    ).reshape(9, 9, 2)
    old_projector_grid += local_correction
    mapping = MappingResult(
        camera_grid=camera_grid,
        projector_grid=old_projector_grid,
        camera_to_projector_homography=old_homography,
        coverage_ratio=1.0,
        decoded_ratio=1.0,
        inlier_ratio=1.0,
        rms_error=0.0,
        maximum_error=0.0,
    )
    board_coordinates = np.asarray(
        [
            [0.5, 0.5],
            [7.5, 0.5],
            [3.5, 3.5],
            [0.5, 7.5],
            [7.5, 7.5],
        ],
        dtype=np.float32,
    )
    true_homography = np.asarray(
        [[1.48, 0.03, 20], [0.015, 1.19, 3], [0, 0, 1]],
        dtype=np.float32,
    )
    inverse_true = np.linalg.inv(true_homography).astype(np.float32)
    commanded = np.asarray(
        [
            interpolate_grid_point(old_projector_grid, float(x), float(y))
            for x, y in board_coordinates
        ],
        dtype=np.float32,
    )
    measured = cv2.perspectiveTransform(
        (commanded - local_correction).reshape(-1, 1, 2),
        inverse_true,
    ).reshape(-1, 2)

    realigned = quick_realign_mapping(
        mapping,
        board_coordinates,
        measured,
        (400, 300),
    )
    expected_grid = cv2.perspectiveTransform(
        camera_grid.reshape(-1, 1, 2),
        true_homography,
    ).reshape(9, 9, 2)
    expected_grid += local_correction

    np.testing.assert_allclose(
        realigned.projector_grid,
        expected_grid,
        atol=0.02,
    )
    np.testing.assert_allclose(
        realigned.camera_to_projector_homography,
        true_homography,
        atol=0.001,
    )


def test_quick_realign_uses_independent_validation_phase() -> None:
    class FakeProjector:
        def __init__(self) -> None:
            self.grid: np.ndarray | None = None

        def set_projector_grid(self, grid: np.ndarray) -> None:
            self.grid = grid.copy()

    detection = _detection()
    mapping = MappingResult(
        camera_grid=detection.grid_points,
        projector_grid=detection.grid_points,
        camera_to_projector_homography=np.eye(3, dtype=np.float32),
        coverage_ratio=1.0,
        decoded_ratio=1.0,
        inlier_ratio=1.0,
        rms_error=0.0,
        maximum_error=0.0,
    )
    projector = FakeProjector()
    controller = ProjectorCalibrationController(projector, object())
    fitting_coordinates = [
        (1.0, 2.0),
        (7.0, 2.0),
        (1.0, 6.0),
        (7.0, 6.0),
        (4.0, 4.0),
    ]
    measured = [
        interpolate_grid_point(detection.grid_points, x, y)
        for x, y in fitting_coordinates
    ]
    controller._active = True
    controller._quick_verification = True
    controller._quick_phase = "fit"
    controller._projector_size = (200, 200)
    controller._detection = detection
    controller._mapping = mapping
    controller._validation_coordinates = fitting_coordinates
    controller._validation_expected = [point.copy() for point in measured]
    controller._validation_measured = [point.copy() for point in measured]
    controller._validation_widths = [10.0] * len(measured)
    controller._captures["validation_black"] = np.zeros(
        (100, 100),
        dtype=np.uint8,
    )
    controller._show_validation_black = lambda: None

    controller._finish_validation()

    assert controller._quick_phase == "validate"
    assert controller._validation_coordinates == [
        (4.0, 2.0),
        (7.0, 4.0),
        (4.0, 6.0),
        (1.0, 4.0),
    ]
    assert controller._validation_index == 0
    assert "validation_black" not in controller._captures
    assert projector.grid is not None


def test_validation_targets_avoid_starting_piece_centers() -> None:
    detection = _detection()
    mapping = MappingResult(
        camera_grid=detection.grid_points,
        projector_grid=detection.grid_points,
        camera_to_projector_homography=np.eye(3, dtype=np.float32),
        coverage_ratio=1.0,
        decoded_ratio=1.0,
        inlier_ratio=1.0,
        rms_error=0.0,
        maximum_error=0.0,
    )
    controller = ProjectorCalibrationController(object(), object())
    controller._detection = detection
    controller._mapping = mapping
    controller._show_validation_black = lambda: None

    controller._start_validation()

    occupied_centers = np.asarray(
        [
            (column + 0.5, row + 0.5)
            for row in (0, 1, 6, 7)
            for column in range(8)
        ],
        dtype=np.float32,
    )
    targets = np.asarray(
        controller._validation_coordinates,
        dtype=np.float32,
    )
    distances = np.linalg.norm(
        targets[:, np.newaxis, :] - occupied_centers[np.newaxis, :, :],
        axis=2,
    )

    assert targets.shape == (9, 2)
    assert float(np.min(distances)) > 0.7
    assert float(np.min(targets)) == pytest.approx(1.0)
    assert float(np.max(targets)) == pytest.approx(7.0)


def test_missing_validation_target_tries_nearby_candidate() -> None:
    controller = ProjectorCalibrationController(object(), object())
    controller._validation_candidates = [
        [(1.0, 1.0), (1.25, 1.0)]
    ]
    controller._validation_coordinates = [(1.0, 1.0)]
    controller._validation_expected = [
        np.asarray([20.0, 20.0], dtype=np.float32)
    ]
    controller._validation_widths = [20.0]
    controller._validation_index = 0
    controller._captures["validation_black"] = np.zeros(
        (50, 50),
        dtype=np.uint8,
    )
    scheduled: list[Callable[[], None]] = []
    controller._schedule = lambda _delay, callback: scheduled.append(
        callback
    )

    controller._accept_validation_capture(
        np.zeros((50, 50), dtype=np.uint8)
    )

    assert controller._validation_coordinates == [(1.25, 1.0)]
    assert controller._validation_expected == []
    assert controller._validation_widths == []
    assert controller._validation_candidate_index == 1
    assert len(scheduled) == 1


def test_validation_only_reuses_mapping_without_refinement() -> None:
    class FakeProjector:
        def __init__(self) -> None:
            self.grid: np.ndarray | None = None

        def set_projector_grid(self, grid: np.ndarray) -> None:
            self.grid = grid.copy()

    detection = _detection()
    mapping = MappingResult(
        camera_grid=detection.grid_points,
        projector_grid=detection.grid_points + 5.0,
        camera_to_projector_homography=np.eye(3, dtype=np.float32),
        coverage_ratio=1.0,
        decoded_ratio=1.0,
        inlier_ratio=1.0,
        rms_error=0.0,
        maximum_error=0.0,
    )
    empty_points = np.empty((0, 2), dtype=np.float32)
    result = CalibrationResult(
        board_detection=detection,
        mapping=mapping,
        validation=ValidationResult(
            measured_points=empty_points,
            expected_points=empty_points,
            errors=np.empty(0, dtype=np.float32),
            mean_error=0.0,
            rms_error=0.0,
            maximum_error=0.0,
            maximum_square_fraction=0.0,
        ),
    )
    projector = FakeProjector()
    controller = ProjectorCalibrationController(projector, object())
    controller._show_validation_black = lambda: None

    controller.start_validation_only(result, 200, 200)

    assert controller.validation_only
    assert not controller.quick_verification
    assert controller._mapping is mapping
    assert len(controller._validation_coordinates) == 9
    assert projector.grid is not None
    np.testing.assert_array_equal(projector.grid, mapping.projector_grid)

    controller._validation_coordinates = [
        candidates[-1] for candidates in controller._validation_candidates
    ]
    controller._validation_expected = [
        np.asarray([20.0, 20.0], dtype=np.float32)
    ] * 9
    controller._validation_measured = [
        np.asarray([22.0, 20.0], dtype=np.float32)
    ] * 9
    controller._validation_widths = [10.0] * 9

    with pytest.raises(CalibrationError, match="20.0%"):
        controller._finish_validation()

    assert controller._refinement_iteration == 0
    assert controller._mapping is mapping


def test_excessive_validation_error_tries_nearby_candidate() -> None:
    detection = _detection()
    mapping = MappingResult(
        camera_grid=detection.grid_points,
        projector_grid=detection.grid_points,
        camera_to_projector_homography=np.eye(3, dtype=np.float32),
        coverage_ratio=1.0,
        decoded_ratio=1.0,
        inlier_ratio=1.0,
        rms_error=0.0,
        maximum_error=0.0,
    )
    controller = ProjectorCalibrationController(
        object(),
        object(),
        maximum_refinement_iterations=0,
    )
    controller._mapping = mapping
    controller._detection = detection
    controller._validation_candidates = [
        [(1.0, 1.0), (1.25, 1.0)]
    ] + [[(float(index), 4.0)] for index in range(2, 10)]
    controller._validation_coordinates = [
        candidates[0] for candidates in controller._validation_candidates
    ]
    controller._validation_expected = [
        np.asarray([20.0, 20.0], dtype=np.float32)
    ] * 9
    controller._validation_measured = [
        np.asarray([24.0, 20.0], dtype=np.float32)
    ] + [
        np.asarray([20.0, 20.0], dtype=np.float32)
    ] * 8
    controller._validation_widths = [20.0] * 9
    scheduled: list[Callable[[], None]] = []
    controller._schedule = lambda _delay, callback: scheduled.append(
        callback
    )

    controller._finish_validation()

    assert controller._validation_coordinates[0] == (1.25, 1.0)
    assert controller._validation_expected == []
    assert controller._validation_measured == []
    assert controller._validation_widths == []
    assert len(scheduled) == 1


def test_validation_target_and_metrics_measure_camera_error() -> None:
    black = np.full((100, 100), 20, dtype=np.uint8)
    illuminated = black.copy()
    cv2.circle(illuminated, (52, 48), 4, 150, -1)
    expected = np.asarray([50, 50], dtype=np.float32)

    measured = locate_validation_target(
        black,
        illuminated,
        expected,
        search_radius=15,
    )
    result = validation_result(
        measured[np.newaxis, :],
        expected[np.newaxis, :],
        np.asarray([40], dtype=np.float32),
    )

    assert measured == pytest.approx(np.asarray([52, 48]), abs=0.1)
    assert result.maximum_error == pytest.approx(np.sqrt(8), abs=0.1)
    assert result.maximum_square_fraction < 0.1


def test_calibration_store_persists_mesh_and_validation(tmp_path) -> None:
    detection = _detection()
    errors = np.asarray([1.0], dtype=np.float32)
    result = CalibrationResult(
        board_detection=detection,
        mapping=MappingResult(
            camera_grid=detection.grid_points,
            projector_grid=detection.grid_points,
            camera_to_projector_homography=np.eye(3, dtype=np.float32),
            coverage_ratio=0.95,
            decoded_ratio=0.9,
            inlier_ratio=0.99,
            rms_error=0.4,
            maximum_error=1.2,
        ),
        validation=ValidationResult(
            measured_points=np.asarray([[51, 50]], dtype=np.float32),
            expected_points=np.asarray([[50, 50]], dtype=np.float32),
            errors=errors,
            mean_error=1.0,
            rms_error=1.0,
            maximum_error=1.0,
            maximum_square_fraction=0.025,
        ),
    )

    saved = CalibrationStore(tmp_path).save(
        result,
        camera_name="Camera",
        camera_id="camera-id",
        camera_resolution=(1920, 1080),
        projector_name="Projector",
        projector_resolution=(1280, 720),
    )
    metadata = json.loads(saved.read_text(encoding="utf-8"))

    assert metadata["projector"]["width"] == 1280
    np.testing.assert_allclose(
        np.asarray(metadata["mapping"]["projector_grid"]),
        detection.grid_points,
    )
    assert metadata["validation"]["maximum_square_fraction"] == 0.025
    assert CalibrationStore(tmp_path).saved_projector_display() == (
        "Projector",
        (1280, 720),
    )

    loaded = CalibrationStore(tmp_path).load_matching(
        camera_id="camera-id",
        camera_resolution=(1920, 1080),
        projector_name="Projector",
        projector_resolution=(1280, 720),
    )
    assert loaded is not None
    np.testing.assert_allclose(
        loaded.mapping.projector_grid,
        detection.grid_points,
    )
    camera_only = CalibrationStore(tmp_path).load_matching(
        camera_id="camera-id",
        camera_resolution=(1920, 1080),
        projector_name=None,
        projector_resolution=None,
    )
    assert camera_only is not None
    np.testing.assert_allclose(
        camera_only.mapping.camera_grid,
        detection.grid_points,
    )
    assert (
        CalibrationStore(tmp_path).load_matching(
            camera_id="different-camera",
            camera_resolution=(1920, 1080),
            projector_name="Projector",
            projector_resolution=(1280, 720),
        )
        is None
    )


def test_calibration_store_archives_invalidated_mapping(tmp_path) -> None:
    store = CalibrationStore(tmp_path)
    source = tmp_path / "latest.json"
    source.write_text('{"schema_version": 1}', encoding="utf-8")

    archived = store.invalidate()

    assert archived is not None
    assert archived.name.startswith("invalidated-")
    assert archived.read_text(encoding="utf-8") == '{"schema_version": 1}'
    assert not source.exists()
    assert store.saved_projector_display() is None
