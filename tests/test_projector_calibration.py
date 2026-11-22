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
    board_circle_polygon,
    clearance_squares,
    coded_marker_pattern,
    decode_gray_code,
    detect_coded_marker_correspondences,
    detect_board_orientation,
    expanded_projector_board_mask,
    fit_coded_pattern_mapping,
    fit_projector_mapping,
    fit_visible_middle_lattice,
    gray_code_bit_count,
    gray_code_pair_quality,
    gray_code_pattern,
    interpolate_grid_point,
    locate_validation_target,
    measure_coverage,
    quick_realign_mapping,
    remap_projector_mapping,
    refine_projector_mesh,
    require_board_plane_support,
    select_board_plane_support,
    select_validation_squares,
    square_polygons,
    support_map_pattern,
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


_PIECE_SCENE_HOMOGRAPHY = np.asarray(
    [[1.4, 0.05, 30.0], [-0.03, 1.35, 25.0], [0.00004, 0.00002, 1.0]],
    dtype=np.float64,
)


def _piece_scene(
    occupied_rows: tuple[int, ...],
) -> tuple[BoardDetection, np.ndarray, np.ndarray, np.ndarray]:
    detection = _detection(40, 40, 360, 360)
    camera_y, camera_x = np.mgrid[0:400, 0:400].astype(np.float64)
    points = np.stack((camera_x, camera_y, np.ones_like(camera_x)), axis=2)
    projected = points @ _PIECE_SCENE_HOMOGRAPHY.T
    projector_x = projected[:, :, 0] / projected[:, :, 2]
    projector_y = projected[:, :, 1] / projected[:, :, 2]
    for row in occupied_rows:
        top = 40 + 40 * row
        # A piece and its occlusion fill the square with smoothly varying,
        # off-plane projector coordinates from the piece's height.
        rise = (camera_y[top : top + 40, 40:360] - top) * 0.3
        projector_x[top : top + 40, 40:360] += 9 + rise
        projector_y[top : top + 40, 40:360] -= 12 + rise
    quantized_x = (np.floor(projector_x / 2) * 2 + 1).astype(np.int32)
    quantized_y = (np.floor(projector_y / 2) * 2 + 1).astype(np.int32)
    valid = np.zeros((400, 400), dtype=bool)
    valid[42:358, 42:358] = True
    return detection, quantized_x, quantized_y, valid


def _true_projector_grid(detection: BoardDetection) -> np.ndarray:
    return cv2.perspectiveTransform(
        detection.grid_points.reshape(-1, 1, 2).astype(np.float64),
        _PIECE_SCENE_HOMOGRAPHY,
    ).reshape(9, 9, 2)


def test_plane_support_calibrates_starting_position_with_pieces() -> None:
    detection, projector_x, projector_y, valid = _piece_scene((0, 1, 6, 7))
    with pytest.raises(CalibrationError, match="inliers"):
        fit_projector_mapping(
            detection,
            projector_x,
            projector_y,
            valid,
            (1280, 720),
            1.0,
        )

    selection = select_board_plane_support(
        detection,
        projector_x,
        projector_y,
        valid,
        (1280, 720),
        empty_squares=np.ones((8, 8), dtype=bool),
    )
    require_board_plane_support(selection)
    mapping = fit_projector_mapping(
        detection,
        projector_x,
        projector_y,
        selection.mask,
        (1280, 720),
        1.0,
    )

    retained_rows = {
        row for row, _column in np.argwhere(
            selection.square_status == "retained"
        )
    }
    assert retained_rows == {2, 3, 4, 5}
    assert not np.any(selection.square_status[[0, 1, 6, 7]] == "retained")
    error = np.linalg.norm(
        mapping.projector_grid - _true_projector_grid(detection),
        axis=2,
    )
    assert float(np.max(error)) < 1.0


def test_plane_support_ignores_wrong_occupancy_prior() -> None:
    detection, projector_x, projector_y, valid = _piece_scene((0, 1, 6, 7))
    prior = np.zeros((8, 8), dtype=bool)
    prior[0:2, :] = True

    selection = select_board_plane_support(
        detection,
        projector_x,
        projector_y,
        valid,
        (1280, 720),
        empty_squares=prior,
    )

    assert not selection.occupancy_used
    assert selection.sufficiently_spread
    assert selection.details()["retained_square_count"] == 32


def test_plane_support_rejects_too_few_board_plane_squares() -> None:
    detection, projector_x, projector_y, valid = _piece_scene(
        (0, 1, 2, 3, 6, 7)
    )

    selection = select_board_plane_support(
        detection,
        projector_x,
        projector_y,
        valid,
        (1280, 720),
    )

    with pytest.raises(CalibrationError, match="board squares"):
        require_board_plane_support(selection)


def test_expanded_projector_board_mask_limits_illumination() -> None:
    grid = _detection(40, 30, 120, 110).grid_points

    mask = expanded_projector_board_mask(
        (160, 140),
        grid,
        margin_square_fraction=0.5,
    )

    assert mask.shape == (140, 160)
    assert mask[70, 80]
    assert mask[25, 80]
    assert not mask[0, 0]
    assert not mask[-1, -1]


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
    assert window._camera_geometry_ready is True

    CameraWindow._activate_calibration_result(
        window,
        result,
        live_verified=False,
    )

    assert window._camera_geometry_ready is False


def test_completed_calibration_returns_to_black_without_grid_preview() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock, patch

    from src.chess_camera.camera_app import CameraWindow

    window = SimpleNamespace(
        _activate_calibration_result=MagicMock(),
        _alignment_monitor=MagicMock(),
        _calibration_progress=MagicMock(),
        _latest_frame=None,
        _projector_window=MagicMock(),
        _restore_remembered_square=MagicMock(),
        _startup_calibration_running=False,
    )

    with patch(
        "src.chess_camera.camera_app.QTimer.singleShot"
    ) as single_shot:
        CameraWindow._projector_calibration_completed(window, object())

    single_shot.assert_not_called()
    window._projector_window.show_black.assert_called_once_with()
    window._projector_window.set_calibration_overlays_suppressed.assert_called_once_with(
        False
    )
    window._alignment_monitor.set_black.assert_called_once_with()
    window._restore_remembered_square.assert_called_once_with()


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
        _piece_side_orientation_needs_detection=True,
        _piece_synchronizer=SimpleNamespace(position={}),
        _settings=MagicMock(),
        _refresh_pending_lichess_guidance=MagicMock(),
    )

    with patch(
        "src.chess_camera.camera_app.detect_board_orientation",
        return_value=0,
    ):
        CameraWindow._detect_current_board_orientation(window)

    assert window._board_quarter_turns == 2
    assert window._board_orientation_needs_detection is False
    assert window._piece_side_orientation_needs_detection is False
    window._settings.setValue.assert_not_called()
    window._settings.sync.assert_not_called()
    window._refresh_pending_lichess_guidance.assert_called_once_with()


def test_parity_detection_keeps_starting_side_check_pending() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock, patch

    from src.chess_camera.camera_app import CameraWindow
    from src.chess_camera.legal_game import LegalGame

    window = SimpleNamespace(
        _board_orientation_needs_detection=True,
        _latest_frame=np.zeros((20, 20, 3), dtype=np.uint8),
        _calibration_result=SimpleNamespace(
            mapping=SimpleNamespace(
                camera_grid=np.zeros((9, 9, 2), dtype=np.float32)
            )
        ),
        _board_quarter_turns=2,
        _piece_side_orientation_needs_detection=True,
        _piece_synchronizer=SimpleNamespace(
            position=LegalGame().position
        ),
        _settings=MagicMock(),
        _refresh_pending_lichess_guidance=MagicMock(),
    )

    with patch(
        "src.chess_camera.camera_app.detect_board_orientation",
        return_value=0,
    ):
        CameraWindow._detect_current_board_orientation(window)

    assert window._board_quarter_turns == 2
    assert window._board_orientation_needs_detection is False
    assert window._piece_side_orientation_needs_detection is True
    window._refresh_pending_lichess_guidance.assert_not_called()


def test_square_projection_waits_for_white_side_orientation() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from src.chess_camera.camera_app import CameraWindow

    window = SimpleNamespace(
        _calibration_result=object(),
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


@pytest.mark.parametrize(
    ("coded_pattern_mode", "adaptive_cropped_mode"),
    [(True, False), (False, True)],
)
def test_failed_experimental_calibration_retains_previous_mapping(
    coded_pattern_mode: bool,
    adaptive_cropped_mode: bool,
) -> None:
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
            coded_pattern_mode=coded_pattern_mode,
            adaptive_cropped_mode=adaptive_cropped_mode,
            diagnostic_captures={},
            diagnostic_details={},
            support_map=None,
            clearance_squares=(),
        ),
        _startup_calibration_running=False,
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


def test_startup_calibration_starts_after_camera_settles() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock, patch

    from src.chess_camera.camera_app import CameraWindow

    window = SimpleNamespace(
        _startup_calibration_pending=True,
        _startup_calibration_ready_at=None,
        _startup_calibration_running=False,
        _calibrator=SimpleNamespace(active=False),
        _start_adaptive_projector_calibration=MagicMock(return_value=True),
    )

    with patch(
        "src.chess_camera.camera_app.monotonic",
        side_effect=[10.0, 11.0, 12.5],
    ):
        for _frame in range(3):
            CameraWindow._advance_startup_calibration(window)

    window._start_adaptive_projector_calibration.assert_called_once_with(
        startup=True
    )
    assert window._startup_calibration_running is True
    assert window._startup_calibration_pending is False


def test_startup_calibration_failure_projects_support_map() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock, patch

    from src.chess_camera.camera_app import CameraWindow

    support_map = np.zeros((720, 1280, 3), dtype=np.uint8)
    window = SimpleNamespace(
        _projector_window=MagicMock(),
        _startup_calibration_running=True,
        _calibrator=SimpleNamespace(
            diagnostic_details={"plane_support": {}},
            diagnostic_captures={},
            support_map=support_map,
            clearance_squares=((2, 3),),
        ),
        _record_runtime_diagnostic=MagicMock(),
        _activate_calibration_result=MagicMock(),
        _invalidate_calibration=MagicMock(),
        _set_calibration_indicator=MagicMock(),
        _projector_calibration_indicator=object(),
        _projector_status=MagicMock(),
    )

    with patch("src.chess_camera.camera_app.QMessageBox") as message_box:
        CameraWindow._projector_calibration_failed(window, "too few squares")

    message_box.warning.assert_not_called()
    window._activate_calibration_result.assert_not_called()
    window._invalidate_calibration.assert_called_once_with()
    window._projector_window.show_pattern.assert_called_once_with(support_map)
    status = window._projector_status.setText.call_args.args[0]
    assert "Remove the pieces from the red squares" in status
    assert window._startup_calibration_running is False
    failure = window._record_runtime_diagnostic.call_args_list[0]
    assert failure.kwargs["details"]["startup"] is True
    assert "plane_support" in failure.kwargs["details"]


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
        _offline_move_marking_active=False,
        _offline_move_marking_blackout_pending=False,
        _offline_move_marking_generation=0,
        _illegal_correction_generation=0,
        _resume_board_edge_activity_indicator=MagicMock(),
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


def test_reconciliation_attaches_and_submits_unique_local_reply() -> None:
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
    from src.chess_camera.reconciliation import matching_legal_successors

    initial_fen = "4k3/3b3p/8/8/8/8/P3B3/4K3 b - - 0 1"
    server_game = LegalGame(initial_fen)
    opponent = server_game.apply_move((1, 3), (3, 1))
    reply_game = LegalGame(opponent.fen)
    reply = reply_game.apply_move((6, 4), (3, 1))
    game = LichessGameFull(
        "game1234",
        initial_fen,
        LichessPlayer("white", "Local", 1500),
        LichessPlayer("black", "Remote", 1500),
        LichessGameState(
            ("d7b5",),
            opponent.fen,
            "white",
            0,
            0,
            0,
            0,
            "started",
            None,
            "*",
        ),
    )
    match = matching_legal_successors(
        opponent.fen,
        frozenset(reply.position),
        {(3, 1): "white", (6, 4): "empty"},
        allow_promotion=True,
    )[0]
    synchronizer = PiecePositionSynchronizer()
    synchronizer.restore(LegalGame(initial_fen).position)
    tracker = MagicMock()
    tracker.reference = None
    begin_submission = MagicMock(return_value=True)
    board_widget = MagicMock()
    window = SimpleNamespace(
        _physical_reconciliation_target=MagicMock(
            return_value=(opponent.position, game)
        ),
        _latest_frame=np.zeros((20, 20, 3), dtype=np.uint8),
        _calibration_result=SimpleNamespace(
            mapping=SimpleNamespace(camera_grid=np.zeros((9, 9, 2)))
        ),
        _lichess_player_color="white",
        _lichess_opponent_name="Remote",
        _lichess_client=object(),
        _lichess_submission_thread=None,
        _pending_lichess_local_move=None,
        _pending_promotion_selection=None,
        _promotion_mode="auto-queen",
        _reference_move_tracker=tracker,
        _board_quarter_turns=0,
        _piece_synchronizer=synchronizer,
        _legal_game=LegalGame(),
        _lichess_game_controller=None,
        _cancel_offline_recovery_for_lichess=MagicMock(),
        _board_widget=board_widget,
        _piece_status=MagicMock(),
        _persist_piece_position=MagicMock(),
        _lichess_sync_status=MagicMock(),
        _reconciliation_requested=True,
        _reconciliation_blackout_pending=False,
        _reconciliation_reference_acceptance_pending=False,
        _reconciliation_blackout_generation=2,
        _reconcile_button=MagicMock(),
        _black_projector=MagicMock(),
        _occupancy_detector=MagicMock(),
        _occupancy_status=MagicMock(),
        _begin_lichess_local_move_submission=begin_submission,
    )

    CameraWindow._update_physical_reconciliation(
        window,
        frozenset(reply.position),
    )

    assert window._lichess_game_controller.server_moves == ("d7b5",)
    assert window._lichess_game_controller.physical_fen == opponent.fen
    assert synchronizer.position == opponent.position
    tracker.stage_accepted_reference.assert_called_once()
    begin_submission.assert_called_once_with(
        ((6, 4), (3, 1)),
        "e2b5",
        match.result,
    )
    board_widget.set_reconciliation_squares.assert_called_with(())
    assert window._reconciliation_requested is False


def test_stop_reconciliation_waits_for_clean_reference_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from src.chess_camera.camera_app import CameraWindow, QTimer

    target_position = {(0, 4): "k", (7, 4): "K"}
    lichess_game = object()
    callback: Callable[[], None] | None = None

    def capture_timer(
        milliseconds: int,
        pending_callback: Callable[[], None],
    ) -> None:
        nonlocal callback
        assert milliseconds == 1000
        callback = pending_callback

    monkeypatch.setattr(QTimer, "singleShot", capture_timer)
    window = SimpleNamespace(
        _physical_reconciliation_target=MagicMock(
            return_value=(target_position, lichess_game)
        ),
        _latest_frame=np.zeros((20, 20, 3), dtype=np.uint8),
        _calibration_result=object(),
        _lichess_player_color="white",
        _reconciliation_requested=True,
        _reconciliation_reference_acceptance_pending=False,
        _reconciliation_blackout_generation=3,
        _reconciliation_blackout_pending=False,
        _black_projector=MagicMock(),
        _occupancy_detector=MagicMock(),
        _reference_move_tracker=MagicMock(),
        _board_widget=MagicMock(),
        _piece_status=MagicMock(),
        _occupancy_status=MagicMock(),
        _projector_status=MagicMock(),
        _finish_reconciliation_blackout=MagicMock(),
    )
    window._queue_clean_reconciliation_reference_capture = lambda: (
        CameraWindow._queue_clean_reconciliation_reference_capture(window)
    )

    CameraWindow._accept_current_physical_reconciliation(window)

    assert window._reconciliation_reference_acceptance_pending is True
    assert window._reconciliation_blackout_pending is True
    assert callback is not None
    window._black_projector.assert_called_once_with()
    window._occupancy_detector.reset.assert_called_once_with()
    window._board_widget.set_reconciliation_squares.assert_called_once_with(())
    callback()
    window._finish_reconciliation_blackout.assert_called_once_with(4)


def test_manual_reconciliation_accepts_next_stable_clean_frame() -> None:
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
        _reconciliation_reference_acceptance_pending=True,
        _complete_physical_reconciliation=complete,
        _board_widget=MagicMock(),
        _piece_status=MagicMock(),
        _occupancy_status=MagicMock(),
    )

    CameraWindow._update_physical_reconciliation(
        window, frozenset()
    )

    complete.assert_called_once_with(
        target_position,
        lichess_game,
        "Current physical position accepted; clean move reference captured.",
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


def test_reconciliation_waits_for_initial_blackout(
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
        _finish_reconciliation_blackout=MagicMock(),
    )
    monkeypatch.setattr(QTimer, "singleShot", capture_timer)

    CameraWindow._begin_physical_reconciliation(window, "Starting.")

    window._reconcile_button.setText.assert_called_once_with(
        "Stop reconciliation"
    )
    assert window._reconciliation_requested is True
    assert window._reconciliation_blackout_pending is True
    assert callback is not None
    callback()
    window._finish_reconciliation_blackout.assert_called_once_with(1)


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


def test_new_lichess_game_preserves_complete_orientation() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from src.chess_camera.camera_app import CameraWindow
    from src.chess_camera.lichess_client import AccountGameEvent

    event = AccountGameEvent(
        "gameStart",
        "new-game",
        "new-game-full",
        "white",
        "startpos",
        "Opponent",
        True,
    )
    window = SimpleNamespace(
        _pending_created_lichess_game_id=None,
        _lichess_game_id="old-game",
        _lichess_game_controller=object(),
        _latest_lichess_game=object(),
        _last_projected_lichess_uci="e2e4",
        _draw_offer_game_id="old-game",
        _responded_draw_offer_game_id="old-game",
        _lichess_draw_offer_state_available=True,
        _abort_requested_game_id="old-game",
        _resign_requested_game_id="old-game",
        _board_orientation_needs_detection=False,
        _piece_side_orientation_needs_detection=False,
        _reconciliation_requested=True,
        _black_projector=MagicMock(),
        _clear_lichess_game_information=MagicMock(),
        _start_new_lichess_game_button=MagicMock(),
        _update_lichess_game_control_buttons=MagicMock(),
        _lichess_status=MagicMock(),
        _record_runtime_diagnostic=MagicMock(),
    )

    CameraWindow._lichess_account_event_received(window, event)

    assert window._board_orientation_needs_detection is False
    assert window._piece_side_orientation_needs_detection is False
    assert window._reconciliation_requested is True
    window._black_projector.assert_not_called()
    window._clear_lichess_game_information.assert_called_once_with(
        "Loading Lichess game information."
    )


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
        _request_startup_calibration=MagicMock(),
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
        _request_startup_calibration=MagicMock(),
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
    window._request_startup_calibration.assert_called_once_with()


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


def test_board_circle_polygon_prewarps_affine_grid() -> None:
    rows, columns = np.indices((9, 9), dtype=np.float32)
    grid = np.stack(
        (
            100 + columns * 30 - rows * 4,
            50 + columns * 6 + rows * 18,
        ),
        axis=2,
    )

    polygon = board_circle_polygon(grid, 4.0, 3.0, 0.2)
    center = interpolate_grid_point(grid, 4.0, 3.0)

    np.testing.assert_allclose(
        polygon[0],
        center + np.asarray([6.0, 1.2]),
        atol=0.001,
    )
    np.testing.assert_allclose(
        polygon[8],
        center + np.asarray([-0.8, 3.6]),
        atol=0.001,
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


def test_adaptive_calibration_waits_for_stable_discovery_frames() -> None:
    from unittest.mock import MagicMock, patch

    class FakeProjector:
        def __init__(self) -> None:
            self.patterns: list[np.ndarray] = []
            self.blackouts = 0

        def show_pattern(self, pattern: np.ndarray) -> None:
            self.patterns.append(pattern.copy())

        def show_black(self) -> None:
            self.blackouts += 1

    detector = MagicMock()
    detection = _detection()
    detector.detect.return_value = detection
    projector = FakeProjector()
    controller = ProjectorCalibrationController(
        projector,
        detector,
        settle_milliseconds=1000,
        frames_per_state=3,
    )
    delays: list[int] = []
    controller._schedule = lambda delay, _callback: delays.append(delay)

    controller.start_adaptive_cropped(320, 180)

    assert controller.adaptive_cropped_mode
    assert delays == [1000]
    controller._step_index = 1
    controller._display_current_step()
    assert delays[-1] == 250

    controller._collecting = True
    controller._frame_accumulator = None
    controller._frame_count = 0
    controller._captures["black"] = np.zeros((60, 80), dtype=np.uint8)
    controller._detection = detection
    detector.detect.reset_mock()
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
    with patch(
        "src.chess_camera.calibration_controller.fit_coded_pattern_mapping",
        return_value=(detection, mapping),
    ):
        for _index in range(2):
            controller.accept_frame(
                np.zeros((60, 80, 3), dtype=np.uint8)
            )
            assert controller._collecting
        for value in (100, 200, 201):
            controller.accept_frame(
                np.full((60, 80, 3), value, dtype=np.uint8)
            )
            assert controller._collecting
        controller.accept_frame(
            np.full((60, 80, 3), 199, dtype=np.uint8)
        )
        assert not controller._collecting
        assert projector.blackouts == 1
        assert delays[-1] == 25
        assert detector.detect.call_count == 0
        controller._fit_adaptive_discovery_candidate()

    assert not controller._collecting
    assert controller._adaptive_rough_mapping is mapping
    assert np.all(controller._captures["coded"] == 200)

    controller._adaptive_phase = "gray-code"
    projector_mask = np.ones((180, 320), dtype=bool)
    controller._adaptive_projector_mask = projector_mask
    controller._adaptive_previous_gray = np.zeros(
        (60, 80), dtype=np.uint8
    )
    controller._steps = controller._build_cropped_gray_code_steps(
        projector_mask
    )
    controller._step_index = 0
    controller._display_current_step()
    assert delays[-1] == 0
    controller._collecting = True
    controller._frame_accumulator = None
    controller._frame_count = 0
    controller.accept_frame(np.zeros((60, 80, 3), dtype=np.uint8))
    assert controller._collecting
    controller.accept_frame(np.full((60, 80, 3), 100, dtype=np.uint8))
    assert controller._collecting
    controller.accept_frame(np.full((60, 80, 3), 102, dtype=np.uint8))
    assert not controller._collecting
    assert np.all(controller._captures["x_0_normal"] == 101)


def test_adaptive_discovery_fit_failure_stops_without_reflashing() -> None:
    from unittest.mock import MagicMock, patch

    projector = MagicMock()
    controller = ProjectorCalibrationController(projector, MagicMock())
    controller._schedule = lambda _delay, _callback: None
    controller._active = True
    controller._collecting = False
    controller._adaptive_cropped_mode = True
    controller._adaptive_phase = "discovery"
    controller._projector_size = (320, 180)
    controller._captures["black"] = np.zeros((60, 80), dtype=np.uint8)
    controller._detection = _detection()
    controller._coded_pattern = coded_marker_pattern(320, 180)
    controller._adaptive_discovery_candidate = np.full(
        (60, 80),
        100,
        dtype=np.uint8,
    )
    failures: list[str] = []
    controller.failed.connect(failures.append)

    with patch(
        "src.chess_camera.calibration_controller.fit_coded_pattern_mapping",
        side_effect=CalibrationError("No markers detected."),
    ):
        controller._fit_adaptive_discovery_candidate()

    assert not controller.active
    assert failures == ["Coded discovery failed: No markers detected."]
    projector.show_pattern.assert_not_called()
    projector.show_black.assert_called_once_with()


def test_adaptive_gray_code_patterns_are_black_outside_board_mask() -> None:
    controller = ProjectorCalibrationController(object(), object())
    controller._projector_size = (32, 24)
    mask = np.zeros((24, 32), dtype=bool)
    mask[6:18, 8:24] = True

    steps = controller._build_cropped_gray_code_steps(mask)

    assert steps
    for step in steps:
        assert np.count_nonzero(step.pattern[~mask]) == 0
    assert any(np.count_nonzero(step.pattern[mask]) > 0 for step in steps)


def test_adaptive_gray_code_falls_back_to_three_frame_average() -> None:
    controller = ProjectorCalibrationController(
        object(),
        object(),
        frames_per_state=3,
    )
    controller._schedule = lambda _delay, _callback: None
    controller._active = True
    controller._collecting = True
    controller._adaptive_cropped_mode = True
    controller._adaptive_phase = "gray-code"
    controller._projector_size = (32, 24)
    controller._steps = controller._build_cropped_gray_code_steps(
        np.ones((24, 32), dtype=bool)
    )
    controller._step_index = 0

    controller._enable_adaptive_gray_fallback(0, 0)

    assert controller._adaptive_gray_fallback
    for value in (10, 20):
        controller.accept_frame(
            np.full((12, 16, 3), value, dtype=np.uint8)
        )
        assert controller._collecting
    controller.accept_frame(np.full((12, 16, 3), 30, dtype=np.uint8))

    assert not controller._collecting
    assert np.all(controller._captures["x_0_normal"] == 20)


def _striped_pattern(stripe_width: int, size: int = 128) -> np.ndarray:
    columns = (np.arange(size) // stripe_width) % 2 == 1
    return np.tile(np.where(columns, 255, 0).astype(np.uint8), (size, 1))


def _captured(pattern: np.ndarray) -> np.ndarray:
    return np.where(pattern > 0, 200, 50).astype(np.uint8)


def test_gray_code_pair_quality_accepts_matching_pair() -> None:
    pattern = _striped_pattern(32)
    quality = gray_code_pair_quality(
        _captured(pattern),
        _captured(255 - pattern),
        np.ones(pattern.shape, dtype=bool),
        pattern,
        np.eye(3),
    )

    assert quality.contrast == 1.0
    assert quality.agreement == 1.0
    assert quality.acceptable


def test_gray_code_pair_quality_rejects_stale_normal_capture() -> None:
    pattern = _striped_pattern(32)
    stale = _striped_pattern(16)
    quality = gray_code_pair_quality(
        _captured(stale),
        _captured(255 - pattern),
        np.ones(pattern.shape, dtype=bool),
        pattern,
        np.eye(3),
    )

    assert quality.contrast >= 0.3
    assert quality.agreement is not None
    assert quality.agreement == pytest.approx(0.5, abs=0.05)
    assert not quality.acceptable


def test_gray_code_pair_quality_rejects_collapsed_pair() -> None:
    pattern = _striped_pattern(32)
    capture = np.full(pattern.shape, 120, dtype=np.uint8)
    quality = gray_code_pair_quality(
        capture,
        capture,
        np.ones(pattern.shape, dtype=bool),
        pattern,
        np.eye(3),
    )

    assert quality.contrast == 0.0
    assert not quality.acceptable


def test_clearance_lights_unretained_central_squares() -> None:
    from src.chess_camera.projector_calibration import PlaneSupportSelection

    status = np.full((8, 8), "retained", dtype="<U16")
    status[2, 3] = "occupied"
    status[5, 5] = "plane-outlier"
    status[0, 0] = "occupied"
    selection = PlaneSupportSelection(
        mask=np.zeros((10, 10), dtype=bool),
        interior_mask=np.zeros((10, 10), dtype=bool),
        plane_inliers=np.zeros((10, 10), dtype=bool),
        square_status=status,
        filter_counts={},
        occupancy_used=False,
    )
    squares = clearance_squares(selection)
    grid = _detection(0, 0, 80, 80).grid_points
    usable = tuple(
        (int(row), int(column))
        for row, column in np.argwhere(status == "retained")
    )

    pattern = support_map_pattern((90, 90), grid, usable, squares)

    assert squares == ((2, 3), (5, 5))
    assert tuple(pattern[25, 35]) == (255, 0, 0)
    assert tuple(pattern[55, 55]) == (255, 0, 0)
    assert tuple(pattern[15, 15]) == (0, 255, 0)
    assert tuple(pattern[5, 5]) == (0, 0, 0)
    assert tuple(pattern[25, 40]) == (0, 0, 0)


def test_validation_squares_avoid_unretained_squares() -> None:
    status = np.full((8, 8), "retained", dtype="<U16")
    assert len(select_validation_squares(status)) == 13

    status[3, 3] = "occupied"
    status[1, 1] = "occupied"
    squares = select_validation_squares(status)

    assert len(squares) == 13
    assert len(set(squares)) == 13
    assert all(status[row, column] == "retained" for row, column in squares)


def _adaptive_validation_controller(
    hit_count: int,
) -> tuple[ProjectorCalibrationController, list[str], list[object]]:
    from unittest.mock import MagicMock

    controller = ProjectorCalibrationController(MagicMock(), MagicMock())
    controller._schedule = lambda _delay, _callback: None
    controller._active = True
    controller._adaptive_cropped_mode = True
    detection = _detection()
    controller._detection = detection
    controller._mapping = MappingResult(
        camera_grid=detection.grid_points,
        projector_grid=detection.grid_points,
        camera_to_projector_homography=np.eye(3, dtype=np.float32),
        coverage_ratio=1.0,
        decoded_ratio=1.0,
        inlier_ratio=1.0,
        rms_error=0.0,
        maximum_error=0.0,
    )
    squares = select_validation_squares(
        np.full((8, 8), "retained", dtype="<U16")
    )
    controller._validation_coordinates = [
        (column + 0.5, row + 0.5) for row, column in squares
    ]
    for index, (x, y) in enumerate(controller._validation_coordinates):
        expected = np.asarray([10 + x * 10, 10 + y * 10], dtype=np.float32)
        controller._validation_expected.append(expected)
        controller._validation_widths.append(10.0)
        if index < hit_count:
            measured = expected + 0.5
        elif index % 2:
            measured = np.full(2, np.nan, dtype=np.float32)
        else:
            measured = expected + 4.0
        controller._validation_measured.append(measured)
    failures: list[str] = []
    completed: list[object] = []
    controller.failed.connect(failures.append)
    controller.completed.connect(completed.append)
    return controller, failures, completed


def test_adaptive_validation_tolerates_targets_on_pieces() -> None:
    from unittest.mock import patch

    controller, failures, completed = _adaptive_validation_controller(10)

    with patch.object(
        controller,
        "_refine_and_repeat_validation",
    ) as refine:
        controller._finish_validation()

    refine.assert_not_called()
    assert not failures
    assert len(completed) == 1
    assert len(completed[0].validation.errors) == 10
    assert sum(entry["hit"] for entry in controller._validation_log) == 10


def test_adaptive_validation_fails_with_too_few_hits() -> None:
    controller, _failures, completed = _adaptive_validation_controller(8)

    with pytest.raises(CalibrationError, match="Only 8 of 13"):
        controller._finish_validation()

    assert not completed


def _controller_at_first_gray_inverse() -> tuple[
    ProjectorCalibrationController,
    list[str],
    list[str],
]:
    from unittest.mock import MagicMock

    controller = ProjectorCalibrationController(MagicMock(), MagicMock())
    controller._schedule = lambda _delay, _callback: None
    controller._active = True
    controller._adaptive_cropped_mode = True
    controller._adaptive_phase = "gray-code"
    controller._projector_size = (100, 100)
    controller._detection = _detection()
    controller._adaptive_rough_homography = np.eye(3)
    controller._steps = controller._build_cropped_gray_code_steps(
        np.ones((100, 100), dtype=bool)
    )
    controller._step_index = next(
        index
        for index, step in enumerate(controller._steps)
        if step.key.endswith("_inverse")
    )
    failures: list[str] = []
    statuses: list[str] = []
    controller.failed.connect(failures.append)
    controller.status_changed.connect(statuses.append)
    return controller, failures, statuses


def test_unreliable_gray_code_pair_is_recaptured_then_fails() -> None:
    controller, failures, statuses = _controller_at_first_gray_inverse()
    inverse_index = controller._step_index
    inverse_step = controller._steps[inverse_index]
    pair_key = inverse_step.key.removesuffix("_inverse")
    collapsed = np.full((100, 100), 120, dtype=np.uint8)

    for attempt in range(2):
        controller._captures[f"{pair_key}_normal"] = collapsed
        controller._complete_frame_capture(collapsed)
        assert controller._step_index == inverse_index - 1
        assert len(statuses) == attempt + 1
        assert not failures
        controller._step_index = inverse_index

    controller._complete_frame_capture(collapsed)

    assert len(failures) == 1
    axis, bit = pair_key.split("_")
    assert f"{axis.upper()} Gray-code bit {int(bit) + 1}" in failures[0]
    assert "after 2 recaptures" in failures[0]
    assert [entry["accepted"] for entry in controller._gray_pair_log] == [
        False,
        False,
        False,
    ]


def test_reliable_gray_code_pair_advances() -> None:
    controller, failures, statuses = _controller_at_first_gray_inverse()
    inverse_index = controller._step_index
    normal_step = controller._steps[inverse_index - 1]
    inverse_step = controller._steps[inverse_index]
    controller._captures[normal_step.key] = _captured(normal_step.pattern)

    controller._complete_frame_capture(_captured(inverse_step.pattern))

    assert controller._step_index == inverse_index + 1
    assert not failures
    assert not statuses
    assert controller._gray_pair_log[-1]["accepted"] is True


def test_forced_validation_capture_after_gray_code_is_not_logged() -> None:
    controller = ProjectorCalibrationController(
        object(),
        object(),
        frames_per_state=1,
    )
    controller._schedule = lambda _delay, _callback: None
    controller._active = True
    controller._collecting = True
    controller._adaptive_cropped_mode = True
    controller._adaptive_phase = "gray-code"
    controller._adaptive_gray_fallback = True
    controller._projector_size = (32, 24)
    controller._steps = controller._build_cropped_gray_code_steps(
        np.ones((24, 32), dtype=bool)
    )
    controller._step_index = len(controller._steps)
    controller._validation_index = 0
    accepted: list[np.ndarray] = []
    controller._accept_validation_capture = accepted.append

    controller.accept_frame(np.full((12, 16, 3), 30, dtype=np.uint8))

    assert len(accepted) == 1
    assert controller._gray_capture_log == []


def test_adaptive_final_mapping_uses_dense_gray_code() -> None:
    from types import SimpleNamespace
    from unittest.mock import patch

    class FakeProjector:
        def __init__(self) -> None:
            self.grid: np.ndarray | None = None

        def set_projector_grid(self, grid: np.ndarray) -> None:
            self.grid = grid.copy()

    projector = FakeProjector()
    controller = ProjectorCalibrationController(projector, object())
    controller._projector_size = (8, 8)
    controller._detection = _detection(1, 1, 8, 8)
    controller._captures["black"] = np.zeros((10, 10), dtype=np.uint8)
    for axis in ("x", "y"):
        for bit in range(2):
            for suffix in ("normal", "inverse"):
                controller._captures[f"{axis}_{bit}_{suffix}"] = np.zeros(
                    (10, 10),
                    dtype=np.uint8,
                )
    mapping = MappingResult(
        camera_grid=controller._detection.grid_points,
        projector_grid=controller._detection.grid_points,
        camera_to_projector_homography=np.eye(3, dtype=np.float32),
        coverage_ratio=1.0,
        decoded_ratio=1.0,
        inlier_ratio=1.0,
        rms_error=0.0,
        maximum_error=0.0,
    )
    decoded = np.zeros((10, 10), dtype=np.int32)
    valid = np.ones((10, 10), dtype=bool)
    selection = SimpleNamespace(
        mask=np.zeros((10, 10), dtype=bool),
        sufficiently_spread=True,
    )

    with (
        patch(
            "src.chess_camera.calibration_controller.decode_gray_code",
            return_value=(decoded, decoded, valid),
        ) as decode,
        patch(
            "src.chess_camera.calibration_controller."
            "select_board_plane_support",
            return_value=selection,
        ) as select,
        patch(
            "src.chess_camera.calibration_controller.plane_support_overlay",
            return_value=np.zeros((10, 10, 3), dtype=np.uint8),
        ),
        patch(
            "src.chess_camera.calibration_controller."
            "require_board_plane_support",
        ),
        patch(
            "src.chess_camera.calibration_controller.fit_projector_mapping",
            return_value=mapping,
        ) as fit,
    ):
        controller._calculate_adaptive_mapping()

    decode.assert_called_once()
    select.assert_called_once()
    fit.assert_called_once()
    assert fit.call_args.args[3] is selection.mask
    assert controller._mapping is mapping
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

    loaded = CalibrationStore(tmp_path).load_for_projector(
        camera_id="camera-id",
        camera_resolution=(1920, 1080),
        projector_name="Projector",
        projector_resolution=(1280, 720),
    )
    assert loaded is not None
    assert loaded.camera_matches
    np.testing.assert_allclose(
        loaded.result.mapping.projector_grid,
        detection.grid_points,
    )
    changed_camera = CalibrationStore(tmp_path).load_for_projector(
        camera_id="different-camera",
        camera_resolution=(1280, 720),
        projector_name="Projector",
        projector_resolution=(1280, 720),
    )
    assert changed_camera is not None
    assert not changed_camera.camera_matches
    assert (
        CalibrationStore(tmp_path).load_for_projector(
            camera_id="camera-id",
            camera_resolution=(1920, 1080),
            projector_name="Other projector",
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


def test_calibration_store_backs_up_mapping_before_replacement(
    tmp_path,
) -> None:
    store = CalibrationStore(tmp_path)
    assert store.backup("before-recovery") is None
    source = tmp_path / "latest.json"
    source.write_text('{"schema_version": 1}', encoding="utf-8")

    backup = store.backup("before-recovery")

    assert backup is not None
    assert backup.name.startswith("before-recovery-")
    assert backup.read_text(encoding="utf-8") == '{"schema_version": 1}'
    assert source.exists()


def test_failed_alignment_recovery_is_reported_without_dialog() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock, patch

    from src.chess_camera.camera_app import CameraWindow

    window = SimpleNamespace(
        _projector_window=MagicMock(),
        _invalidate_calibration=MagicMock(),
        _set_calibration_indicator=MagicMock(),
        _camera_calibration_indicator=object(),
        _projector_status=MagicMock(),
        _record_runtime_diagnostic=MagicMock(),
    )
    window._projector_status.text.return_value = "status"
    images = {"recovery-black": np.zeros((4, 4), dtype=np.uint8)}

    with patch("src.chess_camera.camera_app.QMessageBox") as message_box:
        CameraWindow._alignment_recovery_failed(
            window,
            "No match.",
            {"attempts": 5},
            images,
        )

    assert not message_box.mock_calls
    window._projector_window.show_black.assert_called_once_with()
    window._invalidate_calibration.assert_called_once_with()
    window._record_runtime_diagnostic.assert_called_once_with(
        "alignment-recovery-failed",
        "status",
        details={"attempts": 5},
        include_images=True,
        additional_images=images,
    )
