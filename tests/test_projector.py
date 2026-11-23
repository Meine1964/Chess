import os
from unittest.mock import patch

import numpy as np
import pytest
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QApplication

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from src.chess_camera.projector import (
    DEFAULT_BOARD_ILLUMINATION_PERCENT,
    PROJECTED_LEGAL_MOVE,
    PROJECTED_ON_DARK_SQUARE,
    PROJECTED_ILLEGAL_MOVE,
    PROJECTED_ON_LIGHT_SQUARE,
    PROMOTION_GLYPHS,
    ProjectorWindow,
    board_focus_pattern,
    focus_pattern,
)


@pytest.fixture(scope="module")
def application() -> QApplication:
    existing = QApplication.instance()
    if isinstance(existing, QApplication):
        return existing
    return QApplication([])


def test_focus_pattern_has_checkerboard_cross_and_border() -> None:
    pattern = focus_pattern(80, 48, cell_size=8)

    assert pattern.shape == (48, 80)
    assert pattern.dtype == np.uint8
    assert pattern[4, 4] == 0
    assert pattern[4, 12] == 255
    assert np.all(pattern[0, :] == 255)
    assert np.all(pattern[:, 0] == 255)
    assert pattern[24, 20] == 255
    assert pattern[12, 40] == 255


def test_board_focus_pattern_is_limited_to_enlarged_board_area() -> None:
    columns, rows = np.meshgrid(
        np.linspace(400, 720, 9, dtype=np.float32),
        np.linspace(200, 520, 9, dtype=np.float32),
    )
    grid = np.stack((columns, rows), axis=2)

    pattern = board_focus_pattern(1280, 720, grid)

    full = focus_pattern(1280, 720)
    np.testing.assert_array_equal(pattern[200:520, 400:720], full[200:520, 400:720])
    assert np.any(pattern[165:195, 365:395])
    assert not np.any(pattern[:150, :])
    assert not np.any(pattern[:, 780:])


def test_projector_window_stays_above_screen_savers(
    application: QApplication,
) -> None:
    window = ProjectorWindow()

    assert (
        window.windowFlags() & Qt.WindowType.WindowStaysOnTopHint
        == Qt.WindowType.WindowStaysOnTopHint
    )


def test_projector_grid_update_repaints_active_output(
    application: QApplication,
) -> None:
    window = ProjectorWindow()
    grid = np.zeros((9, 9, 2), dtype=np.float32)

    with patch.object(window, "update") as update:
        window.set_projector_grid(grid)

    update.assert_called_once_with()


def test_projector_illuminates_complete_board_at_configured_level(
    application: QApplication,
) -> None:
    window = ProjectorWindow()
    window.resize(320, 180)
    x_values = np.linspace(80, 240, 9, dtype=np.float32)
    y_values = np.linspace(10, 170, 9, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    grid = np.stack((x_grid, y_grid), axis=2).astype(np.float32)
    window.set_projector_grid(grid)
    window.show_black()
    window.show()
    application.processEvents()

    image = window.grab().toImage()
    expected = round(255 * DEFAULT_BOARD_ILLUMINATION_PERCENT / 100)
    assert image.pixelColor(160, 90).red() == expected
    assert image.pixelColor(40, 90).red() == 0

    window.set_board_illumination_percent(35)
    application.processEvents()

    assert window.grab().toImage().pixelColor(160, 90).red() == 89


def test_calibration_suppression_makes_illuminated_board_black(
    application: QApplication,
) -> None:
    window = ProjectorWindow()
    window.resize(320, 180)
    coordinates = np.linspace(40, 140, 9, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(coordinates, coordinates)
    window.set_projector_grid(
        np.stack((x_grid, y_grid), axis=2).astype(np.float32)
    )
    window.show_black()
    window.set_calibration_overlays_suppressed(True)
    window.show()
    application.processEvents()

    assert window.grab().toImage().pixelColor(90, 90).red() == 0


def test_promotion_icons_rotate_toward_physical_viewer_edge() -> None:
    assert ProjectorWindow._promotion_rotation("top") == 180.0
    assert ProjectorWindow._promotion_rotation("right") == -90.0
    assert ProjectorWindow._promotion_rotation("bottom") == 0.0
    assert ProjectorWindow._promotion_rotation("left") == 90.0


def test_promotion_icons_use_bright_filled_silhouettes() -> None:
    assert PROMOTION_GLYPHS == {
        "n": "♞",
        "b": "♝",
        "r": "♜",
        "q": "♛",
    }


def test_promotion_icon_transform_maps_canonical_square_to_perspective() -> None:
    polygon = np.asarray(
        ((20, 10), (90, 5), (110, 80), (10, 70)),
        dtype=np.float32,
    )

    transform = ProjectorWindow._promotion_glyph_transform(
        polygon,
        "top",
        100.0,
    )

    mapped = (
        transform.map(QPointF(0, 0)),
        transform.map(QPointF(100, 0)),
        transform.map(QPointF(100, 100)),
        transform.map(QPointF(0, 100)),
    )
    for actual, expected in zip(mapped, polygon, strict=True):
        assert actual.x() == pytest.approx(float(expected[0]))
        assert actual.y() == pytest.approx(float(expected[1]))


def test_turn_indicator_prewarps_board_plane_circle() -> None:
    x_values = np.linspace(40, 280, 9, dtype=np.float32)
    y_values = np.linspace(20, 160, 9, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    grid = np.stack((x_grid, y_grid), axis=2).astype(np.float32)

    polygon = ProjectorWindow._turn_indicator_polygon(
        grid,
        "bottom-right",
    )

    center = np.mean(polygon, axis=0)
    assert center[0] == pytest.approx(295.0)
    assert center[1] == pytest.approx(168.75)
    assert polygon[0, 0] - center[0] == pytest.approx(4.2, abs=0.0001)
    assert polygon[8, 1] - center[1] == pytest.approx(
        2.45,
        abs=0.0001,
    )


def test_edge_activity_dot_is_projected_outside_h4_in_every_mode(
    application: QApplication,
) -> None:
    window = ProjectorWindow()
    window.resize(320, 180)
    window.show()
    x_values = np.linspace(40, 280, 9, dtype=np.float32)
    y_values = np.linspace(20, 160, 9, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    grid = np.stack((x_grid, y_grid), axis=2).astype(np.float32)
    window.set_projector_grid(grid)

    polygon = ProjectorWindow._edge_activity_indicator_polygon(grid)
    np.testing.assert_allclose(
        np.mean(polygon, axis=0),
        (295.0, 98.75),
        atol=0.0001,
    )

    window.show_black()
    application.processEvents()
    assert window.grab().toImage().pixelColor(295, 99).red() == 0

    window.set_edge_activity_indicator(True)
    application.processEvents()
    assert window.grab().toImage().pixelColor(295, 99) == PROJECTED_LEGAL_MOVE

    window.show_square(2, 3)
    application.processEvents()
    assert window.grab().toImage().pixelColor(295, 99) == PROJECTED_LEGAL_MOVE


def test_edge_clock_polygons_follow_logical_a_file_orientation() -> None:
    x_values = np.linspace(80, 280, 9, dtype=np.float32)
    y_values = np.linspace(10, 170, 9, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    grid = np.stack((x_grid, y_grid), axis=2).astype(np.float32)

    white = ProjectorWindow._edge_clock_polygon(
        grid,
        6,
        8,
        board_quarter_turns=2,
    )
    black = ProjectorWindow._edge_clock_polygon(
        grid,
        0,
        2,
        board_quarter_turns=2,
    )

    np.testing.assert_allclose(
        white,
        ((280, 50), (280, 10), (305, 10), (305, 50)),
    )
    np.testing.assert_allclose(
        black,
        ((280, 170), (280, 130), (305, 130), (305, 170)),
    )


@pytest.mark.parametrize("board_quarter_turns", range(4))
@pytest.mark.parametrize("rows", ((0, 2), (6, 8)))
def test_edge_clock_transform_never_reflects_glyphs(
    board_quarter_turns: int,
    rows: tuple[int, int],
) -> None:
    x_values = np.linspace(80, 280, 9, dtype=np.float32)
    y_values = np.linspace(10, 170, 9, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    grid = np.stack((x_grid, y_grid), axis=2).astype(np.float32)

    polygon = ProjectorWindow._edge_clock_polygon(
        grid,
        *rows,
        board_quarter_turns=board_quarter_turns,
    )
    along_edge = polygon[1] - polygon[0]
    outward = polygon[3] - polygon[0]
    signed_area = (
        along_edge[0] * outward[1]
        - along_edge[1] * outward[0]
    )

    assert signed_area > 0


def test_edge_clock_content_rotates_and_shrinks_around_fixed_center() -> None:
    transform = ProjectorWindow._edge_clock_content_transform(256.0, 128.0)

    center = transform.map(QPointF(128.0, 64.0))
    original_top = transform.map(QPointF(128.0, 0.0))
    original_bottom = transform.map(QPointF(128.0, 128.0))

    assert center.x() == pytest.approx(128.0)
    assert center.y() == pytest.approx(64.0)
    assert original_top.y() == pytest.approx(108.8)
    assert original_bottom.y() == pytest.approx(19.2)
    assert original_top.y() - original_bottom.y() == pytest.approx(
        128.0 * 0.7
    )


def test_edge_clock_content_transform_precedes_perspective_warp() -> None:
    polygon = np.asarray(
        ((80, 130), (80, 170), (55, 170), (55, 130)),
        dtype=np.float32,
    )

    transform = ProjectorWindow._edge_clock_paint_transform(
        polygon,
        256.0,
        128.0,
    )
    center = transform.map(QPointF(128.0, 64.0))
    original_top = transform.map(QPointF(128.0, 0.0))

    assert center.x() == pytest.approx(67.5)
    assert center.y() == pytest.approx(150.0)
    assert original_top.x() == pytest.approx(58.75)
    assert original_top.y() == pytest.approx(150.0)


def test_edge_clock_compensates_lower_projector_pixel_density() -> None:
    dense_polygon = np.asarray(
        ((10, 10), (266, 10), (266, 138), (10, 138)),
        dtype=np.float32,
    )
    sparse_polygon = np.asarray(
        ((320, 20), (512, 20), (512, 116), (320, 116)),
        dtype=np.float32,
    )

    dense_scale, sparse_scale = ProjectorWindow._edge_stroke_compensation(
        [dense_polygon, sparse_polygon]
    )

    assert dense_scale == pytest.approx((1.0, 1.0))
    assert sparse_scale == pytest.approx((4 / 3, 4 / 3))
    capped_scale = ProjectorWindow._edge_stroke_compensation(
        [
            dense_polygon,
            np.asarray(
                ((320, 20), (448, 20), (448, 84), (320, 84)),
                dtype=np.float32,
            ),
        ]
    )[1]
    assert capped_scale == pytest.approx((1.5, 1.5))

    image = QImage(540, 160, QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.black)
    painter = QPainter(image)
    ProjectorWindow._draw_edge_clock(
        painter,
        dense_polygon,
        "1",
        stroke_scale=dense_scale,
    )
    ProjectorWindow._draw_edge_clock(
        painter,
        sparse_polygon,
        "1",
        stroke_scale=sparse_scale,
    )
    painter.end()

    dense_columns = {
        x
        for x in range(10, 267)
        if any(image.pixelColor(x, y).red() > 150 for y in range(10, 139))
    }
    sparse_columns = {
        x
        for x in range(320, 513)
        if any(image.pixelColor(x, y).red() > 150 for y in range(20, 117))
    }

    assert abs(len(dense_columns) - len(sparse_columns)) <= 1


def test_fractional_clock_text_renders(application: QApplication) -> None:
    window = ProjectorWindow()
    window.resize(360, 180)
    window.show()
    x_values = np.linspace(80, 280, 9, dtype=np.float32)
    y_values = np.asarray((10, 20, 30, 40, 80, 120, 130, 140, 150), dtype=np.float32)
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    window.set_projector_grid(np.stack((x_grid, y_grid), axis=2))
    window.set_edge_clocks("00:09.2", "00:08.45", board_quarter_turns=0)
    window.show_black()
    application.processEvents()

    image = window.grab().toImage()
    assert any(
        image.pixelColor(x, y).red() > 0
        for x in range(50, 82)
        for y in range(5, 35)
    )


def test_result_visibility_does_not_change_clock_rasterization(
    application: QApplication,
) -> None:
    window = ProjectorWindow()
    window.resize(360, 180)
    window.show()
    x_values = np.linspace(80, 280, 9, dtype=np.float32)
    y_values = np.asarray(
        (10, 20, 30, 40, 80, 120, 130, 140, 150),
        dtype=np.float32,
    )
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    grid = np.stack((x_grid, y_grid), axis=2).astype(np.float32)
    window.set_projector_grid(grid)
    window.set_edge_clocks(
        "1:05",
        "9:42",
        board_quarter_turns=0,
    )
    window.show_black()
    application.processEvents()
    active_game = window.grab().toImage()

    window.set_edge_clocks(
        "1:05",
        "9:42",
        board_quarter_turns=0,
        result="1-0",
    )
    application.processEvents()
    terminal_game = window.grab().toImage()

    for left, top, right, bottom in (
        (50, 5, 82, 35),
        (50, 125, 82, 155),
    ):
        for x_coordinate in range(left, right + 1):
            for y_coordinate in range(top, bottom + 1):
                assert active_game.pixelColor(
                    x_coordinate,
                    y_coordinate,
                ) == terminal_game.pixelColor(
                    x_coordinate,
                    y_coordinate,
                )


def test_projector_renders_seven_segment_clocks_outside_board(
    application: QApplication,
) -> None:
    window = ProjectorWindow()
    window.resize(360, 180)
    x_values = np.linspace(80, 280, 9, dtype=np.float32)
    y_values = np.linspace(10, 170, 9, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    grid = np.stack((x_grid, y_grid), axis=2).astype(np.float32)
    window.set_projector_grid(grid)
    window.set_edge_clocks(
        "1:05",
        "9:42",
        board_quarter_turns=0,
    )
    window.show_black()
    window.show()
    application.processEvents()

    image = window.grab().toImage()
    black_clock_pixels = sum(
        image.pixelColor(x, y).red() > 150
        for x in range(55, 81)
        for y in range(10, 51)
    )
    white_clock_pixels = sum(
        image.pixelColor(x, y).red() > 150
        for x in range(55, 81)
        for y in range(130, 171)
    )

    assert black_clock_pixels > 30
    assert white_clock_pixels > 30
    assert image.pixelColor(100, 90).red() == 51


def test_calibration_suppresses_clocks_and_activity_indicator(
    application: QApplication,
) -> None:
    from unittest.mock import patch

    window = ProjectorWindow()
    window.resize(360, 180)
    x_values = np.linspace(80, 280, 9, dtype=np.float32)
    y_values = np.linspace(10, 170, 9, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    grid = np.stack((x_grid, y_grid), axis=2).astype(np.float32)
    window.set_projector_grid(grid)
    window.set_edge_clocks("1:05", "9:42", board_quarter_turns=0)
    window.set_edge_activity_indicator(True)
    window.show_black()
    window.show()

    with patch.object(
        ProjectorWindow,
        "_draw_edge_clock",
    ) as draw_edge_clock:
        window.set_calibration_overlays_suppressed(True)
        application.processEvents()
        suppressed = window.grab().toImage()

    indicator_center = np.rint(
        np.mean(
            ProjectorWindow._edge_activity_indicator_polygon(grid),
            axis=0,
        )
    ).astype(int)
    assert draw_edge_clock.call_count == 0
    assert suppressed.pixelColor(
        int(indicator_center[0]),
        int(indicator_center[1]),
    ).red() == 0

    with patch.object(
        ProjectorWindow,
        "_draw_edge_clock",
    ) as draw_edge_clock:
        window.set_calibration_overlays_suppressed(False)
        application.processEvents()

    assert draw_edge_clock.call_count == 2
    window.close()


def test_projector_places_timeout_result_between_clocks(
    application: QApplication,
) -> None:
    window = ProjectorWindow()
    window.resize(360, 180)
    x_values = np.linspace(100, 260, 9, dtype=np.float32)
    y_values = np.linspace(30, 150, 9, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    grid = np.stack((x_grid, y_grid), axis=2).astype(np.float32)
    window.set_projector_grid(grid)
    window.set_turn_indicator("bottom-left")
    window.set_edge_clocks(
        "00:00",
        "09:58",
        board_quarter_turns=1,
        result="0-1",
        expired_player="white",
    )
    window.show_black()
    window.show()
    application.processEvents()

    image = window.grab().toImage()
    result_pixels = sum(
        image.pixelColor(x, y).red() > 150
        for x in range(166, 195)
        for y in range(10, 36)
    )

    assert result_pixels > 20
    winner_digit_pixels = sum(
        image.pixelColor(x, y).red() > 150
        for x in range(184, 193)
        for y in range(10, 36)
    )
    loser_digit_pixels = sum(
        image.pixelColor(x, y).red() > 150
        for x in range(168, 177)
        for y in range(10, 36)
    )
    assert winner_digit_pixels < loser_digit_pixels
    assert image.pixelColor(180, 90).red() == 51
    assert image.pixelColor(90, 158).red() == 0
    expiration_flag_pixels = sum(
        image.pixelColor(x, y).red() > 150
        for x in range(135, 166)
        for y in range(10, 41)
    )
    assert expiration_flag_pixels > 10


def test_projector_places_terminal_result_without_clocks(
    application: QApplication,
) -> None:
    window = ProjectorWindow()
    window.resize(360, 180)
    x_values = np.linspace(100, 260, 9, dtype=np.float32)
    y_values = np.linspace(30, 150, 9, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    grid = np.stack((x_grid, y_grid), axis=2).astype(np.float32)
    window.set_projector_grid(grid)
    window.set_turn_indicator("bottom-left")
    window.set_edge_clocks(
        None,
        None,
        board_quarter_turns=1,
        result="0-1",
    )
    window.show_black()
    window.show()
    application.processEvents()

    image = window.grab().toImage()
    result_pixels = sum(
        image.pixelColor(x, y).red() > 150
        for x in range(166, 195)
        for y in range(10, 36)
    )
    black_clock_pixels = sum(
        image.pixelColor(x, y).red() > 150
        for x in range(135, 166)
        for y in range(10, 41)
    )
    white_clock_pixels = sum(
        image.pixelColor(x, y).red() > 150
        for x in range(195, 226)
        for y in range(10, 41)
    )

    assert result_pixels > 20
    assert black_clock_pixels == 0
    assert white_clock_pixels == 0
    assert image.pixelColor(90, 158).red() == 0


def test_projector_places_agreed_draw_result_between_clocks(
    application: QApplication,
) -> None:
    window = ProjectorWindow()
    window.resize(360, 180)
    x_values = np.linspace(100, 260, 9, dtype=np.float32)
    y_values = np.linspace(30, 150, 9, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    grid = np.stack((x_grid, y_grid), axis=2).astype(np.float32)
    window.set_projector_grid(grid)
    window.set_turn_indicator("bottom-left")
    window.set_edge_clocks(
        "05:14",
        "02:37",
        board_quarter_turns=1,
        result="1/2-1/2",
    )
    window.show_black()
    window.show()
    application.processEvents()

    image = window.grab().toImage()
    result_pixels = sum(
        image.pixelColor(x, y).red() > 50
        for x in range(166, 195)
        for y in range(17, 29)
    )
    upper_fraction_points = [
        (x, y)
        for x in range(184, 193)
        for y in range(17, 22)
        if image.pixelColor(x, y).red() > 50
    ]
    lower_fraction_points = [
        (x, y)
        for x in range(184, 193)
        for y in range(24, 29)
        if image.pixelColor(x, y).red() > 50
    ]

    assert result_pixels > 30
    assert upper_fraction_points
    assert len(lower_fraction_points) > 3
    upper_center_x = sum(x for x, _y in upper_fraction_points) / len(
        upper_fraction_points
    )
    lower_center_x = sum(x for x, _y in lower_fraction_points) / len(
        lower_fraction_points
    )
    assert upper_center_x == pytest.approx(lower_center_x, abs=1.0)
    assert image.pixelColor(180, 90).red() == 51
    assert image.pixelColor(90, 158).red() == 0
    expiration_flag_pixels = sum(
        image.pixelColor(x, y).red() > 150
        for x in range(135, 166)
        for y in range(10, 41)
    )
    assert expiration_flag_pixels < 10


def test_promotion_projection_keeps_edge_clocks_visible(
    application: QApplication,
) -> None:
    from unittest.mock import patch

    window = ProjectorWindow()
    window.resize(360, 180)
    window.show()
    x_values = np.linspace(80, 280, 9, dtype=np.float32)
    y_values = np.asarray(
        (10, 20, 30, 40, 80, 120, 130, 140, 150),
        dtype=np.float32,
    )
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    window.set_projector_grid(
        np.stack((x_grid, y_grid), axis=2).astype(np.float32)
    )
    window.set_edge_clocks(
        "10:05",
        "10:05",
        board_quarter_turns=0,
    )

    with patch.object(
        ProjectorWindow, "_draw_edge_clock"
    ) as draw_edge_clock:
        window.show_promotion_choices("top", "bottom")
        application.processEvents()

    assert draw_edge_clock.call_count == 2
    window.close()


def test_projector_renders_patterns_and_calibrated_squares(
    application: QApplication,
) -> None:
    from unittest.mock import patch

    window = ProjectorWindow()
    window.resize(320, 180)
    window.show()
    pattern = np.zeros((180, 320), dtype=np.uint8)
    pattern[:, 160:] = 180
    window.show_pattern(pattern)
    application.processEvents()

    pattern_image = window.grab().toImage()
    assert pattern_image.pixelColor(80, 90).red() == 0
    assert pattern_image.pixelColor(240, 90).red() == 180

    color_pattern = np.zeros((180, 320, 3), dtype=np.uint8)
    color_pattern[:, :160, 1] = 200
    color_pattern[:, 160:, 0] = 200
    window.show_pattern(color_pattern)
    application.processEvents()
    color_image = window.grab().toImage()
    assert color_image.pixelColor(80, 90).green() == 200
    assert color_image.pixelColor(80, 90).red() == 0
    assert color_image.pixelColor(240, 90).red() == 200
    assert color_image.pixelColor(240, 90).green() == 0

    x_values = np.linspace(40, 280, 9, dtype=np.float32)
    y_values = np.linspace(20, 160, 9, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    grid = np.stack((x_grid, y_grid), axis=2).astype(np.float32)
    window.set_projector_grid(grid)
    window.set_turn_indicator("bottom-right")
    window.show_black()
    application.processEvents()
    turn_image = window.grab().toImage()
    assert turn_image.pixelColor(295, 169).red() > 150

    window.show_all_squares()
    application.processEvents()

    square_image = window.grab().toImage()
    center = square_image.pixelColor(55, 29)
    gap = square_image.pixelColor(70, 38)
    assert center.red() == 255
    assert gap.red() == 51

    window.show_square(2, 3)
    application.processEvents()
    selected_image = window.grab().toImage()
    selected_center = selected_image.pixelColor(145, 64)
    adjacent_center = selected_image.pixelColor(175, 64)
    assert selected_center == PROJECTED_ON_DARK_SQUARE
    assert adjacent_center.red() == 51

    window.show_square(2, 3, light_square=True)
    application.processEvents()
    logical_light_image = window.grab().toImage()
    assert (
        logical_light_image.pixelColor(145, 64)
        == PROJECTED_ON_LIGHT_SQUARE
    )

    window.show_square(2, 2)
    application.processEvents()
    light_square_image = window.grab().toImage()
    assert (
        light_square_image.pixelColor(115, 64)
        == PROJECTED_ON_LIGHT_SQUARE
    )

    window.show_move(
        (2, 2),
        (4, 5),
        source_is_light=True,
        destination_is_light=False,
    )
    application.processEvents()
    move_image = window.grab().toImage()
    assert move_image.pixelColor(115, 64) == PROJECTED_LEGAL_MOVE
    assert move_image.pixelColor(205, 99) == PROJECTED_LEGAL_MOVE
    assert move_image.pixelColor(145, 64).red() == 51
    assert move_image.pixelColor(295, 169).red() > 150

    window.show_black()
    application.processEvents()
    restored_turn_image = window.grab().toImage()
    assert restored_turn_image.pixelColor(295, 169).red() > 150

    window.show_move_squares(
        (
            (7, 4, False),
            (7, 6, False),
            (7, 7, True),
            (7, 5, True),
        )
    )
    application.processEvents()
    castling_image = window.grab().toImage()
    for x_coordinate in (175, 205, 235, 265):
        assert (
            castling_image.pixelColor(x_coordinate, 151)
            == PROJECTED_LEGAL_MOVE
        )

    window.show_illegal_square(2, 3)
    application.processEvents()
    illegal_image = window.grab().toImage()
    assert illegal_image.pixelColor(145, 64) == PROJECTED_ILLEGAL_MOVE
    assert illegal_image.pixelColor(175, 64).red() == 51
    assert illegal_image.pixelColor(295, 169).red() > 150

    window.show_correction((2, 3), (4, 5))
    application.processEvents()
    correction_image = window.grab().toImage()
    assert correction_image.pixelColor(145, 64) == PROJECTED_ILLEGAL_MOVE
    assert correction_image.pixelColor(205, 99) == PROJECTED_ILLEGAL_MOVE
    assert correction_image.pixelColor(175, 64).red() == 51
    assert correction_image.pixelColor(295, 169).red() > 150

    window.show_reconciliation(((2, 3), (4, 5)))
    application.processEvents()
    reconciliation_image = window.grab().toImage()
    assert reconciliation_image.pixelColor(145, 64) == PROJECTED_ILLEGAL_MOVE
    assert reconciliation_image.pixelColor(205, 99) == PROJECTED_ILLEGAL_MOVE
    assert reconciliation_image.pixelColor(175, 64).red() == 51
    assert reconciliation_image.pixelColor(295, 169).red() > 150

    window.show_promotion_choices("top", "bottom")
    application.processEvents()
    assert window.output_mode == "promotion_choices"
    promotion_choices_image = window.grab().toImage()
    assert promotion_choices_image.pixelColor(295, 169).red() > 150

    with patch.object(
        ProjectorWindow,
        "_draw_promotion_glyph",
    ) as draw_promotion_glyph:
        window.show_promotion_confirmation("top", "bottom", "n", (0, 0))
        application.processEvents()
        promotion_image = window.grab().toImage()
    assert window.output_mode == "promotion_confirmation"
    assert promotion_image.pixelColor(295, 169).red() > 150
    assert promotion_image.pixelColor(115, 11) == PROJECTED_LEGAL_MOVE
    rendered_pieces = [
        call.args[2] for call in draw_promotion_glyph.call_args_list
    ]
    assert rendered_pieces[-4:] == ["b", "r", "q", "n"]
    window.close()
