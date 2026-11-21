import os

import numpy as np
import pytest
from PySide6.QtCore import QPointF, Qt
from PySide6.QtWidgets import QApplication

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from src.chess_camera.projector import (
    PROJECTED_LEGAL_MOVE,
    PROJECTED_ON_DARK_SQUARE,
    PROJECTED_ILLEGAL_MOVE,
    PROJECTED_ON_LIGHT_SQUARE,
    PROMOTION_GLYPHS,
    ProjectorWindow,
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


def test_projector_window_stays_above_screen_savers(
    application: QApplication,
) -> None:
    window = ProjectorWindow()

    assert (
        window.windowFlags() & Qt.WindowType.WindowStaysOnTopHint
        == Qt.WindowType.WindowStaysOnTopHint
    )


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

    x_values = np.linspace(40, 280, 9, dtype=np.float32)
    y_values = np.linspace(20, 160, 9, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    grid = np.stack((x_grid, y_grid), axis=2).astype(np.float32)
    window.set_projector_grid(grid)
    window.show_all_squares()
    application.processEvents()

    square_image = window.grab().toImage()
    center = square_image.pixelColor(55, 29)
    gap = square_image.pixelColor(70, 38)
    assert center.red() == 255
    assert gap.red() == 0

    window.show_square(2, 3)
    application.processEvents()
    selected_image = window.grab().toImage()
    selected_center = selected_image.pixelColor(145, 64)
    adjacent_center = selected_image.pixelColor(175, 64)
    assert selected_center == PROJECTED_ON_DARK_SQUARE
    assert adjacent_center.red() == 0

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
    assert move_image.pixelColor(145, 64).red() == 0

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
    assert illegal_image.pixelColor(175, 64).red() == 0

    window.show_correction((2, 3), (4, 5))
    application.processEvents()
    correction_image = window.grab().toImage()
    assert correction_image.pixelColor(145, 64) == PROJECTED_ILLEGAL_MOVE
    assert correction_image.pixelColor(205, 99) == PROJECTED_ILLEGAL_MOVE
    assert correction_image.pixelColor(175, 64).red() == 0

    window.show_reconciliation(((2, 3), (4, 5)))
    application.processEvents()
    reconciliation_image = window.grab().toImage()
    assert reconciliation_image.pixelColor(145, 64) == PROJECTED_ILLEGAL_MOVE
    assert reconciliation_image.pixelColor(205, 99) == PROJECTED_ILLEGAL_MOVE
    assert reconciliation_image.pixelColor(175, 64).red() == 0

    window.show_promotion_choices("top", "bottom")
    application.processEvents()
    assert window.output_mode == "promotion_choices"

    with patch.object(
        ProjectorWindow,
        "_draw_promotion_glyph",
    ) as draw_promotion_glyph:
        window.show_promotion_confirmation("top", "bottom", "n", (0, 0))
        application.processEvents()
        promotion_image = window.grab().toImage()
    assert window.output_mode == "promotion_confirmation"
    assert promotion_image.pixelColor(115, 11) == PROJECTED_LEGAL_MOVE
    rendered_pieces = [
        call.args[2] for call in draw_promotion_glyph.call_args_list
    ]
    assert rendered_pieces[-4:] == ["b", "r", "q", "n"]
    window.close()
