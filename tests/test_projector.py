import os

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from src.chess_camera.projector import (
    PROJECTED_ON_DARK_SQUARE,
    PROJECTED_ILLEGAL_MOVE,
    PROJECTED_ON_LIGHT_SQUARE,
    ProjectorWindow,
    focus_pattern,
    target_rectangle,
)


@pytest.fixture(scope="module")
def application() -> QApplication:
    existing = QApplication.instance()
    if isinstance(existing, QApplication):
        return existing
    return QApplication([])


def test_target_rectangle_is_centered_and_square() -> None:
    rectangle = target_rectangle(1280, 720, 50, 50, 10)

    assert rectangle.width() == 72
    assert rectangle.height() == 72
    assert rectangle.center().x() in (639, 640)
    assert rectangle.center().y() in (359, 360)


def test_target_rectangle_stays_inside_projector_edges() -> None:
    top_left = target_rectangle(1280, 720, 0, 0, 20)
    bottom_right = target_rectangle(1280, 720, 100, 100, 20)

    assert top_left.left() == 0
    assert top_left.top() == 0
    assert bottom_right.right() == 1279
    assert bottom_right.bottom() == 719


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


@pytest.mark.parametrize(
    ("width", "height", "x", "y", "size"),
    [
        (0, 720, 50, 50, 10),
        (1280, 0, 50, 50, 10),
        (1280, 720, -1, 50, 10),
        (1280, 720, 50, 101, 10),
        (1280, 720, 50, 50, 0),
    ],
)
def test_target_rectangle_rejects_invalid_values(
    width: int,
    height: int,
    x: int,
    y: int,
    size: int,
) -> None:
    with pytest.raises(ValueError):
        target_rectangle(width, height, x, y, size)


def test_projector_renders_patterns_and_calibrated_squares(
    application: QApplication,
) -> None:
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
    assert move_image.pixelColor(115, 64) == PROJECTED_ON_LIGHT_SQUARE
    assert move_image.pixelColor(205, 99) == PROJECTED_ON_DARK_SQUARE
    assert move_image.pixelColor(145, 64).red() == 0

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
    window.close()
