import os

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from src.chess_camera.projector import (
    PROJECTED_ON_DARK_SQUARE,
    PROJECTED_ILLEGAL_MOVE,
    PROJECTED_ON_LIGHT_SQUARE,
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

    window.show_reconciliation(((2, 3), (4, 5)))
    application.processEvents()
    reconciliation_image = window.grab().toImage()
    assert reconciliation_image.pixelColor(145, 64) == PROJECTED_ILLEGAL_MOVE
    assert reconciliation_image.pixelColor(205, 99) == PROJECTED_ILLEGAL_MOVE
    assert reconciliation_image.pixelColor(175, 64).red() == 0
    window.close()
