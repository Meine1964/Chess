import os

import pytest
from PySide6.QtCore import QPointF
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from src.chess_camera.chessboard_widget import (
    ChessboardWidget,
    LICHESS_DARK,
    LICHESS_LIGHT,
    rotate_square,
    square_at_position,
)


@pytest.fixture(scope="module")
def application() -> QApplication:
    existing = QApplication.instance()
    if isinstance(existing, QApplication):
        return existing
    return QApplication([])


def test_lichess_board_colors_match_classic_brown_theme() -> None:
    assert LICHESS_LIGHT.name() == "#f0d9b5"
    assert LICHESS_DARK.name() == "#b58863"


def test_rotates_logical_square_into_physical_orientation() -> None:
    assert rotate_square(1, 2, 0) == (1, 2)
    assert rotate_square(1, 2, 1) == (2, 6)
    assert rotate_square(1, 2, 2) == (6, 5)
    assert rotate_square(1, 2, 3) == (5, 1)


def test_square_at_position_maps_top_left_to_bottom_right() -> None:
    assert square_at_position(QPointF(5, 5), 800, 800) == (0, 0)
    assert square_at_position(QPointF(795, 5), 800, 800) == (0, 7)
    assert square_at_position(QPointF(5, 795), 800, 800) == (7, 0)
    assert square_at_position(QPointF(795, 795), 800, 800) == (7, 7)


def test_square_at_position_ignores_space_outside_centered_board() -> None:
    assert square_at_position(QPointF(10, 100), 1000, 800) is None
    assert square_at_position(QPointF(110, 100), 1000, 800) == (1, 0)


def test_widget_stores_validated_occupied_squares(
    application: QApplication,
) -> None:
    widget = ChessboardWidget()

    widget.set_occupied_squares({(0, 0), (3, 4), (7, 7)})

    assert widget.occupied_squares == frozenset({(0, 0), (3, 4), (7, 7)})
    with pytest.raises(ValueError, match="inside the 8x8 board"):
        widget.set_occupied_squares({(8, 0)})


def test_widget_renders_black_dot_at_occupied_square_center(
    application: QApplication,
) -> None:
    widget = ChessboardWidget()
    widget.resize(320, 320)
    widget.set_occupied_squares({(0, 0)})
    widget.show()
    application.processEvents()

    image = widget.grab().toImage()

    assert QColor(image.pixel(20, 20)).name() == "#111111"
    assert QColor(image.pixel(60, 20)).name() != "#111111"
    widget.close()
