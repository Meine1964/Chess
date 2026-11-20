import os

import pytest
from PySide6.QtCore import QPointF
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from src.chess_camera.chessboard_widget import (
    ChessboardWidget,
    LICHESS_DARK,
    LICHESS_ILLEGAL,
    LICHESS_LIGHT,
    rotate_square,
    square_at_position,
)
from src.chess_camera.chess_position import standard_position


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


def test_widget_renders_standard_cburnett_position(
    application: QApplication,
) -> None:
    widget = ChessboardWidget()
    widget.resize(320, 320)
    widget.set_occupied_squares({(6, 0)})
    widget.set_pieces(standard_position())
    widget.show()
    application.processEvents()

    image = widget.grab().toImage()

    assert len(widget.pieces) == 32
    assert QColor(image.pixel(20, 260)).name() != "#111111"
    widget.close()


def test_widget_persists_last_move_source_and_destination_highlight(
    application: QApplication,
) -> None:
    widget = ChessboardWidget()
    widget.resize(320, 320)
    widget.set_last_move(((4, 4), (5, 4)))
    widget.show()
    application.processEvents()

    image = widget.grab().toImage()

    assert widget.last_move == ((4, 4), (5, 4))
    assert QColor(image.pixel(180, 180)).name() == LICHESS_LIGHT.lighter(
        135
    ).name()
    assert QColor(image.pixel(180, 220)).name() == LICHESS_DARK.lighter(
        135
    ).name()
    assert QColor(image.pixel(220, 180)).name() == LICHESS_DARK.name()

    widget.set_last_move(((3, 0), (2, 0)))
    assert widget.last_move == ((3, 0), (2, 0))
    widget.set_last_move(None)
    assert widget.last_move is None
    with pytest.raises(ValueError, match="inside the 8x8 board"):
        widget.set_last_move(((4, 4), (8, 4)))
    widget.close()


def test_widget_marks_illegal_source_without_moving_piece(
    application: QApplication,
) -> None:
    widget = ChessboardWidget()
    widget.resize(320, 320)
    position = standard_position()
    widget.set_pieces(position)
    widget.set_illegal_source((7, 6))
    widget.show()
    application.processEvents()

    image = widget.grab().toImage()
    marked_square_colors = {
        QColor(image.pixel(x, y)).name()
        for x in range(240, 280)
        for y in range(280, 320)
    }

    assert widget.illegal_source == (7, 6)
    assert widget.pieces == position
    assert LICHESS_ILLEGAL.name() in marked_square_colors
    widget.set_illegal_source(None)
    assert widget.illegal_source is None
    with pytest.raises(ValueError, match="inside the 8x8 board"):
        widget.set_illegal_source((8, 0))
    widget.close()
