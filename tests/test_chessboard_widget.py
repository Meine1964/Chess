from PySide6.QtCore import QPointF

from src.chess_camera.chessboard_widget import (
    LICHESS_DARK,
    LICHESS_LIGHT,
    rotate_square,
    square_at_position,
)


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
