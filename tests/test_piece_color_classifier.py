import cv2
import numpy as np

from src.chess_camera.piece_color_classifier import (
    build_piece_color_reference,
    classify_piece_colors,
    classify_piece_colors_from_reference,
)


def _empty_board() -> np.ndarray:
    board = np.zeros((800, 800, 3), dtype=np.uint8)
    for row in range(8):
        for column in range(8):
            color = 210 if (row + column) % 2 == 0 else 40
            board[
                row * 100 : (row + 1) * 100,
                column * 100 : (column + 1) * 100,
            ] = color
    return board


def _draw_piece(
    image: np.ndarray,
    square: tuple[int, int],
    color: int,
) -> None:
    row, column = square
    cv2.circle(
        image,
        (column * 100 + 50, row * 100 + 50),
        32,
        (color, color, color),
        -1,
    )


def test_classifies_empty_white_and_black_from_reference_samples() -> None:
    reference = _empty_board()
    _draw_piece(reference, (6, 0), 225)
    _draw_piece(reference, (1, 0), 20)
    candidate = reference.copy()

    classifications = classify_piece_colors(
        reference,
        candidate,
        {(6, 0): "P", (1, 0): "p"},
    )

    assert classifications[(4, 4)].label == "empty"
    assert classifications[(6, 0)].label == "white"
    assert classifications[(1, 0)].label == "black"


def test_classifies_capture_source_empty_and_destination_mover_color() -> None:
    reference = _empty_board()
    _draw_piece(reference, (5, 0), 225)
    _draw_piece(reference, (4, 1), 20)
    candidate = _empty_board()
    _draw_piece(candidate, (4, 1), 225)

    classifications = classify_piece_colors(
        reference,
        candidate,
        {(5, 0): "P", (4, 1): "n"},
    )

    assert classifications[(5, 0)].label == "empty"
    assert classifications[(4, 1)].label == "white"


def test_uncertain_color_does_not_force_a_piece_label() -> None:
    reference = _empty_board()
    _draw_piece(reference, (6, 0), 225)
    _draw_piece(reference, (1, 0), 20)
    candidate = reference.copy()
    _draw_piece(candidate, (4, 4), 125)

    classifications = classify_piece_colors(
        reference,
        candidate,
        {(6, 0): "P", (1, 0): "p"},
    )

    assert classifications[(4, 4)].label == "uncertain"


def test_cached_reference_classification_matches_fresh_model() -> None:
    reference = _empty_board()
    _draw_piece(reference, (6, 0), 225)
    _draw_piece(reference, (1, 0), 20)
    candidate = _empty_board()
    _draw_piece(candidate, (4, 1), 225)
    position = {(6, 0): "P", (1, 0): "p"}

    fresh = classify_piece_colors(reference, candidate, position)
    model = build_piece_color_reference(reference, position)
    cached = classify_piece_colors_from_reference(model, candidate)

    assert cached == fresh
