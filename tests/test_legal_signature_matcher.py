import numpy as np

from src.chess_camera.legal_game import LegalGame
from src.chess_camera.legal_signature_matcher import (
    match_legal_move_signature,
)
from src.chess_camera.piece_color_classifier import PieceColorClassification


def _scores(*changes: tuple[tuple[int, int], float]) -> np.ndarray:
    scores = np.zeros((8, 8), dtype=np.float32)
    for square, score in changes:
        scores[square] = score
    return scores


def test_matches_unique_ordinary_legal_signature() -> None:
    game = LegalGame()

    match = match_legal_move_signature(
        _scores(((6, 4), 0.21), ((4, 4), 0.19)),
        game.legal_move_signatures,
    )

    assert match.signature is not None
    assert match.signature.uci == "e2e4"
    assert match.signature.kind == "ordinary"


def test_unexplained_third_square_rejects_signature() -> None:
    game = LegalGame()
    scores = _scores(
        ((6, 4), 0.21),
        ((4, 4), 0.19),
        ((7, 3), 0.18),
    )

    rejected = match_legal_move_signature(
        scores,
        game.legal_move_signatures,
    )
    explained = match_legal_move_signature(
        scores,
        game.legal_move_signatures,
        explained_residuals={"e2e4": ((7, 3),)},
    )

    assert rejected.signature is None
    assert explained.signature is not None
    assert explained.signature.uci == "e2e4"


def test_matches_castling_four_square_signature() -> None:
    game = LegalGame("4k3/8/8/8/8/8/8/4K2R w K - 0 1")

    match = match_legal_move_signature(
        _scores(
            ((7, 4), 0.2),
            ((7, 6), 0.2),
            ((7, 7), 0.18),
            ((7, 5), 0.18),
        ),
        game.legal_move_signatures,
    )

    assert match.signature is not None
    assert match.signature.uci == "e1g1"
    assert match.signature.kind == "castling"


def test_capture_requires_vacated_source_and_mover_color_destination() -> None:
    game = LegalGame("4k3/8/8/4p3/8/5N2/8/4K3 w - - 0 1")
    scores = _scores(((5, 5), 0.2), ((3, 4), 0.2))
    valid_colors = {
        (5, 5): PieceColorClassification("empty", 0.01, 0.2),
        (3, 4): PieceColorClassification("white", 0.02, 0.18),
    }
    uncertain_destination = {
        **valid_colors,
        (3, 4): PieceColorClassification("uncertain", 0.12, 0.01),
    }

    matched = match_legal_move_signature(
        scores,
        game.legal_move_signatures,
        piece_colors=valid_colors,
    )
    rejected = match_legal_move_signature(
        scores,
        game.legal_move_signatures,
        piece_colors=uncertain_destination,
    )

    assert matched.signature is not None
    assert matched.signature.uci == "f3e5"
    assert rejected.signature is None


def test_queen_is_the_only_generated_promotion_signature() -> None:
    game = LegalGame("8/P7/8/8/8/8/8/k6K w - - 0 1")

    match = match_legal_move_signature(
        _scores(((1, 0), 0.2), ((0, 0), 0.2)),
        game.legal_move_signatures,
    )

    assert match.signature is not None
    assert match.signature.uci == "a7a8q"
    assert match.signature.promotion == "q"
