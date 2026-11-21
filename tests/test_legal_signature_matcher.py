import numpy as np

from src.chess_camera.legal_game import LegalGame
from src.chess_camera.legal_signature_matcher import (
    find_illegal_removed_piece,
    match_legal_move_signature,
    match_pending_castling,
    match_pending_castling_relocation,
    match_pending_capture,
    match_pending_en_passant,
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


def test_matches_legal_move_with_weaker_vacated_source_evidence() -> None:
    game = LegalGame()

    match = match_legal_move_signature(
        _scores(
            ((6, 4), 0.067),
            ((4, 4), 0.105),
            ((4, 0), 0.035),
        ),
        game.legal_move_signatures,
    )

    assert match.signature is not None
    assert match.signature.uci == "e2e4"


def test_color_confirmed_quiet_move_uses_lower_destination_floor() -> None:
    game = LegalGame(
        "rnbqkb1r/pppppppp/8/8/3Pn3/8/PPP2PPP/"
        "RNBQKBNR w KQkq - 0 3"
    )
    scores = _scores(((6, 2), 0.072), ((4, 2), 0.069))
    colors = {
        (6, 2): PieceColorClassification("empty", 0.02, 0.16),
        (4, 2): PieceColorClassification("white", 0.04, 0.11),
    }

    without_colors = match_legal_move_signature(
        scores,
        game.physical_move_signatures,
    )
    with_colors = match_legal_move_signature(
        scores,
        game.physical_move_signatures,
        piece_colors=colors,
    )
    wrong_destination = match_legal_move_signature(
        scores,
        game.physical_move_signatures,
        piece_colors={
            **colors,
            (4, 2): PieceColorClassification("black", 0.04, 0.11),
        },
    )

    assert without_colors.signature is None
    assert with_colors.signature is not None
    assert with_colors.signature.uci == "c2c4"
    assert wrong_destination.signature is None


def test_rejects_legal_move_when_source_change_is_only_noise() -> None:
    game = LegalGame()

    match = match_legal_move_signature(
        _scores(((6, 4), 0.055), ((4, 4), 0.105)),
        game.legal_move_signatures,
    )

    assert match.signature is None


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


def test_pending_castling_matches_king_first() -> None:
    game = LegalGame("4k3/8/8/8/8/8/8/4K2R w K - 0 1")
    colors = {
        (7, 4): PieceColorClassification("empty", 0.01, 0.2),
        (7, 6): PieceColorClassification("white", 0.01, 0.2),
        (7, 7): PieceColorClassification("white", 0.01, 0.2),
        (7, 5): PieceColorClassification("empty", 0.01, 0.2),
    }

    pending = match_pending_castling(
        _scores(((7, 4), 0.2), ((7, 6), 0.2)),
        game.physical_move_signatures,
        colors,
    )

    assert pending.signature is not None
    assert pending.signature.uci == "e1g1"
    assert pending.rook_source == (7, 7)
    assert pending.rook_destination == (7, 5)


def test_legal_castling_king_relocation_remains_pending() -> None:
    game = LegalGame("4k3/8/8/8/8/8/8/4K2R w K - 0 1")

    pending = match_pending_castling_relocation(
        ((7, 4), (7, 6)),
        game.physical_move_signatures,
    )

    assert pending.signature is not None
    assert pending.signature.uci == "e1g1"
    assert pending.rook_source == (7, 7)
    assert pending.rook_destination == (7, 5)


def test_matches_complete_low_contrast_en_passant_signature() -> None:
    game = LegalGame(
        "rnbqkbnr/1pp1pppp/p7/3pP3/8/8/PPPP1PPP/"
        "RNBQKBNR w KQkq d6 0 3"
    )
    colors = {
        (3, 4): PieceColorClassification("empty", 0.01, 0.2),
        (2, 3): PieceColorClassification("white", 0.01, 0.2),
        (3, 3): PieceColorClassification("empty", 0.01, 0.2),
    }

    match = match_legal_move_signature(
        _scores(
            ((3, 4), 0.065),
            ((2, 3), 0.067),
            ((3, 3), 0.064),
        ),
        game.physical_move_signatures,
        piece_colors=colors,
    )

    assert match.signature is not None
    assert match.signature.uci == "e5d6"
    assert match.signature.kind == "en-passant"


def test_pending_en_passant_matches_captured_pawn_first() -> None:
    game = LegalGame(
        "rnbqkbnr/1pp1pppp/p7/3pP3/8/8/PPPP1PPP/"
        "RNBQKBNR w KQkq d6 0 3"
    )
    colors = {
        (3, 4): PieceColorClassification("white", 0.01, 0.2),
        (2, 3): PieceColorClassification("empty", 0.01, 0.2),
        (3, 3): PieceColorClassification("empty", 0.01, 0.2),
    }

    pending = match_pending_en_passant(
        _scores(((3, 3), 0.065)),
        game.physical_move_signatures,
        colors,
    )

    assert pending.captured_square == (3, 3)
    assert pending.destination == (2, 3)
    assert pending.sources == ((3, 4),)
    assert tuple(signature.uci for signature in pending.signatures) == (
        "e5d6",
    )


def test_matches_low_contrast_auto_queen_promotion() -> None:
    game = LegalGame("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    colors = {
        (1, 0): PieceColorClassification("empty", 0.01, 0.2),
        (0, 0): PieceColorClassification("white", 0.01, 0.2),
    }

    match = match_legal_move_signature(
        _scores(((1, 0), 0.065), ((0, 0), 0.064)),
        game.physical_move_signatures,
        piece_colors=colors,
    )

    assert match.signature is not None
    assert match.signature.uci == "a7a8q"
    assert match.signature.kind == "promotion"


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


def test_pending_capture_matches_removed_destination_with_attacker_retained() -> None:
    game = LegalGame("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1")
    colors = {
        (4, 4): PieceColorClassification("white", 0.01, 0.2),
        (3, 3): PieceColorClassification("empty", 0.01, 0.2),
    }

    pending = match_pending_capture(
        _scores(((3, 3), 0.2)),
        game.physical_move_signatures,
        colors,
    )

    assert pending.destination == (3, 3)
    assert pending.sources == ((4, 4),)
    assert tuple(signature.uci for signature in pending.signatures) == (
        "e4d5",
    )


def test_uncapturable_opponent_removal_is_illegal() -> None:
    game = LegalGame("4k3/8/8/4P3/1n6/8/8/4K3 w - - 0 1")
    removed = (4, 1)
    colors = {
        removed: PieceColorClassification("empty", 0.01, 0.2),
    }

    illegal = find_illegal_removed_piece(
        _scores((removed, 0.2)),
        game.physical_move_signatures,
        colors,
        game.position,
    )

    assert illegal == removed


def test_capturable_opponent_removal_is_not_illegal() -> None:
    game = LegalGame("4k3/8/8/8/1n6/P7/8/4K3 w - - 0 1")
    removed = (4, 1)
    colors = {
        removed: PieceColorClassification("empty", 0.01, 0.2),
    }

    illegal = find_illegal_removed_piece(
        _scores((removed, 0.2)),
        game.physical_move_signatures,
        colors,
        game.position,
    )

    assert illegal is None


def test_side_to_move_piece_removed_without_destination_is_illegal() -> None:
    game = LegalGame(
        "rnbqkbnr/pppp1ppp/8/4p3/1P6/8/P1PPPPPP/"
        "RNBQKBNR w KQkq - 0 2"
    )
    removed = (4, 1)
    colors = {
        removed: PieceColorClassification("empty", 0.01, 0.2),
    }

    illegal = find_illegal_removed_piece(
        _scores((removed, 0.2)),
        game.physical_move_signatures,
        colors,
        game.position,
    )

    assert illegal == removed


def test_queen_is_the_only_generated_promotion_signature() -> None:
    game = LegalGame("8/P7/8/8/8/8/8/k6K w - - 0 1")

    match = match_legal_move_signature(
        _scores(((1, 0), 0.2), ((0, 0), 0.2)),
        game.legal_move_signatures,
    )

    assert match.signature is not None
    assert match.signature.uci == "a7a8q"
    assert match.signature.promotion == "q"
