import pytest

from src.chess_camera.chess_position import standard_position
from src.chess_camera.legal_game import (
    LegalGame,
    LegalMoveSignature,
    PhysicalSquareChange,
)


def _signature(game: LegalGame, uci: str) -> LegalMoveSignature:
    return next(
        signature
        for signature in game.legal_move_signatures
        if signature.uci == uci
    )


def test_starting_game_exposes_position_and_legal_moves() -> None:
    game = LegalGame()

    assert game.position == standard_position()
    assert game.status.turn == "white"
    assert len(game.legal_uci_moves) == 20
    assert "e2e4" in game.legal_uci_moves


def test_legal_destinations_are_scoped_to_source_and_turn() -> None:
    game = LegalGame()

    assert game.legal_destinations((6, 4)) == ((4, 4), (5, 4))
    assert game.legal_destinations((1, 3)) == ()


def test_ordinary_move_signature_describes_source_and_destination() -> None:
    game = LegalGame()
    original_fen = game.fen

    signature = _signature(game, "e2e4")

    assert signature.san == "e4"
    assert signature.kind == "ordinary"
    assert signature.source == (6, 4)
    assert signature.destination == (4, 4)
    assert signature.promotion is None
    assert signature.changes == (
        PhysicalSquareChange((6, 4), "P", None),
        PhysicalSquareChange((4, 4), None, "P"),
    )
    assert game.fen == original_fen


def test_signature_cache_is_invalidated_after_move() -> None:
    game = LegalGame()
    assert any(
        signature.uci == "e2e4"
        for signature in game.legal_move_signatures
    )

    game.apply_move((6, 4), (4, 4))

    assert all(
        signature.uci != "e2e4"
        for signature in game.legal_move_signatures
    )
    assert any(
        signature.uci == "e7e5"
        for signature in game.legal_move_signatures
    )
    assert all(
        signature.uci != "e2e4"
        for signature in game.physical_move_signatures
    )


def test_physical_signatures_include_move_that_does_not_resolve_check() -> None:
    game = LegalGame("4k3/3P4/8/8/8/8/4p3/5N1K b - - 0 1")

    assert game.status.is_check
    assert all(
        signature.uci != "e2f1q"
        for signature in game.legal_move_signatures
    )
    signature = next(
        signature
        for signature in game.physical_move_signatures
        if signature.uci == "e2f1q"
    )

    assert signature.kind == "capture-promotion"
    assert signature.changes == (
        PhysicalSquareChange((6, 4), "p", None),
        PhysicalSquareChange((7, 5), "N", "q"),
    )
    with pytest.raises(ValueError, match="e2-f1 is not legal for black"):
        game.apply_move((6, 4), (7, 5))


def test_capture_signature_describes_destination_color_change() -> None:
    game = LegalGame("4k3/8/8/4p3/8/5N2/8/4K3 w - - 0 1")

    signature = _signature(game, "f3e5")

    assert signature.kind == "capture"
    assert signature.changes == (
        PhysicalSquareChange((5, 5), "N", None),
        PhysicalSquareChange((3, 4), "p", "N"),
    )


def test_castling_signature_includes_rook_transitions() -> None:
    game = LegalGame("4k3/8/8/8/8/8/8/4K2R w K - 0 1")

    signature = _signature(game, "e1g1")

    assert signature.kind == "castling"
    assert signature.changes == (
        PhysicalSquareChange((7, 4), "K", None),
        PhysicalSquareChange((7, 6), None, "K"),
        PhysicalSquareChange((7, 5), None, "R"),
        PhysicalSquareChange((7, 7), "R", None),
    )


def test_en_passant_signature_includes_captured_pawn_square() -> None:
    game = LegalGame()
    for source, destination in (
        ((6, 4), (4, 4)),
        ((1, 0), (2, 0)),
        ((4, 4), (3, 4)),
        ((1, 3), (3, 3)),
    ):
        game.apply_move(source, destination)

    signature = _signature(game, "e5d6")

    assert signature.kind == "en-passant"
    assert signature.changes == (
        PhysicalSquareChange((3, 4), "P", None),
        PhysicalSquareChange((2, 3), None, "P"),
        PhysicalSquareChange((3, 3), "p", None),
    )


def test_physical_signatures_assume_queen_promotion() -> None:
    game = LegalGame("8/P7/8/8/8/8/8/k6K w - - 0 1")

    signatures = {
        signature.uci: signature
        for signature in game.legal_move_signatures
        if signature.source == (1, 0)
    }

    assert set(signatures) == {"a7a8q"}
    queen = signatures["a7a8q"]
    assert queen.kind == "promotion"
    assert queen.promotion == "q"
    assert queen.changes == (
        PhysicalSquareChange((1, 0), "P", None),
        PhysicalSquareChange((0, 0), None, "Q"),
    )


def test_capture_promotion_signature_replaces_destination_piece() -> None:
    game = LegalGame("1r5k/P7/8/8/8/8/8/7K w - - 0 1")

    signature = _signature(game, "a7b8q")

    assert signature.kind == "capture-promotion"
    assert signature.promotion == "q"
    assert signature.changes == (
        PhysicalSquareChange((1, 0), "P", None),
        PhysicalSquareChange((0, 1), "r", "Q"),
    )


def test_apply_and_rollback_legal_move() -> None:
    game = LegalGame()
    starting_fen = game.fen

    applied = game.apply_move((6, 4), (4, 4))

    assert applied.uci == "e2e4"
    assert applied.san == "e4"
    assert applied.status.turn == "black"
    assert applied.position[(4, 4)] == "P"

    rolled_back = game.rollback()

    assert rolled_back.uci == "e2e4"
    assert rolled_back.fen == starting_fen
    assert rolled_back.position == standard_position()


def test_illegal_move_preserves_position() -> None:
    game = LegalGame()
    starting_fen = game.fen

    with pytest.raises(ValueError, match="e2-e5 is not legal for white"):
        game.apply_move((6, 4), (3, 4))

    assert game.fen == starting_fen
    assert game.position == standard_position()


def test_castling_and_en_passant_are_legal_moves() -> None:
    castling = LegalGame(
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"
    )

    assert "e1g1" in castling.legal_uci_moves
    castled = castling.apply_move((7, 4), (7, 6))
    assert castled.position[(7, 6)] == "K"
    assert castled.position[(7, 5)] == "R"

    en_passant = LegalGame()
    for source, destination in (
        ((6, 4), (4, 4)),
        ((1, 0), (2, 0)),
        ((4, 4), (3, 4)),
        ((1, 3), (3, 3)),
    ):
        en_passant.apply_move(source, destination)

    assert "e5d6" in en_passant.legal_uci_moves
    captured = en_passant.apply_move((3, 4), (2, 3))
    assert captured.position[(2, 3)] == "P"
    assert (3, 3) not in captured.position


def test_promotion_defaults_to_queen() -> None:
    game = LegalGame("8/P7/8/8/8/8/8/k6K w - - 0 1")

    promoted = game.apply_move((1, 0), (0, 0))

    assert promoted.uci == "a7a8q"
    assert promoted.position[(0, 0)] == "Q"


def test_checkmate_status_is_reported() -> None:
    game = LegalGame()
    for source, destination in (
        ((6, 5), (5, 5)),
        ((1, 4), (3, 4)),
        ((6, 6), (4, 6)),
        ((0, 3), (4, 7)),
    ):
        result = game.apply_move(source, destination)

    assert result.san == "Qh4#"
    assert result.status.is_check
    assert result.status.is_checkmate
    assert result.status.is_game_over
    assert result.status.result == "0-1"
