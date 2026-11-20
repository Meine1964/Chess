import pytest

from src.chess_camera.chess_position import standard_position
from src.chess_camera.legal_game import LegalGame


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


def test_promotion_requires_explicit_choice() -> None:
    game = LegalGame("8/P7/8/8/8/8/8/k6K w - - 0 1")

    with pytest.raises(ValueError, match="Promotion choice is required"):
        game.apply_move((1, 0), (0, 0))

    promoted = game.apply_move((1, 0), (0, 0), "q")

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
