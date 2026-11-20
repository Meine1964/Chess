from src.chess_camera.camera_app import (
    _accepted_position_message,
    _illegal_correction_message,
    _turn_indicator_state,
)
from src.chess_camera.legal_game import LegalGame


def test_turn_indicator_is_hidden_in_diagnostic_mode() -> None:
    assert _turn_indicator_state("diagnostic", LegalGame()) is None


def test_turn_indicator_tracks_authoritative_side_to_move() -> None:
    game = LegalGame()

    assert _turn_indicator_state("offline-game", game) == (
        "#ffffff",
        "White to move",
    )

    game.apply_move((6, 4), (4, 4))

    assert _turn_indicator_state("offline-game", game) == (
        "#000000",
        "Black to move",
    )


def test_turn_indicator_reports_completed_game() -> None:
    game = LegalGame()
    for source, destination in (
        ((6, 5), (5, 5)),
        ((1, 4), (3, 4)),
        ((6, 6), (4, 6)),
        ((0, 3), (4, 7)),
    ):
        game.apply_move(source, destination)

    assert _turn_indicator_state("offline-game", game) == (
        "#808080",
        "Game over: 0-1",
    )


def test_illegal_correction_names_squares_legal_choices_and_turn() -> None:
    game = LegalGame()

    assert _illegal_correction_message(
        ((6, 4), (3, 4)),
        game,
    ) == (
        "Illegal physical move e2-e5 rejected. The physical piece is on e5; "
        "the accepted position requires it on e2. White to move. Move it "
        "from e5 back to e2, or directly to a legal destination: e4, e3."
    )
    assert _illegal_correction_message(
        ((1, 3), (3, 3)),
        game,
    ) == (
        "Illegal physical move d7-d5 rejected. The physical piece is on d5; "
        "the accepted position requires it on d7. White to move. This piece "
        "has no legal move now. Return it from d5 to d7, then make a White "
        "move."
    )


def test_accepted_position_message_includes_authoritative_turn() -> None:
    game = LegalGame()

    assert _accepted_position_message(game) == (
        "Board matches the accepted position. White to move."
    )
