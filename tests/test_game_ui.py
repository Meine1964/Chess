from src.chess_camera.camera_app import (
    _accepted_position_message,
    _format_lichess_clock,
    _illegal_correction_message,
    _lichess_board_state,
    _lichess_game_state_message,
    _lichess_polling_snapshot_message,
    _lichess_turn_indicator_state,
    _reference_tracking_status_message,
    _turn_indicator_state,
)
from src.chess_camera.legal_game import LegalGame
from src.chess_camera.lichess_client import (
    LichessGameFull,
    LichessGameState,
    LichessPlayer,
    LichessPollingSnapshot,
)
from src.chess_camera.reference_move_tracker import (
    ReferenceTrackerState,
    ReferenceTrackerUpdate,
)


def test_turn_indicator_is_hidden_in_diagnostic_mode() -> None:
    assert _turn_indicator_state("diagnostic", LegalGame()) is None


def test_lichess_game_state_message_reports_clocks_and_result() -> None:
    state = LichessGameState(
        (),
        "unused",
        "black",
        65000,
        9000,
        0,
        0,
        "started",
        None,
        "*",
    )

    assert _format_lichess_clock(-1) == "0:00"
    assert _lichess_game_state_message(state) == (
        "Lichess: Black to move; White 1:05, Black 0:09; "
        "status started; result *."
    )

    assert _lichess_turn_indicator_state(state) == (
        "#000000",
        "Black to move (Lichess)",
    )


def test_lichess_turn_indicator_reports_completed_game() -> None:
    state = LichessGameState(
        (),
        "unused",
        "white",
        0,
        0,
        0,
        0,
        "mate",
        "black",
        "0-1",
    )

    assert _lichess_turn_indicator_state(state) == (
        "#808080",
        "Lichess game over: 0-1",
    )


def test_polling_message_reports_unavailable_correspondence_clocks() -> None:
    state = LichessGameState(
        (),
        "unused",
        "white",
        0,
        0,
        0,
        0,
        "started",
        None,
        "*",
    )
    snapshot = LichessPollingSnapshot(
        LichessGameFull(
            "oozTO3xk",
            "unused",
            LichessPlayer("white", "Meine", 1500),
            LichessPlayer("black", "Stockfish level 1", None),
            state,
        ),
        False,
    )

    assert _lichess_polling_snapshot_message(snapshot) == (
        "Lichess game oozTO3xk: Meine (White) vs "
        "Stockfish level 1 (Black). White to move; "
        "clocks unavailable for this game; status started; result *. "
        "Connected by read-only polling."
    )


def test_lichess_board_state_uses_remote_fen_and_last_move() -> None:
    game = LegalGame()
    result = game.apply_move((6, 4), (4, 4))
    state = LichessGameState(
        ("e2e4",),
        result.fen,
        "black",
        0,
        0,
        0,
        0,
        "started",
        None,
        "*",
    )

    position, last_move = _lichess_board_state(state)

    assert (6, 4) not in position
    assert position[(4, 4)] == "P"
    assert last_move == ((6, 4), (4, 4))


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
    assert _illegal_correction_message(
        ((4, 1), (4, 1)),
        game,
    ) == (
        "Illegal physical removal from b4 rejected. The accepted position "
        "requires the piece on b4. White to move. Return the piece to b4, "
        "then make a White move."
    )


def test_accepted_position_message_includes_authoritative_turn() -> None:
    game = LegalGame()

    assert _accepted_position_message(game) == (
        "Board matches the accepted position. White to move."
    )


def test_transient_tracking_phases_do_not_replace_visible_guidance() -> None:
    states: tuple[ReferenceTrackerState, ...] = (
        "activity",
        "settling",
        "candidate",
    )
    for state in states:
        assert (
            _reference_tracking_status_message(
                ReferenceTrackerUpdate(
                    state,
                    None,
                    "Internal phase detail.",
                )
            )
            is None
        )


def test_tracking_status_uses_stable_actionable_guidance() -> None:
    assert _reference_tracking_status_message(
        ReferenceTrackerUpdate(
            "waiting-for-clear",
            None,
            "Technical obstruction detail.",
        )
    ) == "Move your hand and other objects away from the board."
    assert _reference_tracking_status_message(
        ReferenceTrackerUpdate(
            "ambiguous",
            None,
            "Technical square scores.",
        )
    ) == (
        "Board change is unclear. Restore the accepted position, then move "
        "one piece at a time."
    )
    assert _reference_tracking_status_message(
        ReferenceTrackerUpdate(
            "pending-capture",
            None,
            "Captured piece removed from b4; complete the capture from a3.",
        )
    ) == "Captured piece removed from b4; complete the capture from a3."
