import pytest

from src.chess_camera.camera_app import (
    CameraWindow,
    _accepted_position_message,
    _can_start_new_lichess_game,
    _format_lichess_clock,
    _illegal_correction_message,
    _lichess_board_state,
    _lichess_game_information,
    _lichess_game_state_message,
    _lichess_polling_snapshot_message,
    _lichess_turn_indicator_state,
    _lichess_token_configuration_message,
    _ongoing_game_label,
    _reference_tracking_status_message,
    _turn_indicator_state,
)
from src.chess_camera.legal_game import LegalGame
from src.chess_camera.lichess_client import (
    LichessGameFull,
    LichessGameState,
    LichessPlayer,
    LichessPollingSnapshot,
    OngoingGame,
)
from src.chess_camera.reference_move_tracker import (
    ReferenceTrackerState,
    ReferenceTrackerUpdate,
)


def test_turn_indicator_is_hidden_in_diagnostic_mode() -> None:
    assert _turn_indicator_state("diagnostic", LegalGame()) is None


def test_lichess_token_status_reports_environment_without_revealing_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LICHESS_TOKEN", "secret-token-value")

    configured = _lichess_token_configuration_message()

    assert configured == "LICHESS_TOKEN is configured in the environment."
    assert "secret-token-value" not in configured

    monkeypatch.setenv("LICHESS_TOKEN", "   ")

    assert _lichess_token_configuration_message() == (
        "LICHESS_TOKEN is not configured in the environment."
    )


def test_stable_menu_preferences_are_persisted() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock, call

    settings = MagicMock()
    window = SimpleNamespace(
        _settings=settings,
        _preferred_lichess_game_id="",
        _promotion_mode="auto-queen",
    )

    CameraWindow._save_diagnostics_preference_changed(window, False)
    CameraWindow._remember_lichess_game_selection(window, "game1234")
    CameraWindow._set_promotion_mode(window, "auto-queen")

    assert settings.setValue.call_args_list == [
        call("diagnostics/save_snapshots", False),
        call("lichess/selected_game_id", "game1234"),
        call("chess/promotion_mode", "auto-queen"),
    ]
    assert settings.sync.call_count == 3
    assert window._preferred_lichess_game_id == "game1234"
    assert window._promotion_mode == "auto-queen"


def test_normal_close_persists_main_window_geometry() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from PySide6.QtCore import QByteArray
    from PySide6.QtGui import QCloseEvent

    settings = MagicMock()
    geometry = QByteArray(b"saved-geometry")
    window = SimpleNamespace(
        _lichess_submission_thread=None,
        _lichess_worker=None,
        _lichess_thread=None,
        _settings=settings,
        saveGeometry=MagicMock(return_value=geometry),
        _calibrator=MagicMock(),
        _alignment_monitor=MagicMock(),
        _camera=None,
        _projector_window=MagicMock(),
    )
    event = QCloseEvent()

    CameraWindow.closeEvent(window, event)

    settings.setValue.assert_called_once_with(
        "window/main_geometry",
        geometry,
    )
    settings.sync.assert_called_once_with()
    assert event.isAccepted()


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
    assert _can_start_new_lichess_game(state)


def test_new_lichess_game_is_disabled_during_active_play() -> None:
    state = LichessGameState(
        (),
        "unused",
        "white",
        60000,
        60000,
        0,
        0,
        "started",
        None,
        "*",
    )

    assert not _can_start_new_lichess_game(state)
    assert _can_start_new_lichess_game(None)


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
        "Connected by polling."
    )


def test_lichess_game_information_formats_players_and_unavailable_clocks(
) -> None:
    game = LichessGameFull(
        "oozTO3xk",
        "unused",
        LichessPlayer("white", "Meine", 1500),
        LichessPlayer("black", "Stockfish level 1", None),
        LichessGameState(
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
        ),
    )

    assert _lichess_game_information(
        game,
        clocks_available=False,
    ) == (
        "Black: Stockfish level 1",
        "Black clock: --:--",
        "White: Meine (1500)",
        "White clock: --:--",
        "Status: started; White to move",
    )


def test_lichess_game_information_formats_completed_game() -> None:
    game = LichessGameFull(
        "game1234",
        "unused",
        LichessPlayer("white", "Local", 1510),
        LichessPlayer("black", "Remote", 1490),
        LichessGameState(
            (),
            "unused",
            "white",
            65000,
            0,
            0,
            0,
            "mate",
            "white",
            "1-0",
        ),
    )

    assert _lichess_game_information(
        game,
        clocks_available=True,
    ) == (
        "Black: Remote (1490)",
        "Black clock: 0:00",
        "White: Local (1510)",
        "White clock: 1:05",
        "Status: mate; result 1-0",
    )


def test_ongoing_game_label_identifies_selection() -> None:
    game = OngoingGame(
        "jgWZLWs6",
        "jgWZLWs6abcd",
        "white",
        "unused",
        "Stockfish level 1",
        True,
    )

    assert _ongoing_game_label(game) == (
        "Stockfish level 1 - playing White, your turn - jgWZLWs6"
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


def test_lichess_board_update_keeps_reconciliation_target_current() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    original = LichessGameState(
        (),
        LegalGame().fen,
        "white",
        0,
        0,
        0,
        0,
        "started",
        None,
        "*",
    )
    legal_game = LegalGame()
    moved = legal_game.apply_move((6, 4), (4, 4))
    current = LichessGameState(
        ("e2e4",),
        moved.fen,
        "black",
        0,
        0,
        0,
        0,
        "started",
        None,
        "*",
    )
    full = LichessGameFull(
        "game1234",
        "startpos",
        LichessPlayer("white", "Local", 1500),
        LichessPlayer("black", "Remote", 1500),
        original,
    )
    window = SimpleNamespace(
        _latest_lichess_state=None,
        _latest_lichess_game=full,
        _lichess_game_id=full.game_id,
        _lichess_board_widget=MagicMock(),
        _update_turn_indicator=MagicMock(),
    )

    CameraWindow._update_lichess_board(window, current)

    assert window._latest_lichess_game.state == current


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
