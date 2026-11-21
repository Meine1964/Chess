import pytest

from src.chess_camera.camera_app import (
    CameraWindow,
    _accepted_position_message,
    _can_abort_lichess_game,
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


def test_created_ai_game_follows_exact_returned_id() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    connection_worker = MagicMock()
    window = SimpleNamespace(
        _pending_created_lichess_game_id=None,
        _remember_lichess_game_selection=MagicMock(),
        _lichess_worker=connection_worker,
        _lichess_status=MagicMock(),
        _record_runtime_diagnostic=MagicMock(),
    )

    CameraWindow._lichess_ai_game_created(window, "createdGame")

    assert window._pending_created_lichess_game_id == "createdGame"
    window._remember_lichess_game_selection.assert_called_once_with(
        "createdGame"
    )
    connection_worker.select_game.assert_called_once_with("createdGame")


@pytest.mark.parametrize(
    "handler_name",
    [
        "_lichess_ai_game_creation_rejected",
        "_lichess_ai_game_creation_uncertain",
    ],
)
def test_failed_ai_game_creation_restores_new_game_button(
    handler_name: str,
) -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    window = SimpleNamespace(
        _pending_created_lichess_game_id="pendingGame",
        _lichess_status=MagicMock(),
        _record_runtime_diagnostic=MagicMock(),
        _lichess_ai_creation_thread=MagicMock(),
        _lichess_ai_creation_worker=MagicMock(),
        _start_new_lichess_game_button=MagicMock(),
        _latest_lichess_state=None,
        _close_pending=False,
    )

    getattr(CameraWindow, handler_name)(window, "Creation failed.")
    CameraWindow._lichess_ai_creation_finished(window)

    assert window._pending_created_lichess_game_id is None
    window._start_new_lichess_game_button.setEnabled.assert_called_once_with(
        True
    )


def test_finished_game_event_terminalizes_cached_started_state() -> None:
    from dataclasses import replace
    from types import SimpleNamespace

    from src.chess_camera.lichess_client import LichessGameFull, LichessPlayer

    started = LichessGameState(
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
    game = LichessGameFull(
        "game1234",
        "startpos",
        LichessPlayer("white", "White", None),
        LichessPlayer("black", "Black", None),
        started,
    )
    window = SimpleNamespace(
        _lichess_game_id="game1234",
        _latest_lichess_state=started,
        _latest_lichess_game=game,
    )

    CameraWindow._mark_lichess_game_finished(window, "game1234")

    expected = replace(started, status="finished")
    assert window._latest_lichess_state == expected
    assert window._latest_lichess_game.state == expected
    assert _can_start_new_lichess_game(window._latest_lichess_state)


def test_connected_status_enables_disconnect_button() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    window = SimpleNamespace(
        _lichess_disconnect_requested=False,
        _lichess_connection_button=MagicMock(),
        _lichess_status=MagicMock(),
        _record_runtime_diagnostic=MagicMock(),
        _show_lichess_connected=lambda: CameraWindow._show_lichess_connected(
            window
        ),
    )

    CameraWindow._lichess_connection_status_changed(
        window,
        "Connected to Lichess by polling; waiting for a game.",
    )

    window._lichess_connection_button.setText.assert_called_once_with(
        "Disconnect"
    )
    window._lichess_connection_button.setEnabled.assert_called_once_with(
        True
    )


def test_polling_result_establishes_connected_button_state() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    window = SimpleNamespace(
        _show_lichess_connected=MagicMock(),
        _mark_lichess_game_finished=MagicMock(),
        _lichess_game_id=None,
        _preferred_lichess_game_id="",
        _last_lichess_game_prompt_ids=frozenset(),
        _lichess_game_selector=MagicMock(),
        _clear_lichess_game_information=MagicMock(),
        _start_new_lichess_game_button=MagicMock(),
        _lichess_ai_creation_thread=None,
        _pending_created_lichess_game_id=None,
        _rebuild_lichess_games_menu=MagicMock(),
        _lichess_worker=None,
    )

    CameraWindow._lichess_ongoing_games_received(window, ())

    window._show_lichess_connected.assert_called_once_with()


def test_normal_close_persists_main_window_geometry() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from PySide6.QtCore import QByteArray
    from PySide6.QtGui import QCloseEvent

    settings = MagicMock()
    geometry = QByteArray(b"saved-geometry")
    window = SimpleNamespace(
        _lichess_ai_creation_thread=None,
        _lichess_game_action_thread=None,
        _lichess_submission_thread=None,
        _lichess_worker=None,
        _lichess_thread=None,
        _settings=settings,
        saveGeometry=MagicMock(return_value=geometry),
        screen=MagicMock(return_value=None),
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


def test_draw_and_resign_controls_enable_for_active_game() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

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
    draw_button = MagicMock()
    resign_button = MagicMock()
    window = SimpleNamespace(
        _lichess_client=object(),
        _lichess_game_id="game1234",
        _latest_lichess_game=None,
        _latest_lichess_state=state,
        _lichess_game_action_thread=None,
        _pending_lichess_game_action=None,
        _draw_offer_game_id=None,
        _responded_draw_offer_game_id=None,
        _lichess_draw_offer_state_available=True,
        _lichess_player_color="white",
        _abort_requested_game_id=None,
        _resign_requested_game_id=None,
        _lichess_draw_button=draw_button,
        _lichess_accept_draw_button=MagicMock(),
        _lichess_decline_draw_button=MagicMock(),
        _lichess_abort_button=MagicMock(),
        _lichess_resign_button=resign_button,
    )

    CameraWindow._update_lichess_game_control_buttons(window)

    draw_button.setEnabled.assert_called_once_with(True)
    resign_button.setEnabled.assert_called_once_with(True)


def test_incoming_draw_offer_shows_accept_and_decline_controls() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

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
        False,
        True,
    )
    draw_button = MagicMock()
    accept_button = MagicMock()
    decline_button = MagicMock()
    window = SimpleNamespace(
        _lichess_client=object(),
        _lichess_game_id="game1234",
        _latest_lichess_game=None,
        _latest_lichess_state=state,
        _lichess_game_action_thread=None,
        _pending_lichess_game_action=None,
        _draw_offer_game_id=None,
        _responded_draw_offer_game_id=None,
        _lichess_draw_offer_state_available=True,
        _lichess_player_color="white",
        _abort_requested_game_id=None,
        _resign_requested_game_id=None,
        _lichess_draw_button=draw_button,
        _lichess_accept_draw_button=accept_button,
        _lichess_decline_draw_button=decline_button,
        _lichess_abort_button=MagicMock(),
        _lichess_resign_button=MagicMock(),
    )

    CameraWindow._update_lichess_game_control_buttons(window)

    draw_button.setVisible.assert_called_once_with(False)
    accept_button.setVisible.assert_called_once_with(True)
    decline_button.setVisible.assert_called_once_with(True)
    accept_button.setEnabled.assert_called_once_with(True)
    decline_button.setEnabled.assert_called_once_with(True)


def test_abort_is_available_only_during_standard_game_opening() -> None:
    state = LichessGameState(
        (),
        LegalGame().fen,
        "white",
        60000,
        60000,
        0,
        0,
        "started",
        None,
        "*",
    )
    standard = LichessGameFull(
        "game1234",
        LegalGame().fen,
        LichessPlayer("white", "Local", 1500),
        LichessPlayer("black", "Remote", 1500),
        state,
    )
    custom = LichessGameFull(
        "game1234",
        "8/8/8/8/8/8/8/K6k w - - 0 1",
        standard.white,
        standard.black,
        state,
    )

    assert _can_abort_lichess_game(standard)
    assert not _can_abort_lichess_game(
        LichessGameFull(
            standard.game_id,
            standard.initial_fen,
            standard.white,
            standard.black,
            LichessGameState(
                ("e2e4", "e7e5"),
                state.fen,
                "white",
                60000,
                60000,
                0,
                0,
                "started",
                None,
                "*",
            ),
        )
    )
    assert not _can_abort_lichess_game(custom)


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
