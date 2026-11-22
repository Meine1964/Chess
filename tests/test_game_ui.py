from dataclasses import replace

import pytest

from src.chess_camera.camera_app import (
    CameraWindow,
    _accepted_position_message,
    _can_abort_lichess_game,
    _can_start_new_lichess_game,
    _current_lichess_game_text,
    _displayed_lichess_clock_state,
    _format_lichess_clock,
    _format_projected_lichess_clock,
    _illegal_correction_message,
    _lichess_board_state,
    _lichess_game_information,
    _lichess_game_state_message,
    _lichess_polling_snapshot_message,
    _lichess_turn_indicator_state,
    _lichess_token_configuration_message,
    _ongoing_game_label,
    _physical_player_edge,
    _physical_promotion_edge,
    _projected_turn_corner,
    _reference_tracking_status_message,
    _turn_indicator_state,
    _inferred_offline_game_for_source,
)
from src.chess_camera.legal_game import LegalGame
from src.chess_camera.lichess_client import (
    LichessGameFull,
    LichessGameState,
    LichessPlayer,
    LichessPollingSnapshot,
    OngoingGame,
    PlayerColor,
)
from src.chess_camera.lichess_game_controller import LichessGameController
from src.chess_camera.offline_clock import OfflineClock
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


def test_promotion_edge_follows_physical_board_orientation() -> None:
    move = ((1, 4), (0, 4))

    assert _physical_promotion_edge(move, 0) == "top"
    assert _physical_promotion_edge(move, 1) == "right"
    assert _physical_promotion_edge(move, 2) == "bottom"
    assert _physical_promotion_edge(move, 3) == "left"


def test_projected_turn_dot_tracks_a_file_and_board_orientation() -> None:
    from types import SimpleNamespace

    game = LegalGame()

    assert (
        _projected_turn_corner(True, "offline-game", game, None, 0)
        == "bottom-left"
    )
    assert (
        _projected_turn_corner(True, "offline-game", game, None, 1)
        == "top-left"
    )
    game.apply_move((6, 4), (4, 4))
    assert (
        _projected_turn_corner(True, "offline-game", game, None, 0)
        == "top-left"
    )
    assert (
        _projected_turn_corner(False, "offline-game", game, None, 0) is None
    )
    assert _projected_turn_corner(True, "diagnostic", game, None, 0) is None
    lichess_state = SimpleNamespace(result="*", turn="black")
    assert (
        _projected_turn_corner(
            True,
            "diagnostic",
            game,
            lichess_state,
            1,
        )
        == "top-right"
    )
    assert (
        _projected_turn_corner(
            True,
            "diagnostic",
            game,
            lichess_state,
            1,
            "white",
        )
        == "top-left"
    )
    lichess_state.result = "1-0"
    assert (
        _projected_turn_corner(
            True,
            "diagnostic",
            game,
            lichess_state,
            0,
        )
        is None
    )


def test_promotion_icons_face_local_player_from_either_promotion_edge() -> None:
    assert _physical_player_edge("white", 0) == "bottom"
    assert _physical_player_edge("black", 0) == "top"
    assert _physical_player_edge("white", 1) == "left"
    assert _physical_player_edge("black", 1) == "right"


def test_stable_menu_preferences_are_persisted() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock, call

    settings = MagicMock()
    window = SimpleNamespace(
        _settings=settings,
        _preferred_lichess_game_id="",
        _promotion_mode="auto-queen",
        _automatic_start_reset=True,
        _project_turn_indicator=True,
        _reset_pieces_button=MagicMock(),
        _projector_window=MagicMock(),
        _update_projected_turn_indicator=MagicMock(),
    )

    CameraWindow._save_diagnostics_preference_changed(window, False)
    CameraWindow._remember_lichess_game_selection(window, "game1234")
    CameraWindow._set_promotion_mode(window, "select-promoted-piece")
    CameraWindow._set_automatic_start_reset(window, False)
    CameraWindow._set_project_turn_indicator(window, False)

    assert settings.setValue.call_args_list == [
        call("diagnostics/save_snapshots", False),
        call("lichess/selected_game_id", "game1234"),
        call("chess/promotion_mode", "select-promoted-piece"),
        call("chess/automatic_start_reset", False),
        call("chess/project_turn_indicator", False),
    ]
    assert settings.sync.call_count == 5
    assert window._preferred_lichess_game_id == "game1234"
    assert window._promotion_mode == "select-promoted-piece"
    assert window._automatic_start_reset is False
    assert window._project_turn_indicator is False
    window._reset_pieces_button.setVisible.assert_called_once_with(True)
    window._update_projected_turn_indicator.assert_called_once_with()


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
        _latest_lichess_game=None,
        _lichess_clocks_available=False,
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
        _latest_lichess_game=None,
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


def test_waiting_status_does_not_overwrite_time_forfeit() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    game = LichessGameFull(
        "timeout1",
        "unused",
        LichessPlayer("white", "Local", 1510),
        LichessPlayer("black", "Remote", 1490),
        LichessGameState(
            (),
            "unused",
            "white",
            0,
            610000,
            5000,
            5000,
            "outoftime",
            "black",
            "0-1",
        ),
    )
    window = SimpleNamespace(
        _show_lichess_connected=MagicMock(),
        _latest_lichess_game=game,
        _lichess_clocks_available=True,
        _lichess_status=MagicMock(),
        _record_runtime_diagnostic=MagicMock(),
    )

    CameraWindow._lichess_connection_status_changed(
        window,
        "Connected to Lichess by polling; waiting for a game.",
    )

    message = window._lichess_status.setText.call_args.args[0]
    assert "White lost on time; result 0-1" in message
    assert "waiting for a game" not in message


def test_no_ongoing_games_retains_terminal_game_information() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    terminal_game = LichessGameFull(
        "timeout1",
        "unused",
        LichessPlayer("white", "Local", 1510),
        LichessPlayer("black", "Remote", 1490),
        LichessGameState(
            (),
            "unused",
            "white",
            0,
            610000,
            5000,
            5000,
            "outoftime",
            "black",
            "0-1",
        ),
    )
    window = SimpleNamespace(
        _show_lichess_connected=MagicMock(),
        _mark_lichess_game_finished=MagicMock(),
        _lichess_game_id=terminal_game.game_id,
        _latest_lichess_game=terminal_game,
        _preferred_lichess_game_id=terminal_game.game_id,
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

    window._clear_lichess_game_information.assert_not_called()


def test_promotion_blackout_detects_choice_and_shows_confirmation() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    import cv2
    import numpy as np

    from src.chess_camera.camera_app import _PendingPromotionSelection
    from src.chess_camera.promotion_selection import (
        promotion_choice_polygons,
    )

    grid = np.asarray(
        [
            [[100 + column * 50, 100 + row * 50] for column in range(9)]
            for row in range(9)
        ],
        dtype=np.float32,
    )
    polygons = promotion_choice_polygons(grid, "top")
    reference = np.full((600, 600, 3), 120, dtype=np.uint8)
    current = reference.copy()
    cv2.circle(current, (325, 75), 14, (20, 20, 20), -1)
    selection = _PendingPromotionSelection(
        ((1, 4), (0, 4)),
        "top",
        polygons,
        reference,
        stage="choices-blackout",
        generation=3,
    )
    window = SimpleNamespace(
        _pending_promotion_selection=selection,
        _latest_frame=current,
        _show_promotion_confirmation=MagicMock(),
        _show_promotion_choices=MagicMock(),
        _fail_promotion_selection=MagicMock(),
    )

    CameraWindow._finish_promotion_blackout(window, 3)

    assert selection.piece == "r"
    window._show_promotion_confirmation.assert_called_once_with()


def test_promotion_confirmation_allows_changing_selected_piece() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    import cv2
    import numpy as np

    from src.chess_camera.camera_app import _PendingPromotionSelection
    from src.chess_camera.promotion_selection import (
        promotion_choice_polygons,
    )

    grid = np.asarray(
        [
            [[100 + column * 50, 100 + row * 50] for column in range(9)]
            for row in range(9)
        ],
        dtype=np.float32,
    )
    polygons = promotion_choice_polygons(grid, "top")
    reference = np.full((600, 600, 3), 120, dtype=np.uint8)
    current = reference.copy()
    cv2.circle(current, (275, 75), 14, (20, 20, 20), -1)
    selection = _PendingPromotionSelection(
        ((1, 4), (0, 4)),
        "top",
        polygons,
        reference,
        stage="confirmation-blackout",
        piece="r",
        generation=4,
    )
    window = SimpleNamespace(
        _pending_promotion_selection=selection,
        _latest_frame=current,
        _show_promotion_confirmation=MagicMock(),
        _fail_promotion_selection=MagicMock(),
    )

    CameraWindow._finish_promotion_blackout(window, 4)

    assert selection.piece == "b"
    window._show_promotion_confirmation.assert_called_once_with()


def test_verified_replacement_submits_selected_promotion() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    import numpy as np

    from src.chess_camera.camera_app import _PendingPromotionSelection
    from src.chess_camera.reference_move_tracker import (
        ReferenceTrackerUpdate,
    )

    move = ((1, 4), (0, 4))
    selection = _PendingPromotionSelection(
        move,
        "top",
        np.zeros((4, 4, 2), dtype=np.float32),
        np.zeros((20, 20, 3), dtype=np.uint8),
        stage="verifying",
        piece="n",
    )
    result = MagicMock()
    controller = MagicMock()
    controller.prepare_local_move.return_value = ("e7e8n", result)
    tracker = MagicMock()
    tracker.observe.return_value = ReferenceTrackerUpdate(
        "accepted",
        move,
        "Promotion replacement accepted.",
    )
    begin_submission = MagicMock()
    window = SimpleNamespace(
        _pending_promotion_selection=selection,
        _lichess_game_controller=controller,
        _latest_frame=np.zeros((20, 20, 3), dtype=np.uint8),
        _calibration_result=SimpleNamespace(
            mapping=SimpleNamespace(
                camera_grid=np.zeros((9, 9, 2), dtype=np.float32)
            )
        ),
        _reference_move_tracker=tracker,
        _board_quarter_turns=0,
        _fail_promotion_selection=MagicMock(),
        _begin_lichess_local_move_submission=begin_submission,
        _show_promotion_confirmation=MagicMock(),
    )

    CameraWindow._verify_promotion_replacement(window)

    controller.prepare_local_move.assert_called_once_with(move, "n")
    begin_submission.assert_called_once_with(move, "e7e8n", result)
    assert window._pending_promotion_selection is None


def test_offline_selected_promotion_commits_selected_piece() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from src.chess_camera.chess_position import PiecePositionSynchronizer

    game = LegalGame("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    synchronizer = PiecePositionSynchronizer()
    synchronizer.restore(game.position)
    window = SimpleNamespace(
        _piece_synchronizer=synchronizer,
        _legal_game=game,
        _reference_move_tracker=MagicMock(),
        _pending_promotion_selection=object(),
        _board_widget=MagicMock(),
        _last_rejected_move=None,
        _persist_piece_position=MagicMock(),
        _persist_move_reference=MagicMock(),
        _persist_game_state=MagicMock(),
        _offline_clock_move_accepted=MagicMock(),
        _update_turn_indicator=MagicMock(),
        _piece_status=MagicMock(),
        _project_offline_accepted_move=MagicMock(),
        _fail_promotion_selection=MagicMock(),
    )

    CameraWindow._accept_offline_promotion(
        window,
        ((1, 0), (0, 0)),
        "n",
    )

    assert game.position[(0, 0)] == "N"
    assert synchronizer.position[(0, 0)] == "N"
    assert window._pending_promotion_selection is None
    window._project_offline_accepted_move.assert_called_once_with(
        ((1, 0), (0, 0))
    )
    window._offline_clock_move_accepted.assert_called_once_with()


def test_offline_mode_mismatch_is_preserved_for_explicit_recovery() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from src.chess_camera.chess_position import PiecePositionSynchronizer

    saved = LegalGame()
    physical = LegalGame()
    physical.apply_move((6, 4), (4, 4))
    synchronizer = PiecePositionSynchronizer()
    synchronizer.restore(physical.position)
    settings = MagicMock()
    settings.value.side_effect = lambda key, default="": {
        "game/mode": "offline-game",
        "game/offline_fen": saved.fen,
    }.get(key, default)
    window = SimpleNamespace(
        _settings=settings,
        _piece_synchronizer=synchronizer,
        _legal_game=LegalGame(),
        _game_mode="diagnostic",
        _pending_offline_restore_game=None,
        _offline_mode_recovery_error=None,
        _automatic_start_reset_recovery=False,
        _activate_offline_after_reset=False,
        _automatic_start_reset=True,
    )

    CameraWindow._restore_game_mode(window)

    assert window._pending_offline_restore_game is not None
    assert window._pending_offline_restore_game.fen == saved.fen
    assert window._automatic_start_reset_recovery is True
    assert window._activate_offline_after_reset is True
    assert window._game_mode == "diagnostic"
    settings.setValue.assert_not_called()
    settings.remove.assert_not_called()
    settings.sync.assert_not_called()


def test_starting_board_automatically_resets_saved_offline_game() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from src.chess_camera.chess_position import PiecePositionSynchronizer

    saved = LegalGame()
    saved.apply_move((6, 4), (4, 4))
    synchronizer = PiecePositionSynchronizer()
    synchronizer.reset()
    settings = MagicMock()
    settings.value.side_effect = lambda key, default="": {
        "game/mode": "offline-game",
        "game/offline_fen": saved.fen,
    }.get(key, default)
    window = SimpleNamespace(
        _settings=settings,
        _piece_synchronizer=synchronizer,
        _legal_game=LegalGame(),
        _game_mode="diagnostic",
        _pending_offline_restore_game=None,
        _offline_mode_recovery_error=None,
        _automatic_start_reset_recovery=False,
        _activate_offline_after_reset=False,
        _automatic_start_reset=True,
    )

    CameraWindow._restore_game_mode(window)

    assert window._pending_offline_restore_game is not None
    assert window._automatic_start_reset_recovery is True
    assert window._activate_offline_after_reset is True
    assert window._game_mode == "diagnostic"


def test_disabled_automatic_start_reset_preserves_pending_recovery() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from src.chess_camera.chess_position import PiecePositionSynchronizer

    saved = LegalGame()
    saved.apply_move((6, 4), (4, 4))
    synchronizer = PiecePositionSynchronizer()
    synchronizer.reset()
    settings = MagicMock()
    settings.value.side_effect = lambda key, default="": {
        "game/mode": "offline-game",
        "game/offline_fen": saved.fen,
    }.get(key, default)
    window = SimpleNamespace(
        _settings=settings,
        _piece_synchronizer=synchronizer,
        _legal_game=LegalGame(),
        _game_mode="diagnostic",
        _pending_offline_restore_game=None,
        _offline_mode_recovery_error=None,
        _automatic_start_reset_recovery=False,
        _activate_offline_after_reset=False,
        _automatic_start_reset=False,
    )

    CameraWindow._restore_game_mode(window)

    assert window._pending_offline_restore_game is not None
    assert window._automatic_start_reset_recovery is False
    assert window._activate_offline_after_reset is False


def test_pending_startup_offline_recovery_is_non_modal() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    window = SimpleNamespace(
        _automatic_start_reset_recovery=False,
        _pending_offline_restore_game=LegalGame(),
        _offline_mode_recovery_error=None,
        _piece_status=MagicMock(),
        _record_runtime_diagnostic=MagicMock(),
    )

    CameraWindow._report_pending_startup_offline_recovery(window)

    message = (
        "A saved Offline game requires recovery. Select Offline mode to "
        "choose a recovery action."
    )
    window._piece_status.setText.assert_called_once_with(message)
    window._record_runtime_diagnostic.assert_called_once_with(
        "offline-recovery-pending",
        message,
    )


def test_failed_automatic_startup_reset_does_not_open_dialog() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock, patch

    window = SimpleNamespace(
        _reset_to_start_requested=True,
        _automatic_start_reset_recovery=True,
        _activate_offline_after_reset=True,
        _offline_mode_recovery_error=None,
        _piece_status=MagicMock(),
        _occupancy_status=MagicMock(),
    )

    with patch(
        "src.chess_camera.camera_app.QTimer.singleShot"
    ) as single_shot:
        CameraWindow._complete_starting_position_reset(
            window,
            frozenset(),
        )

    single_shot.assert_not_called()
    assert window._automatic_start_reset_recovery is False
    assert window._activate_offline_after_reset is False
    assert window._offline_mode_recovery_error is not None
    assert "Select Offline mode" in window._offline_mode_recovery_error


def test_explicit_offline_selection_opens_pending_recovery_dialog() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    selector = MagicMock()
    selector.itemData.return_value = "offline-game"
    synchronizer = MagicMock()
    synchronizer.position = {}
    window = SimpleNamespace(
        _game_mode_selector=selector,
        _piece_synchronizer=synchronizer,
        _legal_game=LegalGame(),
        _game_mode="diagnostic",
        _sync_game_mode_actions=MagicMock(),
        _update_turn_indicator=MagicMock(),
        _pending_offline_restore_game=LegalGame(),
        _offline_mode_recovery_error="Saved Offline recovery is pending.",
        _automatic_start_reset=True,
        _show_offline_mode_recovery_dialog=MagicMock(),
        _begin_automatic_offline_start_reset=MagicMock(),
    )

    CameraWindow._game_mode_changed(window, 1)

    window._show_offline_mode_recovery_dialog.assert_called_once_with()
    window._begin_automatic_offline_start_reset.assert_not_called()


def test_stable_starting_occupancy_triggers_automatic_reset() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from src.chess_camera.chess_position import starting_occupied_squares

    detector = MagicMock()
    detector.observe.return_value = SimpleNamespace(
        state="stable",
        result=SimpleNamespace(
            occupied_squares=starting_occupied_squares()
        ),
    )
    window = SimpleNamespace(
        _automatic_start_reset=True,
        _lichess_game_controller=None,
        _latest_frame=object(),
        _calibration_result=SimpleNamespace(
            mapping=SimpleNamespace(camera_grid=object())
        ),
        _game_mode="diagnostic",
        _legal_game=LegalGame(),
        _piece_synchronizer=SimpleNamespace(position={}),
        _automatic_start_occupancy_detector=detector,
        _board_quarter_turns=0,
        _piece_side_orientation_needs_detection=False,
        _activate_offline_after_reset=False,
        _automatic_start_reset_recovery=True,
        _occupancy_status=MagicMock(),
        _piece_status=MagicMock(),
        _complete_starting_position_reset=MagicMock(),
    )

    detected = CameraWindow._detect_automatic_starting_position(window)

    assert detected is True
    assert window._activate_offline_after_reset is True
    assert window._automatic_start_reset_recovery is False
    window._complete_starting_position_reset.assert_called_once_with(
        starting_occupied_squares()
    )
    window._piece_status.setText.assert_called_once_with(
        "Starting position detected automatically; Offline game ready. "
        "White to move."
    )


def test_automatic_starting_position_corrects_saved_opposite_side() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock, patch

    from src.chess_camera.camera_app import CameraWindow
    from src.chess_camera.chess_position import starting_occupied_squares

    occupied = starting_occupied_squares()
    detector = MagicMock()
    detector.observe.return_value = SimpleNamespace(
        state="stable",
        result=SimpleNamespace(occupied_squares=occupied),
    )
    window = SimpleNamespace(
        _automatic_start_reset=True,
        _lichess_game_controller=None,
        _latest_frame=object(),
        _calibration_result=SimpleNamespace(
            mapping=SimpleNamespace(camera_grid=object())
        ),
        _game_mode="offline-game",
        _legal_game=LegalGame(),
        _piece_synchronizer=SimpleNamespace(
            position=LegalGame().position
        ),
        _piece_side_orientation_needs_detection=True,
        _automatic_start_occupancy_detector=detector,
        _board_quarter_turns=0,
        _settings=MagicMock(),
        _refresh_pending_lichess_guidance=MagicMock(),
        _activate_offline_after_reset=False,
        _automatic_start_reset_recovery=False,
        _occupancy_status=MagicMock(),
        _piece_status=MagicMock(),
        _complete_starting_position_reset=MagicMock(),
    )

    with patch(
        "src.chess_camera.camera_app.detect_starting_position_orientation",
        return_value=2,
    ):
        detected = CameraWindow._detect_automatic_starting_position(window)

    assert detected is True
    assert window._board_quarter_turns == 2
    assert window._piece_side_orientation_needs_detection is False
    window._settings.setValue.assert_called_once()
    window._settings.sync.assert_called_once_with()
    window._complete_starting_position_reset.assert_called_once_with(occupied)


def test_completed_starting_reset_replaces_occupancy_overlay() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from src.chess_camera.chess_position import (
        PiecePositionSynchronizer,
        starting_occupied_squares,
    )

    occupied = starting_occupied_squares()
    board_widget = MagicMock()
    window = SimpleNamespace(
        _reset_to_start_requested=True,
        _automatic_start_reset_recovery=False,
        _activate_offline_after_reset=False,
        _piece_synchronizer=PiecePositionSynchronizer(),
        _legal_game=LegalGame(),
        _update_turn_indicator=MagicMock(),
        _board_widget=board_widget,
        _persist_piece_position=MagicMock(),
        _persist_game_state=MagicMock(),
        _last_rejected_move=((1, 1), (2, 1)),
        _illegal_correction_last_projected_move=((1, 1), (2, 1)),
        _piece_status=MagicMock(),
        _latest_frame=object(),
        _calibration_result=SimpleNamespace(
            mapping=SimpleNamespace(camera_grid=object())
        ),
        _reference_move_tracker=MagicMock(),
        _board_quarter_turns=0,
        _persist_move_reference=MagicMock(),
        _occupancy_status=MagicMock(),
    )

    CameraWindow._complete_starting_position_reset(window, occupied)

    board_widget.set_occupied_squares.assert_called_once_with(occupied)
    assert window._illegal_correction_last_projected_move is None


def test_provisional_starting_occupancy_reports_fast_confirmation() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from src.chess_camera.chess_position import starting_occupied_squares

    detector = MagicMock()
    detector.observe.return_value = SimpleNamespace(
        state="stabilizing",
        observation_count=1,
        required_observations=2,
        result=SimpleNamespace(
            occupied_squares=starting_occupied_squares()
        ),
    )
    window = SimpleNamespace(
        _automatic_start_reset=True,
        _lichess_game_controller=None,
        _latest_frame=object(),
        _calibration_result=SimpleNamespace(
            mapping=SimpleNamespace(camera_grid=object())
        ),
        _game_mode="diagnostic",
        _legal_game=LegalGame(),
        _piece_synchronizer=SimpleNamespace(position={}),
        _automatic_start_occupancy_detector=detector,
        _board_quarter_turns=0,
        _occupancy_status=MagicMock(),
        _piece_status=MagicMock(),
        _complete_starting_position_reset=MagicMock(),
    )

    detected = CameraWindow._detect_automatic_starting_position(window)

    assert detected is True
    window._complete_starting_position_reset.assert_not_called()
    window._piece_status.setText.assert_called_once_with(
        "Starting position detected; confirming stability (1/2)."
    )


def test_matching_offline_mode_is_restored() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from src.chess_camera.chess_position import PiecePositionSynchronizer

    saved = LegalGame()
    saved.apply_move((6, 4), (4, 4))
    synchronizer = PiecePositionSynchronizer()
    synchronizer.restore(saved.position)
    settings = MagicMock()
    settings.value.side_effect = lambda key, default="": {
        "game/mode": "offline-game",
        "game/offline_fen": saved.fen,
    }.get(key, default)
    window = SimpleNamespace(
        _settings=settings,
        _piece_synchronizer=synchronizer,
        _legal_game=LegalGame(),
        _game_mode="diagnostic",
        _pending_offline_restore_game=None,
        _offline_mode_recovery_error=None,
        _automatic_start_reset_recovery=False,
        _activate_offline_after_reset=False,
        _automatic_start_reset=True,
    )

    CameraWindow._restore_game_mode(window)

    assert window._game_mode == "offline-game"
    assert window._legal_game.fen == saved.fen
    assert window._pending_offline_restore_game is None
    assert window._startup_offline_reference_refresh_pending is True
    settings.setValue.assert_not_called()


def test_pending_turn_inference_is_restored_with_matching_position() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from src.chess_camera.chess_position import PiecePositionSynchronizer

    position = LegalGame().position
    position.pop((6, 0))
    game = _inferred_offline_game_for_source(position, (1, 0))
    assert game is not None
    synchronizer = PiecePositionSynchronizer()
    synchronizer.restore(position)
    settings = MagicMock()
    settings.value.side_effect = lambda key, default="": {
        "game/mode": "offline-game",
        "game/offline_fen": game.fen,
        "game/offline_turn_inference_pending": True,
    }.get(key, default)
    window = SimpleNamespace(
        _settings=settings,
        _piece_synchronizer=synchronizer,
        _legal_game=LegalGame(),
        _game_mode="diagnostic",
        _offline_turn_inference_pending=False,
        _pending_offline_games=None,
        _pending_offline_restore_game=None,
        _offline_mode_recovery_error=None,
        _automatic_start_reset_recovery=False,
        _activate_offline_after_reset=False,
        _automatic_start_reset=True,
    )

    CameraWindow._restore_game_mode(window)

    assert window._game_mode == "offline-game"
    assert window._offline_turn_inference_pending is True
    assert window._pending_offline_games is not None
    assert window._startup_offline_reference_refresh_pending is True


def test_restored_offline_game_starts_clean_reference_refresh() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    window = SimpleNamespace(
        _startup_offline_reference_refresh_pending=True,
        _begin_physical_reconciliation=MagicMock(),
        _queue_clean_reconciliation_reference_capture=MagicMock(),
    )

    CameraWindow._start_pending_offline_reference_refresh(window)

    assert window._startup_offline_reference_refresh_pending is False
    window._begin_physical_reconciliation.assert_called_once_with(
        "Restored Offline game; preparing a clean move reference."
    )
    window._queue_clean_reconciliation_reference_capture.assert_called_once_with()


def test_lichess_authority_cancels_stale_offline_recovery() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    window = SimpleNamespace(
        _reset_to_start_requested=True,
        _activate_offline_after_reset=True,
        _automatic_start_reset_recovery=True,
        _offline_restore_requested=True,
        _pending_offline_restore_game=LegalGame(),
        _offline_mode_recovery_error="saved position mismatch",
        _occupancy_detector=MagicMock(),
        _reference_move_tracker=MagicMock(),
        _record_runtime_diagnostic=MagicMock(),
    )

    CameraWindow._cancel_offline_recovery_for_lichess(window)

    assert window._reset_to_start_requested is False
    assert window._activate_offline_after_reset is False
    assert window._automatic_start_reset_recovery is False
    assert window._offline_restore_requested is False
    assert window._pending_offline_restore_game is None
    assert window._offline_mode_recovery_error is None
    window._occupancy_detector.reset.assert_called_once_with()
    window._reference_move_tracker.reset_activity.assert_called_once_with()
    assert (
        window._record_runtime_diagnostic.call_args.args[0]
        == "lichess-offline-recovery-cancelled"
    )


def test_board_setup_choice_preserves_saved_offline_fen() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock, call

    settings = MagicMock()
    selector = MagicMock()
    window = SimpleNamespace(
        _pending_offline_restore_game=LegalGame(),
        _offline_mode_recovery_error="mismatch",
        _offline_restore_requested=True,
        _activate_offline_after_reset=True,
        _game_mode="offline-game",
        _game_mode_selector=selector,
        _sync_game_mode_actions=MagicMock(),
        _board_widget=MagicMock(),
        _last_rejected_move=((6, 4), (4, 4)),
        _settings=settings,
        _update_turn_indicator=MagicMock(),
        _piece_status=MagicMock(),
    )

    CameraWindow._activate_board_setup_mode(window)

    assert window._game_mode == "diagnostic"
    assert settings.setValue.call_args_list == [
        call("game/mode", "diagnostic"),
        call("game/offline_turn_inference_pending", False),
    ]
    settings.remove.assert_not_called()
    settings.sync.assert_called_once_with()
    assert window._pending_offline_restore_game is None


def test_start_offline_from_current_position_waits_for_first_move() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from src.chess_camera.chess_position import PiecePositionSynchronizer

    game = LegalGame()
    game.apply_move((6, 4), (4, 4))
    synchronizer = PiecePositionSynchronizer()
    synchronizer.restore(game.position)
    window = SimpleNamespace(
        _piece_synchronizer=synchronizer,
        _pending_offline_restore_game=LegalGame(),
        _offline_mode_recovery_error="mismatch",
        _offline_restore_requested=True,
        _automatic_start_reset_recovery=True,
        _activate_offline_after_reset=True,
        _legal_game=LegalGame(),
        _pending_offline_games=None,
        _offline_turn_inference_pending=False,
        _game_mode="diagnostic",
        _game_mode_selector=MagicMock(),
        _sync_game_mode_actions=MagicMock(),
        _board_widget=MagicMock(),
        _last_rejected_move=((6, 4), (4, 4)),
        _reference_move_tracker=MagicMock(),
        _persist_game_state=MagicMock(),
        _update_turn_indicator=MagicMock(),
        _piece_status=MagicMock(),
        _begin_physical_reconciliation=MagicMock(),
        _queue_clean_reconciliation_reference_capture=MagicMock(),
    )

    CameraWindow._start_offline_game_from_current_position(window)

    assert window._game_mode == "offline-game"
    assert window._offline_turn_inference_pending is True
    assert window._pending_offline_games is not None
    window._persist_game_state.assert_called_once_with()
    window._begin_physical_reconciliation.assert_called_once_with(
        "Current Offline position accepted; preparing a clean move reference."
    )
    window._queue_clean_reconciliation_reference_capture.assert_called_once_with()


def test_start_offline_from_standard_position_keeps_white_to_move() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from src.chess_camera.chess_position import PiecePositionSynchronizer

    synchronizer = PiecePositionSynchronizer()
    synchronizer.restore(LegalGame().position)
    window = SimpleNamespace(
        _piece_synchronizer=synchronizer,
        _pending_offline_restore_game=None,
        _offline_mode_recovery_error=None,
        _offline_restore_requested=False,
        _automatic_start_reset_recovery=False,
        _activate_offline_after_reset=False,
        _legal_game=LegalGame(),
        _pending_offline_games=None,
        _offline_turn_inference_pending=True,
        _game_mode="diagnostic",
        _game_mode_selector=MagicMock(),
        _sync_game_mode_actions=MagicMock(),
        _board_widget=MagicMock(),
        _last_rejected_move=None,
        _reference_move_tracker=MagicMock(),
        _persist_game_state=MagicMock(),
        _update_turn_indicator=MagicMock(),
        _piece_status=MagicMock(),
        _begin_physical_reconciliation=MagicMock(),
        _queue_clean_reconciliation_reference_capture=MagicMock(),
    )

    CameraWindow._start_offline_game_from_current_position(window)

    assert window._offline_turn_inference_pending is False
    assert window._pending_offline_games is None
    assert window._legal_game.status.turn == "white"
    window._queue_clean_reconciliation_reference_capture.assert_called_once_with()


def test_saved_offline_position_restore_commits_after_occupancy_match() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    import numpy as np

    from src.chess_camera.chess_position import (
        PiecePositionSynchronizer,
        starting_occupied_squares,
    )

    game = LegalGame()
    synchronizer = PiecePositionSynchronizer()
    tracker = MagicMock()
    window = SimpleNamespace(
        _legal_game=game,
        _latest_frame=np.zeros((20, 20, 3), dtype=np.uint8),
        _calibration_result=SimpleNamespace(
            mapping=SimpleNamespace(
                camera_grid=np.zeros((9, 9, 2), dtype=np.float32)
            )
        ),
        _piece_synchronizer=synchronizer,
        _reference_move_tracker=tracker,
        _board_quarter_turns=0,
        _offline_restore_requested=True,
        _board_widget=MagicMock(),
        _persist_piece_position=MagicMock(),
        _persist_move_reference=MagicMock(),
        _persist_game_state=MagicMock(),
        _piece_status=MagicMock(),
        _occupancy_status=MagicMock(),
    )

    CameraWindow._update_offline_position_restore(
        window,
        starting_occupied_squares(),
    )

    assert not window._offline_restore_requested
    assert synchronizer.position == game.position
    tracker.set_reference.assert_called_once()
    window._persist_game_state.assert_called_once_with()


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
        _runtime_diagnostics=MagicMock(),
    )
    event = QCloseEvent()

    CameraWindow.closeEvent(window, event)

    settings.setValue.assert_called_once_with(
        "window/main_geometry",
        geometry,
    )
    settings.sync.assert_called_once_with()
    window._runtime_diagnostics.close.assert_called_once_with()
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
        "Black to move physically",
    )
    assert _lichess_turn_indicator_state(state, "white") == (
        "#ffffff",
        "White to move physically",
    )


def test_projected_lichess_clocks_use_fixed_minute_width() -> None:
    assert _format_projected_lichess_clock(9 * 60_000 + 42_000) == "09:42"
    assert _format_projected_lichess_clock(10 * 60_000 + 5_000) == "10:05"


def test_lichess_clocks_show_hours_above_one_hour() -> None:
    assert _format_lichess_clock(59 * 60_000 + 59_000) == "59:59"
    assert _format_lichess_clock(60 * 60_000) == "1.00:00"
    assert _format_lichess_clock(89 * 60_000 + 20_000) == "1.29:20"
    assert _format_projected_lichess_clock(89 * 60_000 + 20_000) == (
        "1.29:20"
    )


def test_displayed_lichess_clock_counts_down_only_active_player() -> None:
    white_to_move = LichessGameState(
        (),
        "unused",
        "white",
        65000,
        9000,
        0,
        0,
        "started",
        None,
        "*",
    )
    black_to_move = LichessGameState(
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

    assert _displayed_lichess_clock_state(
        white_to_move,
        elapsed_ms=2500,
        clocks_available=True,
    ) == replace(white_to_move, white_time_ms=62500)
    assert _displayed_lichess_clock_state(
        black_to_move,
        elapsed_ms=12000,
        clocks_available=True,
    ) == replace(black_to_move, black_time_ms=0)


def test_displayed_lichess_clock_preserves_non_counting_states() -> None:
    active = LichessGameState(
        (),
        "unused",
        "white",
        65000,
        9000,
        0,
        0,
        "started",
        None,
        "*",
    )
    terminal = replace(
        active,
        white_time_ms=0,
        status="outoftime",
        winner="black",
        result="0-1",
    )

    assert _displayed_lichess_clock_state(
        active,
        elapsed_ms=5000,
        clocks_available=False,
    ) is active
    assert _displayed_lichess_clock_state(
        terminal,
        elapsed_ms=5000,
        clocks_available=True,
    ) is terminal


def test_lichess_clock_refresh_updates_labels_from_monotonic_elapsed_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    state = LichessGameState(
        (),
        "unused",
        "white",
        65000,
        9000,
        0,
        0,
        "started",
        None,
        "*",
    )
    game = LichessGameFull(
        "game1234",
        "unused",
        LichessPlayer("white", "Local", 1510),
        LichessPlayer("black", "Remote", 1490),
        state,
    )
    window = SimpleNamespace(
        _latest_lichess_game=game,
        _lichess_clock_snapshot_monotonic=10.0,
        _lichess_clocks_available=True,
        _lichess_black_clock=MagicMock(),
        _lichess_white_clock=MagicMock(),
        _update_projected_lichess_clocks=MagicMock(),
        _displayed_lichess_game=(
            lambda: CameraWindow._displayed_lichess_game(window)
        ),
    )
    monkeypatch.setattr(
        "src.chess_camera.camera_app.monotonic",
        lambda: 12.5,
    )

    CameraWindow._refresh_lichess_clock_display(window)

    window._lichess_black_clock.setText.assert_called_once_with(
        "Black clock: 0:09"
    )
    window._lichess_white_clock.setText.assert_called_once_with(
        "White clock: 1:02"
    )
    window._update_projected_lichess_clocks.assert_called_once()
    projected_game = (
        window._update_projected_lichess_clocks.call_args.args[0]
    )
    assert projected_game.state.white_time_ms == 62500
    assert projected_game.state.black_time_ms == 9000
    assert window._latest_lichess_game.state is state


@pytest.mark.parametrize(
    (
        "winner",
        "result",
        "white_clock",
        "black_clock",
        "projected_result",
        "expired_player",
    ),
    (
        ("black", "0-1", "00:00", "10:10", "0-1", "white"),
        ("white", "1-0", "02:05", "00:00", "1-0", "black"),
    ),
)
def test_time_forfeit_keeps_projected_clocks_and_replaces_turn_dots(
    winner: PlayerColor,
    result: str,
    white_clock: str,
    black_clock: str,
    projected_result: str,
    expired_player: PlayerColor,
) -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    game = LichessGameFull(
        "timeout1",
        "unused",
        LichessPlayer("white", "Local", 1510),
        LichessPlayer("black", "Remote", 1490),
        LichessGameState(
            (),
            "unused",
            "white",
            125000,
            610000,
            5000,
            5000,
            "outoftime",
            winner,
            result,
        ),
    )
    window = SimpleNamespace(
        _lichess_clocks_available=True,
        _lichess_clock_snapshot_monotonic=None,
        _calibration_result=object(),
        _board_orientation_needs_detection=False,
        _piece_side_orientation_needs_detection=False,
        _projector_window=MagicMock(),
        _board_quarter_turns=2,
    )

    CameraWindow._update_projected_lichess_clocks(window, game)

    window._projector_window.set_edge_clocks.assert_called_once_with(
        white_clock,
        black_clock,
        board_quarter_turns=2,
        result=projected_result,
        expired_player=expired_player,
    )


def test_agreed_draw_keeps_stopped_projected_clocks_and_replaces_turn_dot(
) -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    game = LichessGameFull(
        "draw1",
        "unused",
        LichessPlayer("white", "Local", 1510),
        LichessPlayer("black", "Remote", 1490),
        LichessGameState(
            (),
            "unused",
            "black",
            314000,
            157000,
            5000,
            5000,
            "draw",
            None,
            "1/2-1/2",
        ),
    )
    window = SimpleNamespace(
        _lichess_clocks_available=True,
        _lichess_clock_snapshot_monotonic=None,
        _calibration_result=object(),
        _board_orientation_needs_detection=False,
        _piece_side_orientation_needs_detection=False,
        _projector_window=MagicMock(),
        _board_quarter_turns=3,
    )

    CameraWindow._update_projected_lichess_clocks(window, game)

    window._projector_window.set_edge_clocks.assert_called_once_with(
        "05:14",
        "02:37",
        board_quarter_turns=3,
        result="1/2-1/2",
        expired_player=None,
    )


def test_resignation_keeps_stopped_projected_clocks_and_shows_result() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    game = LichessGameFull(
        "resigned1",
        "unused",
        LichessPlayer("white", "Local", 1510),
        LichessPlayer("black", "Remote", 1490),
        LichessGameState(
            (),
            "unused",
            "white",
            284000,
            600000,
            0,
            0,
            "resign",
            "black",
            "0-1",
        ),
    )
    window = SimpleNamespace(
        _lichess_clocks_available=True,
        _lichess_clock_snapshot_monotonic=None,
        _calibration_result=object(),
        _board_orientation_needs_detection=False,
        _piece_side_orientation_needs_detection=False,
        _projector_window=MagicMock(),
        _board_quarter_turns=1,
    )

    CameraWindow._update_projected_lichess_clocks(window, game)

    window._projector_window.set_edge_clocks.assert_called_once_with(
        "04:44",
        "10:00",
        board_quarter_turns=1,
        result="0-1",
        expired_player=None,
    )


def test_terminal_lichess_result_projects_without_available_clocks() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    game = LichessGameFull(
        "mate1",
        "unused",
        LichessPlayer("white", "Local", 1510),
        LichessPlayer("black", "Remote", 1490),
        LichessGameState(
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
        ),
    )
    window = SimpleNamespace(
        _lichess_clocks_available=False,
        _lichess_clock_snapshot_monotonic=None,
        _calibration_result=object(),
        _board_orientation_needs_detection=False,
        _piece_side_orientation_needs_detection=False,
        _projector_window=MagicMock(),
        _board_quarter_turns=2,
    )

    CameraWindow._update_projected_lichess_clocks(window, game)

    window._projector_window.set_edge_clocks.assert_called_once_with(
        None,
        None,
        board_quarter_turns=2,
        result="0-1",
    )


def test_successful_resignation_immediately_sets_authoritative_result() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    state = LichessGameState(
        (),
        "unused",
        "white",
        284000,
        600000,
        0,
        0,
        "started",
        None,
        "*",
    )
    game = LichessGameFull(
        "resigned1",
        "unused",
        LichessPlayer("white", "Local", 1510),
        LichessPlayer("black", "Remote", 1490),
        state,
    )
    window = SimpleNamespace(
        _lichess_game_id="resigned1",
        _lichess_player_color="white",
        _latest_lichess_state=state,
        _latest_lichess_game=game,
        _lichess_clocks_available=True,
        _resign_requested_game_id=None,
        _pending_lichess_game_action="resign",
        _lichess_status=MagicMock(),
        _record_runtime_diagnostic=MagicMock(),
        _update_lichess_game_control_buttons=MagicMock(),
        _update_lichess_board=MagicMock(),
        _update_lichess_game_information=MagicMock(),
    )
    window._apply_successful_lichess_resignation = lambda game_id: (
        CameraWindow._apply_successful_lichess_resignation(window, game_id)
    )
    window._apply_authoritative_lichess_terminal_result = (
        lambda game_id, **kwargs: (
            CameraWindow._apply_authoritative_lichess_terminal_result(
                window,
                game_id,
                **kwargs,
            )
        )
    )

    CameraWindow._lichess_game_action_succeeded(
        window,
        "resigned1",
        "resign",
    )

    assert window._latest_lichess_state.status == "resign"
    assert window._latest_lichess_state.winner == "black"
    assert window._latest_lichess_state.result == "0-1"
    assert window._latest_lichess_game.state is window._latest_lichess_state
    window._update_lichess_game_information.assert_called_once_with(
        window._latest_lichess_game,
        clocks_available=True,
    )
    window._lichess_status.setText.assert_called_once_with(
        "Lichess resignation accepted; result 0-1."
    )
    assert (
        window._record_runtime_diagnostic.call_args.args[0]
        == "lichess-game-action-succeeded"
    )


def test_successful_draw_acceptance_immediately_sets_draw_result() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    state = LichessGameState(
        (),
        "unused",
        "black",
        284000,
        600000,
        0,
        0,
        "started",
        None,
        "*",
        black_draw_offer=True,
    )
    game = LichessGameFull(
        "drawn1",
        "unused",
        LichessPlayer("white", "Local", 1510),
        LichessPlayer("black", "Remote", 1490),
        state,
    )
    window = SimpleNamespace(
        _lichess_game_id="drawn1",
        _latest_lichess_state=state,
        _latest_lichess_game=game,
        _lichess_clocks_available=True,
        _responded_draw_offer_game_id=None,
        _pending_lichess_game_action="accept-draw",
        _lichess_status=MagicMock(),
        _record_runtime_diagnostic=MagicMock(),
        _update_lichess_game_control_buttons=MagicMock(),
        _update_lichess_board=MagicMock(),
        _update_lichess_game_information=MagicMock(),
    )
    window._apply_authoritative_lichess_terminal_result = (
        lambda game_id, **kwargs: (
            CameraWindow._apply_authoritative_lichess_terminal_result(
                window,
                game_id,
                **kwargs,
            )
        )
    )

    CameraWindow._lichess_game_action_succeeded(
        window,
        "drawn1",
        "accept-draw",
    )

    assert window._latest_lichess_state.status == "draw"
    assert window._latest_lichess_state.winner is None
    assert window._latest_lichess_state.result == "1/2-1/2"
    assert window._latest_lichess_game.state is window._latest_lichess_state
    window._update_lichess_game_information.assert_called_once_with(
        window._latest_lichess_game,
        clocks_available=True,
    )
    window._lichess_status.setText.assert_called_once_with(
        "Lichess draw offer accepted; result 1/2-1/2."
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


def test_lichess_game_information_formats_time_forfeit() -> None:
    game = LichessGameFull(
        "timeout1",
        "unused",
        LichessPlayer("white", "Local", 1510),
        LichessPlayer("black", "Remote", 1490),
        LichessGameState(
            (),
            "unused",
            "white",
            0,
            610000,
            5000,
            5000,
            "outoftime",
            "black",
            "0-1",
        ),
    )

    assert _lichess_game_information(
        game,
        clocks_available=True,
    )[-1] == "Status: White lost on time; result 0-1"
    assert "White lost on time; result 0-1" in (
        _lichess_polling_snapshot_message(
            LichessPollingSnapshot(game, True)
        )
    )


def test_current_lichess_game_text_includes_runtime_details() -> None:
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

    assert _current_lichess_game_text(
        game,
        clocks_available=True,
        synchronization_status="Final move awaiting physical confirmation.",
    ) == (
        "Game ID: game1234\n"
        "Black: Remote (1490)\n"
        "Black clock: 0:00\n"
        "White: Local (1510)\n"
        "White clock: 1:05\n"
        "Status: mate; result 1-0\n"
        "Physical synchronization: "
        "Final move awaiting physical confirmation."
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


def test_turn_indicator_is_hidden_while_offline_turn_is_unknown() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    indicator = MagicMock()
    turn_text = MagicMock()
    window = SimpleNamespace(
        _latest_lichess_state=None,
        _offline_turn_inference_pending=True,
        _game_mode="offline-game",
        _legal_game=LegalGame(),
        _turn_indicator=indicator,
        _turn_text=turn_text,
        _update_projected_turn_indicator=MagicMock(),
    )

    CameraWindow._update_turn_indicator(window)

    indicator.hide.assert_called_once_with()
    indicator.show.assert_not_called()
    turn_text.setText.assert_called_once_with("Not enforced")
    window._update_projected_turn_indicator.assert_called_once_with()


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


def test_offline_checkmate_projects_result_without_clocks() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    game = LegalGame()
    for source, destination in (
        ((6, 5), (5, 5)),
        ((1, 4), (3, 4)),
        ((6, 6), (4, 6)),
        ((0, 3), (4, 7)),
    ):
        game.apply_move(source, destination)
    window = SimpleNamespace(
        _project_turn_indicator=True,
        _game_mode="offline-game",
        _legal_game=game,
        _latest_lichess_state=None,
        _board_quarter_turns=3,
        _projector_window=MagicMock(),
    )

    CameraWindow._update_projected_turn_indicator(window)

    window._projector_window.set_turn_indicator.assert_called_once_with(None)
    window._projector_window.set_edge_clocks.assert_called_once_with(
        None,
        None,
        board_quarter_turns=3,
        result="0-1",
    )


def test_first_offline_move_starts_next_players_clock() -> None:
    from types import SimpleNamespace

    now = [0.0]
    clock = OfflineClock(600_000, 5_000, clock=lambda: now[0])
    game = LegalGame()
    game.apply_move((6, 4), (4, 4))
    window = SimpleNamespace(
        _offline_clock_enabled=True,
        _offline_clock=clock,
        _legal_game=game,
        _persist_offline_clock=lambda: None,
    )

    CameraWindow._offline_clock_move_accepted(window)

    snapshot = clock.snapshot()
    assert snapshot.active_player == "black"
    assert snapshot.running is True
    assert snapshot.white_time_ms == 605_000
    assert snapshot.black_time_ms == 605_000


def test_offline_clock_timeout_projects_result_and_expired_player() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    now = [0.0]
    clock = OfflineClock(1_000, 0, clock=lambda: now[0])
    clock.start("white")
    now[0] = 1.0
    window = SimpleNamespace(
        _offline_clock_panel=MagicMock(),
        _offline_clock_enabled=True,
        _game_mode="offline-game",
        _latest_lichess_state=None,
        _offline_clock=clock,
        _offline_white_clock=MagicMock(),
        _offline_black_clock=MagicMock(),
        _offline_clock_status=MagicMock(),
        _offline_clock_button=MagicMock(),
        _offline_turn_inference_pending=False,
        _legal_game=LegalGame(),
        _projector_window=MagicMock(),
        _board_quarter_turns=2,
    )

    CameraWindow._refresh_offline_clock_display(window)

    window._offline_clock_status.setText.assert_called_once_with(
        "White flag fell"
    )
    window._projector_window.set_edge_clocks.assert_called_once_with(
        "00:00",
        "00:01",
        board_quarter_turns=2,
        result="0-1",
        expired_player="white",
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


def test_illegal_relocation_projects_source_and_destination_red() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    move = ((6, 1), (4, 1))
    tracker = MagicMock()
    tracker.observe.return_value = ReferenceTrackerUpdate(
        "illegal-relocation",
        move,
        "Illegal relocation b2-b4.",
    )
    tracker.rejected_move_is_restored.return_value = False
    window = SimpleNamespace(
        _latest_frame=object(),
        _calibration_result=SimpleNamespace(
            mapping=SimpleNamespace(camera_grid=object())
        ),
        _pending_lichess_local_move=None,
        _lichess_game_controller=None,
        _piece_synchronizer=SimpleNamespace(position={}),
        _legal_game=LegalGame(
            "rnbqkbnr/p3pppp/2p5/1p3P2/2PpP3/3P4/PP4PP/"
            "RNBQKBNR b KQkq - 0 1"
        ),
        _game_mode="offline-game",
        _reference_move_tracker=tracker,
        _board_quarter_turns=0,
        _occupancy_status=MagicMock(),
        _last_runtime_tracker_state="watching",
        _record_runtime_diagnostic=MagicMock(),
        _restore_pending_lichess_guidance_after_no_change=MagicMock(),
        _last_rejected_move=None,
        _board_widget=SimpleNamespace(
            set_illegal_source=MagicMock(),
        ),
        _piece_status=MagicMock(),
        _project_illegal_move_correction=MagicMock(),
    )

    CameraWindow._update_reference_move_tracking(window)

    window._board_widget.set_illegal_source.assert_called_once_with(move[0])
    window._project_illegal_move_correction.assert_called_once_with(move)
    assert window._last_rejected_move == move


def test_first_accepted_black_move_determines_offline_turn() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from src.chess_camera.chess_position import PiecePositionSynchronizer

    position = LegalGame().position
    position.pop((6, 0))
    synchronizer = PiecePositionSynchronizer()
    synchronizer.restore(position)
    move = ((1, 0), (2, 0))
    tracker = MagicMock()
    tracker.observe.return_value = ReferenceTrackerUpdate(
        "accepted",
        move,
        "Accepted a7-a6.",
    )
    tracker.rejected_move_is_restored.return_value = False
    automatic_start_detector = MagicMock()
    window = SimpleNamespace(
        _latest_frame=object(),
        _calibration_result=SimpleNamespace(
            mapping=SimpleNamespace(camera_grid=object())
        ),
        _pending_lichess_local_move=None,
        _lichess_game_controller=None,
        _piece_synchronizer=synchronizer,
        _legal_game=_inferred_offline_game_for_source(position, (7, 4)),
        _game_mode="offline-game",
        _offline_turn_inference_pending=True,
        _pending_offline_games=None,
        _reference_move_tracker=tracker,
        _automatic_start_occupancy_detector=automatic_start_detector,
        _board_quarter_turns=0,
        _occupancy_status=MagicMock(),
        _last_runtime_tracker_state="watching",
        _record_runtime_diagnostic=MagicMock(),
        _restore_pending_lichess_guidance_after_no_change=MagicMock(),
        _last_rejected_move=None,
        _illegal_correction_last_projected_move=None,
        _board_widget=MagicMock(),
        _piece_status=MagicMock(),
        _project_illegal_move_correction=MagicMock(),
        _promotion_mode="auto-queen",
        _persist_piece_position=MagicMock(),
        _persist_move_reference=MagicMock(),
        _persist_game_state=MagicMock(),
        _update_turn_indicator=MagicMock(),
        _project_offline_accepted_move=MagicMock(),
    )

    CameraWindow._update_reference_move_tracking(window)

    observed_signatures = tracker.observe.call_args.args[4]
    observed_uci = {signature.uci for signature in observed_signatures}
    assert "e2e4" in observed_uci
    assert "e7e5" in observed_uci
    assert window._offline_turn_inference_pending is False
    assert window._pending_offline_games is None
    assert window._legal_game.status.turn == "white"
    assert synchronizer.position[(2, 0)] == "p"
    automatic_start_detector.reset_observations.assert_called_once_with()
    window._persist_game_state.assert_called_once_with()
    window._project_offline_accepted_move.assert_called_once_with(move)


@pytest.mark.parametrize(
    "promotion_mode",
    ["auto-queen", "select-promoted-piece"],
)
def test_pending_opponent_move_offers_compound_replies_to_tracker(
    promotion_mode: str,
) -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    controller = LichessGameController(
        "game1234",
        "startpos",
        "black",
        "Opponent",
    )
    controller.accept_server_history(())
    controller.accept_server_history(("e2e4",))
    candidate = next(
        item
        for item in controller.compound_move_candidates
        if item.reply_signature.uci == "e7e5"
    )
    tracker = MagicMock()
    tracker.observe.return_value = ReferenceTrackerUpdate(
        "compound-accepted",
        (
            candidate.reply_signature.source,
            candidate.reply_signature.destination,
        ),
        "Compound reply accepted.",
        compound_uci="e7e5",
    )
    begin_submission = MagicMock()
    window = SimpleNamespace(
        _latest_frame=object(),
        _calibration_result=SimpleNamespace(
            mapping=SimpleNamespace(camera_grid=object())
        ),
        _pending_lichess_local_move=None,
        _lichess_game_controller=controller,
        _piece_synchronizer=SimpleNamespace(
            position=controller.physical_position
        ),
        _legal_game=LegalGame(),
        _game_mode="diagnostic",
        _promotion_mode=promotion_mode,
        _reference_move_tracker=tracker,
        _board_quarter_turns=0,
        _occupancy_status=MagicMock(),
        _last_runtime_tracker_state="watching",
        _record_runtime_diagnostic=MagicMock(),
        _restore_pending_lichess_guidance_after_no_change=MagicMock(),
        _last_rejected_move=None,
        _illegal_correction_last_projected_move=None,
        _board_widget=MagicMock(),
        _piece_status=MagicMock(),
        _lichess_sync_status=MagicMock(),
        _begin_lichess_local_move_submission=begin_submission,
    )

    CameraWindow._update_reference_move_tracking(window)

    assert (
        candidate.verification_signature
        in tracker.observe.call_args.kwargs["compound_signatures"]
    )
    begin_submission.assert_called_once_with(
        (
            candidate.reply_signature.source,
            candidate.reply_signature.destination,
        ),
        "e7e5",
        candidate.final_result,
        compound_candidate=candidate,
    )


def test_compound_correction_commits_only_proven_opponent_move() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from src.chess_camera.chess_position import PiecePositionSynchronizer

    controller = LichessGameController(
        "game1234",
        "startpos",
        "white",
        "Opponent",
        physical_fen=(
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/"
            "RNBQKBNR b KQkq - 0 1"
        ),
    )
    controller.accept_server_history(("e2e4",))
    controller.accept_server_history(("e2e4", "e7e5"))
    expected = controller.expected_signature
    opponent_result = controller.pending_result
    assert expected is not None
    assert opponent_result is not None
    wrong_reply = ((6, 5), (3, 5))
    correction_squares = wrong_reply
    tracker = MagicMock()
    tracker.observe.return_value = ReferenceTrackerUpdate(
        "compound-correction",
        None,
        "Opponent move verified; undo the incorrect local reply.",
        correction_moves=(
            (expected.source, expected.destination),
            wrong_reply,
        ),
        correction_squares=correction_squares,
        opponent_move_proven=True,
    )
    tracker.rejected_move_is_restored.return_value = False
    synchronizer = PiecePositionSynchronizer()
    synchronizer.restore(controller.physical_position)
    board_widget = MagicMock()
    window = SimpleNamespace(
        _latest_frame=object(),
        _calibration_result=SimpleNamespace(
            mapping=SimpleNamespace(camera_grid=object())
        ),
        _pending_lichess_local_move=None,
        _lichess_game_controller=controller,
        _piece_synchronizer=synchronizer,
        _legal_game=LegalGame(),
        _game_mode="diagnostic",
        _promotion_mode="auto-queen",
        _reference_move_tracker=tracker,
        _board_quarter_turns=0,
        _occupancy_status=MagicMock(),
        _last_runtime_tracker_state="watching",
        _record_runtime_diagnostic=MagicMock(),
        _restore_pending_lichess_guidance_after_no_change=MagicMock(),
        _last_rejected_move=None,
        _illegal_correction_last_projected_move=None,
        _board_widget=board_widget,
        _piece_status=MagicMock(),
        _lichess_sync_status=MagicMock(),
        _last_projected_lichess_uci=expected.uci,
        _persist_piece_position=MagicMock(),
        _project_illegal_correction_squares=MagicMock(),
    )

    CameraWindow._update_reference_move_tracking(window)

    assert controller.pending_moves == ()
    assert controller.physical_position == opponent_result.position
    assert synchronizer.position == opponent_result.position
    tracker.commit_accepted.assert_not_called()
    board_widget.set_reconciliation_squares.assert_called_once_with(
        correction_squares
    )
    window._project_illegal_correction_squares.assert_called_once_with(
        correction_squares
    )


def test_compound_correction_commits_neither_when_opponent_move_is_wrong() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    controller = LichessGameController(
        "game1234",
        "startpos",
        "white",
        "Opponent",
        physical_fen=(
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/"
            "RNBQKBNR b KQkq - 0 1"
        ),
    )
    controller.accept_server_history(("e2e4",))
    controller.accept_server_history(("e2e4", "e7e5"))
    original_position = controller.physical_position
    moves = (((1, 4), (2, 4)), ((6, 5), (5, 5)))
    correction_squares = tuple(
        square for move in moves for square in move
    )
    tracker = MagicMock()
    tracker.observe.return_value = ReferenceTrackerUpdate(
        "compound-correction",
        None,
        "Opponent move is incorrect; restore both relocations.",
        correction_moves=moves,
        correction_squares=correction_squares,
        opponent_move_proven=False,
    )
    tracker.rejected_move_is_restored.return_value = False
    board_widget = MagicMock()
    window = SimpleNamespace(
        _latest_frame=object(),
        _calibration_result=SimpleNamespace(
            mapping=SimpleNamespace(camera_grid=object())
        ),
        _pending_lichess_local_move=None,
        _lichess_game_controller=controller,
        _piece_synchronizer=SimpleNamespace(position=original_position),
        _legal_game=LegalGame(),
        _game_mode="diagnostic",
        _promotion_mode="auto-queen",
        _reference_move_tracker=tracker,
        _board_quarter_turns=0,
        _occupancy_status=MagicMock(),
        _last_runtime_tracker_state="watching",
        _record_runtime_diagnostic=MagicMock(),
        _restore_pending_lichess_guidance_after_no_change=MagicMock(),
        _last_rejected_move=None,
        _illegal_correction_last_projected_move=None,
        _board_widget=board_widget,
        _piece_status=MagicMock(),
        _lichess_sync_status=MagicMock(),
        _project_illegal_correction_squares=MagicMock(),
    )

    CameraWindow._update_reference_move_tracking(window)

    assert controller.pending_moves == ("e7e5",)
    assert controller.physical_position == original_position
    tracker.commit_accepted.assert_not_called()
    board_widget.set_reconciliation_squares.assert_called_once_with(
        correction_squares
    )
    window._project_illegal_correction_squares.assert_called_once_with(
        correction_squares
    )


def test_compound_correction_restoration_commits_new_reference() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    tracker = MagicMock()
    tracker.observe.return_value = ReferenceTrackerUpdate(
        "compound-correction-restored",
        ((1, 4), (3, 4)),
        "Opponent-only position verified.",
        opponent_move_proven=True,
    )
    tracker.rejected_move_is_restored.return_value = False
    board_widget = MagicMock()
    window = SimpleNamespace(
        _latest_frame=object(),
        _calibration_result=SimpleNamespace(
            mapping=SimpleNamespace(camera_grid=object())
        ),
        _pending_lichess_local_move=None,
        _lichess_game_controller=MagicMock(expected_signature=None),
        _piece_synchronizer=SimpleNamespace(position={}),
        _legal_game=LegalGame(),
        _game_mode="diagnostic",
        _promotion_mode="auto-queen",
        _reference_move_tracker=tracker,
        _board_quarter_turns=0,
        _occupancy_status=MagicMock(),
        _last_runtime_tracker_state="compound-correction",
        _record_runtime_diagnostic=MagicMock(),
        _restore_pending_lichess_guidance_after_no_change=MagicMock(),
        _last_rejected_move=((6, 5), (3, 5)),
        _illegal_correction_last_projected_move=((6, 5), (3, 5)),
        _board_widget=board_widget,
        _piece_status=MagicMock(),
        _lichess_sync_status=MagicMock(),
        _persist_move_reference=MagicMock(),
        _resume_board_edge_activity_indicator=MagicMock(),
    )

    CameraWindow._update_reference_move_tracking(window)

    tracker.commit_compound_correction_reference.assert_called_once_with()
    window._persist_move_reference.assert_called_once_with()
    board_widget.set_reconciliation_squares.assert_called_once_with(())
    assert window._last_rejected_move is None


def test_select_promotion_mode_excludes_only_compound_promotions() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    controller = LichessGameController(
        "game1234",
        "7k/P7/8/8/8/8/8/7K b - - 0 1",
        "white",
        "Opponent",
    )
    controller.accept_server_history(())
    controller.accept_server_history(("h8g8",))
    tracker = MagicMock()
    tracker.observe.return_value = ReferenceTrackerUpdate(
        "watching",
        None,
        "Watching.",
    )
    window = SimpleNamespace(
        _latest_frame=object(),
        _calibration_result=SimpleNamespace(
            mapping=SimpleNamespace(camera_grid=object())
        ),
        _pending_lichess_local_move=None,
        _lichess_game_controller=controller,
        _piece_synchronizer=SimpleNamespace(
            position=controller.physical_position
        ),
        _legal_game=LegalGame(),
        _game_mode="diagnostic",
        _promotion_mode="select-promoted-piece",
        _reference_move_tracker=tracker,
        _board_quarter_turns=0,
        _occupancy_status=MagicMock(),
        _last_runtime_tracker_state="watching",
        _record_runtime_diagnostic=MagicMock(),
        _restore_pending_lichess_guidance_after_no_change=MagicMock(),
        _last_rejected_move=None,
        _illegal_correction_last_projected_move=None,
        _board_widget=MagicMock(),
        _piece_status=MagicMock(),
        _lichess_sync_status=MagicMock(),
    )

    CameraWindow._update_reference_move_tracking(window)

    compound_signatures = tracker.observe.call_args.kwargs[
        "compound_signatures"
    ]
    compound_uci = {signature.uci for signature in compound_signatures}
    assert "h8g8+a7a8q" not in compound_uci
    assert any(uci.startswith("h8g8+h1") for uci in compound_uci)


def test_compound_submission_confirmation_commits_both_plies() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from src.chess_camera.chess_position import PiecePositionSynchronizer

    controller = LichessGameController(
        "game1234",
        "startpos",
        "black",
        "Opponent",
    )
    controller.accept_server_history(())
    controller.accept_server_history(("e2e4",))
    candidate = next(
        item
        for item in controller.compound_move_candidates
        if item.reply_signature.uci == "e7e5"
    )
    synchronizer = PiecePositionSynchronizer()
    synchronizer.restore(controller.physical_position)
    move = (
        candidate.reply_signature.source,
        candidate.reply_signature.destination,
    )
    window = SimpleNamespace(
        _lichess_game_controller=controller,
        _pending_lichess_local_move=move,
        _pending_lichess_local_uci="e7e5",
        _pending_lichess_local_result=candidate.final_result,
        _pending_lichess_compound_candidate=candidate,
        _piece_synchronizer=synchronizer,
        _reference_move_tracker=MagicMock(),
        _lichess_worker=None,
        _board_widget=MagicMock(),
        _last_rejected_move=None,
        _persist_piece_position=MagicMock(),
        _persist_move_reference=MagicMock(),
        _latest_lichess_state=None,
        _piece_status=MagicMock(),
        _lichess_sync_status=MagicMock(),
        _record_runtime_diagnostic=MagicMock(),
        _project_accepted_move=MagicMock(),
    )
    window._clear_pending_lichess_local_move = (
        lambda: CameraWindow._clear_pending_lichess_local_move(window)
    )

    CameraWindow._lichess_local_move_accepted(
        window,
        "game1234",
        "e7e5",
    )

    assert controller.pending_moves == ()
    assert controller.server_moves == ("e2e4", "e7e5")
    assert synchronizer.position == candidate.final_result.position
    window._reference_move_tracker.commit_accepted.assert_called_once_with(
        move
    )
    window._project_accepted_move.assert_called_once_with(move)
    assert window._pending_lichess_compound_candidate is None


def test_rejected_compound_reply_leaves_opponent_move_pending() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    controller = LichessGameController(
        "game1234",
        "startpos",
        "black",
        "Opponent",
    )
    controller.accept_server_history(())
    controller.accept_server_history(("e2e4",))
    candidate = next(
        item
        for item in controller.compound_move_candidates
        if item.reply_signature.uci == "e7e5"
    )
    move = (
        candidate.reply_signature.source,
        candidate.reply_signature.destination,
    )
    window = SimpleNamespace(
        _lichess_game_controller=controller,
        _pending_lichess_local_move=move,
        _pending_lichess_local_uci="e7e5",
        _pending_lichess_local_result=candidate.final_result,
        _pending_lichess_compound_candidate=candidate,
        _reference_move_tracker=MagicMock(),
        _last_rejected_move=None,
        _board_widget=MagicMock(),
        _piece_status=MagicMock(),
        _lichess_sync_status=MagicMock(),
        _record_runtime_diagnostic=MagicMock(),
        _project_illegal_move_correction=MagicMock(),
    )
    window._clear_pending_lichess_local_move = (
        lambda: CameraWindow._clear_pending_lichess_local_move(window)
    )

    CameraWindow._lichess_local_move_rejected(
        window,
        "game1234",
        "e7e5",
        "server rejected the move.",
    )

    assert controller.pending_moves == ("e2e4",)
    assert controller.physical_position == LegalGame().position
    window._reference_move_tracker.reset_activity.assert_called_once_with()
    window._project_illegal_move_correction.assert_called_once_with(move)
    assert window._pending_lichess_compound_candidate is None


def test_routine_tracker_transition_does_not_write_diagnostics() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    tracker = MagicMock()
    tracker.observe.return_value = ReferenceTrackerUpdate(
        "settling",
        None,
        "Board is clear; confirming a settled image.",
    )
    tracker.rejected_move_is_restored.return_value = False
    window = SimpleNamespace(
        _latest_frame=object(),
        _calibration_result=SimpleNamespace(
            mapping=SimpleNamespace(camera_grid=object())
        ),
        _pending_lichess_local_move=None,
        _lichess_game_controller=None,
        _piece_synchronizer=SimpleNamespace(position={}),
        _legal_game=LegalGame(),
        _game_mode="diagnostic",
        _reference_move_tracker=tracker,
        _board_quarter_turns=0,
        _occupancy_status=MagicMock(),
        _last_runtime_tracker_state="watching",
        _record_runtime_diagnostic=MagicMock(),
        _restore_pending_lichess_guidance_after_no_change=MagicMock(),
        _last_rejected_move=None,
    )

    CameraWindow._update_reference_move_tracking(window)

    assert window._last_runtime_tracker_state == "settling"
    window._record_runtime_diagnostic.assert_not_called()


def test_alignment_remap_waits_for_physical_move_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import MethodType, SimpleNamespace
    from unittest.mock import MagicMock

    scheduled_callbacks: list[object] = []
    monkeypatch.setattr(
        "src.chess_camera.camera_app.QTimer.singleShot",
        lambda _delay, callback: scheduled_callbacks.append(callback),
    )
    mapping = object()
    window = SimpleNamespace(
        _calibration_result=object(),
        _reference_move_tracker=SimpleNamespace(initialized=True),
        _pending_alignment_mapping=None,
        _pending_alignment_mapping_generation=0,
        _pending_alignment_mapping_verified=False,
        _verified_calibration_result=None,
        _projector_status=MagicMock(),
        _pending_lichess_local_move=None,
        _pending_promotion_selection=None,
        _last_rejected_move=None,
        _last_runtime_tracker_state="candidate",
        _apply_alignment_mapping_update=MagicMock(),
    )
    window._finish_pending_alignment_mapping = MethodType(
        CameraWindow._finish_pending_alignment_mapping,
        window,
    )
    window._alignment_mapping_conflicts_with_move_evaluation = MethodType(
        CameraWindow._alignment_mapping_conflicts_with_move_evaluation,
        window,
    )

    CameraWindow._alignment_mapping_updated(window, mapping, 4.7)

    assert window._pending_alignment_mapping == (mapping, 4.7)
    assert len(scheduled_callbacks) == 1

    CameraWindow._alignment_mapping_verified(window, mapping)
    assert window._pending_alignment_mapping_verified
    assert window._verified_calibration_result is None

    first_callback = scheduled_callbacks.pop()
    first_callback()

    window._apply_alignment_mapping_update.assert_not_called()
    assert len(scheduled_callbacks) == 1

    window._last_runtime_tracker_state = "watching"
    window._pending_lichess_local_move = ((6, 4), (4, 4))
    retry_callback = scheduled_callbacks.pop()
    retry_callback()

    window._apply_alignment_mapping_update.assert_not_called()
    assert len(scheduled_callbacks) == 1

    window._pending_lichess_local_move = None
    submission_finished_callback = scheduled_callbacks.pop()
    submission_finished_callback()

    window._apply_alignment_mapping_update.assert_called_once_with(
        mapping,
        4.7,
    )
    assert window._pending_alignment_mapping is None
    assert window._verified_calibration_result is window._calibration_result


def test_alignment_remap_applies_immediately_without_move_reference() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    mapping = object()
    apply_update = MagicMock()
    window = SimpleNamespace(
        _calibration_result=object(),
        _reference_move_tracker=SimpleNamespace(initialized=False),
        _apply_alignment_mapping_update=apply_update,
    )

    CameraWindow._alignment_mapping_updated(window, mapping, 8.0)

    apply_update.assert_called_once_with(mapping, 8.0)


def test_illegal_first_move_keeps_offline_turn_undetermined() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    move = ((1, 0), (4, 0))
    tracker = MagicMock()
    tracker.observe.return_value = ReferenceTrackerUpdate(
        "illegal-relocation",
        move,
        "Illegal relocation a7-a4.",
    )
    tracker.rejected_move_is_restored.return_value = False
    position = LegalGame().position
    window = SimpleNamespace(
        _latest_frame=object(),
        _calibration_result=SimpleNamespace(
            mapping=SimpleNamespace(camera_grid=object())
        ),
        _pending_lichess_local_move=None,
        _lichess_game_controller=None,
        _piece_synchronizer=SimpleNamespace(position=position),
        _legal_game=LegalGame(),
        _game_mode="offline-game",
        _offline_turn_inference_pending=True,
        _pending_offline_games=None,
        _reference_move_tracker=tracker,
        _board_quarter_turns=0,
        _occupancy_status=MagicMock(),
        _last_runtime_tracker_state="watching",
        _record_runtime_diagnostic=MagicMock(),
        _restore_pending_lichess_guidance_after_no_change=MagicMock(),
        _last_rejected_move=None,
        _board_widget=MagicMock(),
        _piece_status=MagicMock(),
        _project_illegal_move_correction=MagicMock(),
    )

    CameraWindow._update_reference_move_tracking(window)

    assert window._offline_turn_inference_pending is True
    assert "Black to move" in window._piece_status.setText.call_args.args[0]
    window._project_illegal_move_correction.assert_called_once_with(move)


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
