import os
from typing import cast

import pytest
from PySide6.QtCore import QPointF, QUrl
from PySide6.QtGui import QAction, QColor
from PySide6.QtWidgets import QApplication, QComboBox, QMainWindow

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from src.chess_camera.chessboard_widget import (
    ChessboardWidget,
    LICHESS_DARK,
    LICHESS_ILLEGAL,
    LICHESS_LIGHT,
    rotate_square,
    square_at_position,
)
from src.chess_camera.chess_position import standard_position
from src.chess_camera.lichess_client import OngoingGame


@pytest.fixture(scope="module")
def application() -> QApplication:
    existing = QApplication.instance()
    if isinstance(existing, QApplication):
        return existing
    return QApplication([])


def test_lichess_board_colors_match_classic_brown_theme() -> None:
    assert LICHESS_LIGHT.name() == "#f0d9b5"
    assert LICHESS_DARK.name() == "#b58863"


def test_rotates_logical_square_into_physical_orientation() -> None:
    assert rotate_square(1, 2, 0) == (1, 2)
    assert rotate_square(1, 2, 1) == (2, 6)
    assert rotate_square(1, 2, 2) == (6, 5)
    assert rotate_square(1, 2, 3) == (5, 1)


def test_square_at_position_maps_top_left_to_bottom_right() -> None:
    assert square_at_position(QPointF(5, 5), 800, 800) == (0, 0)
    assert square_at_position(QPointF(795, 5), 800, 800) == (0, 7)
    assert square_at_position(QPointF(5, 795), 800, 800) == (7, 0)
    assert square_at_position(QPointF(795, 795), 800, 800) == (7, 7)


def test_square_at_position_ignores_space_outside_centered_board() -> None:
    assert square_at_position(QPointF(10, 100), 1000, 800) is None
    assert square_at_position(QPointF(110, 100), 1000, 800) == (1, 0)


def test_widget_stores_validated_occupied_squares(
    application: QApplication,
) -> None:
    widget = ChessboardWidget()

    widget.set_occupied_squares({(0, 0), (3, 4), (7, 7)})

    assert widget.occupied_squares == frozenset({(0, 0), (3, 4), (7, 7)})
    with pytest.raises(ValueError, match="inside the 8x8 board"):
        widget.set_occupied_squares({(8, 0)})


def test_widget_renders_black_dot_at_occupied_square_center(
    application: QApplication,
) -> None:
    widget = ChessboardWidget()
    widget.resize(320, 320)
    widget.set_occupied_squares({(0, 0)})
    widget.show()
    application.processEvents()

    image = widget.grab().toImage()

    assert QColor(image.pixel(20, 20)).name() == "#111111"
    assert QColor(image.pixel(60, 20)).name() != "#111111"
    widget.close()


def test_widget_renders_standard_cburnett_position(
    application: QApplication,
) -> None:
    widget = ChessboardWidget()
    widget.resize(320, 320)
    widget.set_occupied_squares({(6, 0)})
    widget.set_pieces(standard_position())
    widget.show()
    application.processEvents()

    image = widget.grab().toImage()

    assert len(widget.pieces) == 32
    assert QColor(image.pixel(20, 260)).name() != "#111111"
    widget.close()


def test_widget_persists_last_move_source_and_destination_highlight(
    application: QApplication,
) -> None:
    widget = ChessboardWidget()
    widget.resize(320, 320)
    widget.set_last_move(((4, 4), (5, 4)))
    widget.show()
    application.processEvents()

    image = widget.grab().toImage()

    assert widget.last_move == ((4, 4), (5, 4))
    assert QColor(image.pixel(180, 180)).name() == LICHESS_LIGHT.lighter(
        135
    ).name()
    assert QColor(image.pixel(180, 220)).name() == LICHESS_DARK.lighter(
        135
    ).name()
    assert QColor(image.pixel(220, 180)).name() == LICHESS_DARK.name()

    widget.set_last_move(((3, 0), (2, 0)))
    assert widget.last_move == ((3, 0), (2, 0))
    widget.set_last_move(None)
    assert widget.last_move is None
    with pytest.raises(ValueError, match="inside the 8x8 board"):
        widget.set_last_move(((4, 4), (8, 4)))
    widget.close()


def test_widget_marks_illegal_source_without_moving_piece(
    application: QApplication,
) -> None:
    widget = ChessboardWidget()
    widget.resize(320, 320)
    position = standard_position()
    widget.set_pieces(position)
    widget.set_illegal_source((7, 6))
    widget.show()
    application.processEvents()

    image = widget.grab().toImage()
    marked_square_colors = {
        QColor(image.pixel(x, y)).name()
        for x in range(240, 280)
        for y in range(280, 320)
    }

    assert widget.illegal_source == (7, 6)
    assert widget.pieces == position
    assert LICHESS_ILLEGAL.name() in marked_square_colors
    widget.set_illegal_source(None)
    assert widget.illegal_source is None
    with pytest.raises(ValueError, match="inside the 8x8 board"):
        widget.set_illegal_source((8, 0))
    widget.close()


def test_widget_marks_multiple_reconciliation_squares(
    application: QApplication,
) -> None:
    widget = ChessboardWidget()
    widget.resize(320, 320)
    widget.set_reconciliation_squares({(0, 0), (7, 7)})
    widget.show()
    application.processEvents()

    image = widget.grab().toImage()

    assert widget.reconciliation_squares == frozenset({(0, 0), (7, 7)})
    assert QColor(image.pixel(20, 20)).name() == LICHESS_ILLEGAL.name()
    assert QColor(image.pixel(300, 300)).name() == LICHESS_ILLEGAL.name()
    with pytest.raises(ValueError, match="inside the 8x8 board"):
        widget.set_reconciliation_squares({(8, 0)})
    widget.close()


def test_widget_shows_pending_legal_move_in_white(
    application: QApplication,
) -> None:
    widget = ChessboardWidget()
    widget.resize(320, 320)
    widget.set_guidance_move(((1, 1), (3, 2)))
    widget.show()
    application.processEvents()

    image = widget.grab().toImage()

    assert widget.guidance_move == ((1, 1), (3, 2))
    assert QColor(image.pixel(60, 60)).name() == "#ffffff"
    assert QColor(image.pixel(100, 140)).name() == "#ffffff"
    widget.set_guidance_move(None)
    assert widget.guidance_move is None
    with pytest.raises(ValueError, match="inside the 8x8 board"):
        widget.set_guidance_move(((1, 1), (8, 2)))
    widget.close()


def test_widget_shows_all_castling_guidance_squares_in_white(
    application: QApplication,
) -> None:
    widget = ChessboardWidget()
    widget.resize(320, 320)
    squares = ((7, 4), (7, 6), (7, 7), (7, 5))
    widget.set_guidance_squares(squares)
    widget.show()
    application.processEvents()

    image = widget.grab().toImage()

    assert widget.guidance_squares == squares
    for column in (4, 5, 6, 7):
        assert QColor(image.pixel(column * 40 + 20, 300)).name() == "#ffffff"
    widget.close()


def test_multiple_ongoing_games_require_explicit_ui_selection(
    application: QApplication,
) -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from src.chess_camera.camera_app import CameraWindow

    games = (
        OngoingGame(
            "oldGame1",
            "oldGame1abcd",
            "black",
            "unused",
            "Older opponent",
            True,
        ),
        OngoingGame(
            "jgWZLWs6",
            "jgWZLWs6abcd",
            "white",
            "unused",
            "Stockfish level 1",
            False,
        ),
    )
    selector = QComboBox()
    selector.addItem("Waiting for ongoing games...", None)
    worker = MagicMock()
    window = cast(
        CameraWindow,
        SimpleNamespace(
            _lichess_game_selector=selector,
            _lichess_worker=worker,
            _lichess_status=MagicMock(),
            _rebuild_lichess_games_menu=MagicMock(),
            _preferred_lichess_game_id="",
            _remember_lichess_game_selection=MagicMock(),
            _lichess_game_id=None,
            _clear_lichess_game_information=MagicMock(),
            _awaiting_new_lichess_game=False,
            _new_lichess_game_existing_ids=None,
        ),
    )

    CameraWindow._lichess_ongoing_games_received(window, games)

    assert selector.isEnabled()
    assert selector.count() == 3
    assert selector.currentData() is None
    worker.select_game.assert_called_once_with(None)

    selector.setCurrentIndex(selector.findData("jgWZLWs6"))
    CameraWindow._lichess_game_selection_changed(
        window,
        selector.currentIndex(),
    )

    worker.select_game.assert_called_with("jgWZLWs6")
    window._remember_lichess_game_selection.assert_called_once_with(
        "jgWZLWs6"
    )
    window._clear_lichess_game_information.assert_called_once_with(
        "Loading selected Lichess game information."
    )
    selector.close()


def test_remembered_lichess_game_is_not_replaced_by_another_game(
    application: QApplication,
) -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from src.chess_camera.camera_app import CameraWindow

    selector = QComboBox()
    worker = MagicMock()
    window = cast(
        CameraWindow,
        SimpleNamespace(
            _lichess_game_selector=selector,
            _lichess_worker=worker,
            _preferred_lichess_game_id="remembered",
            _remember_lichess_game_selection=MagicMock(),
            _rebuild_lichess_games_menu=MagicMock(),
            _awaiting_new_lichess_game=False,
            _new_lichess_game_existing_ids=None,
        ),
    )
    available = OngoingGame(
        "different",
        "differentabcd",
        "white",
        "unused",
        "Different opponent",
        True,
    )

    CameraWindow._lichess_ongoing_games_received(window, (available,))

    assert selector.currentData() is None
    assert selector.findData("different") == 1
    worker.select_game.assert_called_once_with(None)
    window._remember_lichess_game_selection.assert_not_called()
    selector.close()


def test_newly_started_game_is_selected_among_older_ongoing_games(
    application: QApplication,
) -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from src.chess_camera.camera_app import CameraWindow

    games = (
        OngoingGame(
            "oldGame1",
            "oldGame1abcd",
            "black",
            "unused",
            "Older opponent",
            True,
        ),
        OngoingGame(
            "newGame2",
            "newGame2abcd",
            "white",
            "unused",
            "New opponent",
            False,
        ),
    )
    selector = QComboBox()
    worker = MagicMock()
    remember_selection = MagicMock()
    window = cast(
        CameraWindow,
        SimpleNamespace(
            _lichess_game_selector=selector,
            _lichess_worker=worker,
            _preferred_lichess_game_id="",
            _remember_lichess_game_selection=remember_selection,
            _rebuild_lichess_games_menu=MagicMock(),
            _awaiting_new_lichess_game=True,
            _new_lichess_game_existing_ids=frozenset({"oldGame1"}),
        ),
    )

    CameraWindow._lichess_ongoing_games_received(window, games)

    assert selector.currentData() == "newGame2"
    remember_selection.assert_called_once_with("newGame2")
    worker.select_game.assert_called_once_with("newGame2")
    selector.close()


def test_settings_lists_remembered_unavailable_device() -> None:
    from src.chess_camera.camera_app import _persistent_choice_options

    options, selected_index = _persistent_choice_options(
        (("Camera B", "b"),),
        "a",
        "Mini projector",
    )

    assert options == (
        ("Camera B", "b"),
        ("Mini projector (unavailable)", "a"),
    )
    assert selected_index == 1


def test_main_commands_are_available_from_menus(
    application: QApplication,
) -> None:
    from src.chess_camera.camera_app import CameraWindow

    class MenuTestWindow(CameraWindow):
        def __init__(self) -> None:
            QMainWindow.__init__(self)
            action_names = (
                "_snapshot_button",
                "_resume_button",
                "_reset_pieces_button",
                "_reconcile_button",
                "_lichess_button",
                "_open_projector_button",
                "_focus_pattern_button",
                "_black_projector_button",
                "_close_projector_button",
                "_show_grid_button",
                "_show_squares_button",
                "_start_calibration_button",
                "_coded_calibration_button",
                "_quick_verify_button",
                "_validate_calibration_button",
                "_cancel_calibration_button",
            )
            for name in action_names:
                setattr(self, name, QAction(name, self))
            self._start_new_lichess_game_action = QAction(
                "Start new Lichess game",
                self,
            )
            self._game_mode_selector = QComboBox()
            self._game_mode_selector.addItem(
                "Diagnostic movement",
                "diagnostic",
            )
            self._game_mode_selector.addItem(
                "Offline legal game",
                "offline-game",
            )
            self._lichess_game_selector = QComboBox()
            self._lichess_game_selector.addItem("No ongoing games", None)
            self._create_application_menus()

        def _show_settings_dialog(self) -> None:
            return

    window = MenuTestWindow()

    assert [
        action.menu().title()
        for action in window.menuBar().actions()
        if action.menu() is not None
    ] == ["&File", "&Camera", "&Game", "&Projector", "&Calibration"]
    assert window._settings_action.text() == "Settings..."
    camera_actions = {
        action.text()
        for action in window.menuBar().actions()[1].menu().actions()
        if not action.isSeparator()
    }
    assert {
        "_snapshot_button",
        "_resume_button",
        "_reset_pieces_button",
        "_reconcile_button",
    } <= camera_actions
    game_menu = window.menuBar().actions()[2].menu()
    assert game_menu is not None
    lichess_menu = next(
        action.menu()
        for action in game_menu.actions()
        if action.menu() is not None
        and action.menu().title() == "Lichess"
    )
    assert lichess_menu is not None
    assert "Start new Lichess game" in {
        action.text() for action in lichess_menu.actions()
    }
    window.deleteLater()


def test_start_new_lichess_game_opens_browser_and_waits_for_game(
    application: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from src.chess_camera import camera_app
    from src.chess_camera.camera_app import CameraWindow
    from src.chess_camera.lichess_client import LichessGameState

    state = LichessGameState(
        (),
        "unused",
        "white",
        0,
        0,
        0,
        0,
        "mate",
        "white",
        "1-0",
    )
    opened_urls: list[str] = []

    class DesktopServices:
        @staticmethod
        def openUrl(url: QUrl) -> bool:
            opened_urls.append(url.toString())
            return True

    monkeypatch.setattr(camera_app, "QDesktopServices", DesktopServices)
    selector = QComboBox()
    selector.addItem("Completed game", "oldGame")
    worker = MagicMock()
    window = cast(
        CameraWindow,
        SimpleNamespace(
            _latest_lichess_state=state,
            _lichess_status=MagicMock(),
            _record_runtime_diagnostic=MagicMock(),
            _remember_lichess_game_selection=MagicMock(),
            _awaiting_new_lichess_game=False,
            _new_lichess_game_existing_ids=None,
            _lichess_game_id="oldGame",
            _lichess_player_color="white",
            _lichess_opponent_name="Opponent",
            _lichess_game_controller=MagicMock(),
            _last_projected_lichess_uci="e2e4",
            _lichess_game_selector=selector,
            _rebuild_lichess_games_menu=MagicMock(),
            _lichess_worker=worker,
            _toggle_lichess_connection=MagicMock(),
            _start_new_lichess_game_action=MagicMock(),
        ),
    )

    CameraWindow._start_new_lichess_game(window)

    assert opened_urls == ["https://lichess.org/"]
    window._remember_lichess_game_selection.assert_called_once_with("")
    worker.expect_new_game.assert_called_once_with(frozenset({"oldGame"}))
    assert window._awaiting_new_lichess_game
    assert window._lichess_game_id is None
    assert selector.currentText() == "Waiting for a new Lichess game..."
    window._start_new_lichess_game_action.setEnabled.assert_called_with(False)
    selector.close()


def test_start_new_lichess_game_preserves_state_when_browser_fails(
    application: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from src.chess_camera import camera_app
    from src.chess_camera.camera_app import CameraWindow
    from src.chess_camera.lichess_client import LichessGameState

    class DesktopServices:
        @staticmethod
        def openUrl(_url: QUrl) -> bool:
            return False

    monkeypatch.setattr(camera_app, "QDesktopServices", DesktopServices)
    state = LichessGameState(
        (),
        "unused",
        "white",
        0,
        0,
        0,
        0,
        "mate",
        "white",
        "1-0",
    )
    remember_selection = MagicMock()
    window = cast(
        CameraWindow,
        SimpleNamespace(
            _latest_lichess_state=state,
            _lichess_status=MagicMock(),
            _record_runtime_diagnostic=MagicMock(),
            _remember_lichess_game_selection=remember_selection,
            _lichess_game_id="completedGame",
        ),
    )

    CameraWindow._start_new_lichess_game(window)

    remember_selection.assert_not_called()
    assert window._lichess_game_id == "completedGame"
    window._lichess_status.setText.assert_called_once_with(
        "Could not open the Lichess website in the system browser."
    )


def test_interaction_panel_font_is_readable(
    application: QApplication,
) -> None:
    from PySide6.QtWidgets import QLabel, QWidget

    from src.chess_camera.camera_app import _set_readable_interaction_font

    panel = QWidget()
    _set_readable_interaction_font(panel)
    status = QLabel("Camera status", panel)

    assert status.font().pointSize() >= 12
    panel.close()


def test_main_window_geometry_restores_previous_size(
    application: QApplication,
) -> None:
    from src.chess_camera.camera_app import _restore_window_geometry

    previous = QMainWindow()
    previous.resize(700, 500)
    stored = previous.saveGeometry()
    restored = QMainWindow()
    restored.resize(320, 240)

    assert _restore_window_geometry(restored, stored)
    assert restored.size() == previous.size()

    previous.close()
    restored.close()
