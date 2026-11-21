import os
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

import pytest
from PySide6.QtCore import QPointF, QSettings, Qt
from PySide6.QtGui import QAction, QColor
from PySide6.QtWidgets import QApplication, QComboBox, QMainWindow, QToolButton

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from src.chess_camera.chessboard_widget import (
    ChessboardWidget,
    LICHESS_DARK,
    LICHESS_ILLEGAL,
    LICHESS_LIGHT,
    rotate_square,
    square_at_position,
)
from src.chess_camera.chess_position import (
    position_fen_with_inferred_rights,
    standard_position,
)
from src.chess_camera.legal_game import LegalGame
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


def test_position_fen_infers_permissive_physical_metadata() -> None:
    assert position_fen_with_inferred_rights(standard_position()) == (
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR "
        "w KQkq - 0 1"
    )
    assert position_fen_with_inferred_rights(
        {
            (0, 4): "k",
            (3, 3): "p",
            (3, 4): "P",
            (7, 4): "K",
            (7, 7): "R",
        }
    ) == (
        "4k3/8/8/3pP3/8/8/8/4K2R w K d6 0 1"
    )
    assert position_fen_with_inferred_rights(
        {
            (0, 4): "k",
            (4, 3): "P",
            (4, 4): "p",
            (7, 4): "K",
        },
        turn="black",
    ) == (
        "4k3/8/8/8/3Pp3/8/8/4K3 b - d3 0 1"
    )
    assert position_fen_with_inferred_rights(
        {
            (0, 4): "k",
            (3, 2): "p",
            (3, 3): "P",
            (3, 5): "p",
            (3, 6): "P",
            (7, 4): "K",
        }
    ).split()[3] == "-"
    assert position_fen_with_inferred_rights(
        {(0, 0): "k", (7, 7): "K"}
    ) == (
        "k7/8/8/8/8/8/8/7K w - - 0 1"
    )
    with pytest.raises(ValueError, match="inside the 8x8 board"):
        position_fen_with_inferred_rights({(8, 0): "K"})


def test_widget_context_actions_emit_copy_requests(
    application: QApplication,
) -> None:
    widget = ChessboardWidget()
    copy_fen = MagicMock()
    copy_board = MagicMock()
    widget.copy_fen_requested.connect(copy_fen)
    widget.copy_board_requested.connect(copy_board)
    actions = {action.text(): action for action in widget.actions()}

    actions["Copy FEN"].trigger()
    actions["Copy Board"].trigger()

    copy_fen.assert_called_once_with()
    copy_board.assert_called_once_with()
    assert (
        widget.contextMenuPolicy()
        == Qt.ContextMenuPolicy.ActionsContextMenu
    )


def test_rendered_board_pixmap_excludes_layout_margins(
    application: QApplication,
) -> None:
    widget = ChessboardWidget()
    widget.resize(400, 320)
    widget.set_pieces(standard_position())
    widget.show()
    application.processEvents()

    pixmap = widget.rendered_board_pixmap()

    assert pixmap.width() == 320
    assert pixmap.height() == 320
    widget.close()


def test_board_copy_handlers_use_physical_defaults_and_lichess_authority(
    application: QApplication,
) -> None:
    from types import SimpleNamespace

    from src.chess_camera.camera_app import CameraWindow

    physical = ChessboardWidget()
    physical.set_pieces(standard_position())
    lichess = ChessboardWidget()
    lichess.set_pieces({(0, 0): "k", (7, 7): "K"})
    authoritative_fen = "8/8/8/8/8/8/4k3/4K3 b - - 17 42"
    status_bar = MagicMock()
    window = SimpleNamespace(
        _board_widget=physical,
        _lichess_board_widget=lichess,
        _lichess_game_controller=SimpleNamespace(
            server_fen=authoritative_fen
        ),
        statusBar=MagicMock(return_value=status_bar),
    )

    CameraWindow._copy_physical_board_fen(window)
    assert application.clipboard().text() == (
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR "
        "w KQkq - 0 1"
    )

    CameraWindow._copy_lichess_board_fen(window)
    assert application.clipboard().text() == authoritative_fen
    assert status_bar.showMessage.call_count == 2


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


def test_widget_shows_multiple_guidance_squares_in_white(
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
            _last_lichess_game_prompt_ids=None,
            _choose_lichess_game=MagicMock(return_value=None),
            _show_lichess_connected=MagicMock(),
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
            _last_lichess_game_prompt_ids=None,
            _show_lichess_connected=MagicMock(),
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


def test_multiple_game_dialog_previews_position_without_hash(
    application: QApplication,
) -> None:
    from src.chess_camera.camera_app import LichessGameSelectionDialog

    game = LegalGame()
    moved = game.apply_move((6, 4), (4, 4))
    games = (
        OngoingGame(
            "oldGame1",
            "oldGame1abcd",
            "black",
            LegalGame().fen,
            "Older opponent",
            True,
        ),
        OngoingGame(
            "newGame2",
            "newGame2abcd",
            "white",
            moved.fen,
            "New opponent",
            False,
            "e2e4",
        ),
    )
    dialog = LichessGameSelectionDialog(games)

    assert "oldGame1" not in dialog._selector.itemText(0)
    assert dialog._selector.itemText(0) == (
        "Older opponent - playing Black, your turn"
    )
    dialog._selector.setCurrentIndex(1)
    assert dialog.selected_game_id == "newGame2"
    assert dialog._board.pieces[(4, 4)] == "P"
    assert dialog._board.last_move == ((6, 4), (4, 4))
    assert "Opponent: New opponent." in dialog._details.text()
    dialog.close()


def test_multiple_game_popup_selection_is_applied_once(
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
            LegalGame().fen,
            "Older opponent",
            True,
        ),
        OngoingGame(
            "newGame2",
            "newGame2abcd",
            "white",
            LegalGame().fen,
            "New opponent",
            False,
        ),
    )
    selector = QComboBox()
    worker = MagicMock()
    choose = MagicMock(return_value="newGame2")
    remember = MagicMock()
    window = cast(
        CameraWindow,
        SimpleNamespace(
            _lichess_game_selector=selector,
            _lichess_worker=worker,
            _preferred_lichess_game_id="",
            _remember_lichess_game_selection=remember,
            _rebuild_lichess_games_menu=MagicMock(),
            _last_lichess_game_prompt_ids=None,
            _choose_lichess_game=choose,
            _show_lichess_connected=MagicMock(),
        ),
    )
    remember.side_effect = lambda game_id: setattr(
        window,
        "_preferred_lichess_game_id",
        game_id,
    )

    CameraWindow._lichess_ongoing_games_received(window, games)
    CameraWindow._lichess_ongoing_games_received(window, games)

    choose.assert_called_once_with(games)
    remember.assert_called_once_with("newGame2")
    assert selector.currentData() == "newGame2"
    worker.select_game.assert_called_with("newGame2")
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
    assert [action.text() for action in lichess_menu.actions()] == [
        "Active game"
    ]
    assert "Start new Lichess game" not in {
        action.text() for action in lichess_menu.actions()
    }
    window.deleteLater()


def test_lichess_control_button_has_icon_accessibility_and_placeholder_state(
    application: QApplication,
) -> None:
    from src.chess_camera.camera_app import (
        _configure_lichess_control_button,
    )

    button = QToolButton()

    _configure_lichess_control_button(
        button,
        "↶",
        "Request takeback",
        "Request takeback (not implemented)",
        enabled=False,
    )

    assert button.text() == "↶"
    assert button.accessibleName() == "Request takeback"
    assert button.toolTip() == "Request takeback (not implemented)"
    assert not button.isEnabled()
    assert button.autoRaise()
    button.close()


def test_ai_game_dialog_uses_requested_development_defaults(
    application: QApplication,
) -> None:
    from src.chess_camera.camera_app import LichessAiGameDialog

    dialog = LichessAiGameDialog(
        level=4,
        color="white",
        clock_key="10+5",
    )

    assert dialog.configuration == (4, "white", "10+5", 600, 5)
    dialog.close()


def test_lichess_ai_defaults_apply_and_cancel_transactionally(
    application: QApplication,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from PySide6.QtGui import QScreen
    from PySide6.QtWidgets import (
        QCheckBox,
        QDialog,
        QDialogButtonBox,
        QGroupBox,
        QLabel,
        QSpinBox,
    )

    from src.chess_camera.camera_app import CameraWindow

    class SettingsTestWindow(CameraWindow):
        def __init__(self) -> None:
            QMainWindow.__init__(self)
            self._settings = QSettings(
                str(tmp_path / "settings.ini"),
                QSettings.Format.IniFormat,
            )
            self._devices = []
            self._screens = list(application.screens())
            self._camera_selector = QComboBox()
            self._projector_selector = QComboBox()
            self._save_diagnostics = QCheckBox()
            self._lichess_status = QLabel("Disconnected")
            self._promotion_mode = "auto-queen"
            self._automatic_start_reset = True
            self._project_turn_indicator = True

        def _refresh_projector_screens(
            self,
            screens: list[QScreen],
        ) -> None:
            self._screens = screens

        def _set_promotion_mode(self, mode: str) -> None:
            assert mode == "auto-queen"

    window = SettingsTestWindow()
    invocation = 0

    def interact(dialog: QDialog) -> int:
        nonlocal invocation
        assert all(
            label.text() != "Disconnected"
            for label in dialog.findChildren(QLabel)
        )
        defaults = next(
            group
            for group in dialog.findChildren(QGroupBox)
            if group.title() == "New AI game defaults"
        )
        level = defaults.findChild(QSpinBox)
        combos = defaults.findChildren(QComboBox)
        color = next(combo for combo in combos if combo.findData("random") >= 0)
        clock = next(combo for combo in combos if combo.findData("15+10") >= 0)
        assert level is not None
        if invocation == 0:
            assert (level.value(), color.currentData(), clock.currentData()) == (
                4,
                "white",
                "10+5",
            )
            level.setValue(6)
            color.setCurrentIndex(color.findData("black"))
            clock.setCurrentIndex(clock.findData("15+10"))
            buttons = dialog.findChild(QDialogButtonBox)
            assert buttons is not None
            apply_button = buttons.button(
                QDialogButtonBox.StandardButton.Apply
            )
            assert apply_button is not None
            apply_button.click()
        else:
            level.setValue(2)
            color.setCurrentIndex(color.findData("random"))
            clock.setCurrentIndex(clock.findData("unlimited"))
        invocation += 1
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(QDialog, "exec", interact)

    window._show_settings_dialog()
    window._show_settings_dialog()

    assert window._settings.value("lichess/ai_level", type=int) == 6
    assert window._settings.value("lichess/ai_color") == "black"
    assert window._settings.value("lichess/ai_clock") == "15+10"
    window.deleteLater()


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


def test_main_window_placement_restores_screen_position_and_size(
    application: QApplication,
    tmp_path: Path,
) -> None:
    from src.chess_camera.camera_app import (
        _restore_window_placement,
        _save_window_placement,
    )

    settings = QSettings(
        str(tmp_path / "window-placement.ini"),
        QSettings.Format.IniFormat,
    )
    screen = application.primaryScreen()
    assert screen is not None
    available = screen.availableGeometry()
    expected_x = available.x() + 80
    expected_y = available.y() + 90
    expected_width = min(700, available.width() - 80)
    expected_height = min(500, available.height() - 90)
    previous = QMainWindow()
    previous.setGeometry(
        expected_x,
        expected_y,
        expected_width,
        expected_height,
    )

    _save_window_placement(previous, settings)

    restored = QMainWindow()
    restored.resize(320, 240)
    assert _restore_window_placement(restored, settings)
    assert restored.geometry().getRect() == (
        expected_x,
        expected_y,
        expected_width,
        expected_height,
    )

    previous.close()
    restored.close()
