import argparse
import json
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from time import monotonic
from typing import Literal, cast

import cv2
import numpy as np
from PySide6.QtCore import QByteArray, QSettings, QThread, QTimer, Qt
from PySide6.QtGui import (
    QAction,
    QActionGroup,
    QCloseEvent,
    QGuiApplication,
    QImage,
    QKeyEvent,
    QPixmap,
    QResizeEvent,
    QScreen,
)

from PySide6.QtMultimedia import (
    QCamera,
    QCameraDevice,
    QCameraFormat,
    QMediaCaptureSession,
    QMediaDevices,
    QVideoFrame,
    QVideoSink,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .board_detector import BoardDetection, BoardDetector, ImageArray
from .alignment_monitor import AlignmentMonitor
from .calibration_controller import (
    CalibrationResult,
    ProjectorCalibrationController,
)
from .calibration_store import CalibrationStore
from .chess_position import (
    PieceCode,
    PiecePositionSynchronizer,
    Position,
    position_fen_with_inferred_rights,
    square_name,
    standard_position,
    starting_occupied_squares,
)
from .chessboard_widget import ChessboardWidget, rotate_square
from .diagnostics import DiagnosticStore, RuntimeDiagnosticLog
from .guidance_activity import (
    BlackoutSettlingDetector,
    BoardEdgeActivityDetector,
    GuidanceActivityDetector,
)
from .legal_game import (
    GameMode,
    LegalGame,
    LegalMoveResult,
    PromotionPiece,
)
from .lichess_client import (
    AccountGameEvent,
    AiChallengeColor,
    LichessConfigurationError,
    LichessGameFull,
    LichessGameState,
    LichessPlayer,
    LichessPollingSnapshot,
    LichessReadOnlyClient,
    OngoingGame,
    PlayerColor,
)
from .lichess_game_controller import LichessGameController
from .lichess_worker import (
    LichessAiGameCreationWorker,
    LichessConnectionWorker,
    LichessGameAction,
    LichessGameActionWorker,
    LichessMoveSubmissionWorker,
)
from .occupancy import (
    SquareOccupancyDetector,
    detect_starting_position_orientation,
)
from .piece_color_classifier import PieceColorLabel, classify_piece_colors
from .projector import ProjectorWindow, TurnIndicatorCorner, focus_pattern
from .promotion_selection import (
    ManipulationSettlingDetector,
    PromotionEdge,
    PromotionPolygons,
    observe_promotion_choices,
    promotion_choice_polygons,
    validate_promotion_polygons,
)
from .projector_calibration import (
    CalibrationError,
    MappingResult,
    bgr_to_gray,
    detect_board_orientation,
    square_polygons,
)
from .reconciliation import compare_physical_occupancy
from .reference_move_tracker import (
    ReferenceMoveTracker,
    ReferenceTrackerUpdate,
)
from .reference_store import ReferenceImageStore

_SETTINGS_ORGANIZATION = "ChessProject"
_SETTINGS_APPLICATION = "ChessboardCamera"
_CAMERA_SETTING = "camera/device_id"
_CAMERA_NAME_SETTING = "camera/device_name"
_PROJECTOR_SETTING = "projector/screen_name"
_PROJECTOR_LABEL_SETTING = "projector/screen_label"
_SAVE_DIAGNOSTICS_SETTING = "diagnostics/save_snapshots"
_LICHESS_GAME_SETTING = "lichess/selected_game_id"
_PROMOTION_MODE_SETTING = "chess/promotion_mode"
_AUTOMATIC_START_RESET_SETTING = "chess/automatic_start_reset"
_ALIGNMENT_REMAP_DELAY_MILLISECONDS = 750
_ALIGNMENT_REMAP_RETRY_MILLISECONDS = 250
_BOARD_EDGE_STUCK_DIAGNOSTIC_SECONDS = 3.0
_BOARD_EDGE_STUCK_REPEAT_SECONDS = 5.0
_ALIGNMENT_REMAP_PROTECTED_TRACKER_STATES = frozenset(
    {
        "candidate",
        "pending-capture",
        "pending-castling",
        "pending-en-passant",
        "accepted",
        "illegal-removal",
        "illegal-relocation",
    }
)
_PROJECT_TURN_INDICATOR_SETTING = "chess/project_turn_indicator"
_WINDOW_GEOMETRY_SETTING = "window/main_geometry"
_WINDOW_SCREEN_SETTING = "window/main_screen"
_WINDOW_X_OFFSET_SETTING = "window/main_x_offset"
_WINDOW_Y_OFFSET_SETTING = "window/main_y_offset"
_WINDOW_WIDTH_SETTING = "window/main_width"
_WINDOW_HEIGHT_SETTING = "window/main_height"
_WINDOW_MAXIMIZED_SETTING = "window/main_maximized"

@dataclass
class _PendingPromotionSelection:
    move: tuple[tuple[int, int], tuple[int, int]]
    edge: PromotionEdge
    camera_polygons: PromotionPolygons
    reference_frame: ImageArray
    viewer_edge: PromotionEdge = "bottom"
    stage: str = "choices"
    piece: PromotionPiece | None = None
    generation: int = 0
_LICHESS_AI_LEVEL_SETTING = "lichess/ai_level"
_LICHESS_AI_COLOR_SETTING = "lichess/ai_color"
_LICHESS_AI_CLOCK_SETTING = "lichess/ai_clock"
_PIECE_POSITION_SETTING = "pieces/accepted_position"
_GAME_MODE_SETTING = "game/mode"
_LEGAL_GAME_FEN_SETTING = "game/offline_fen"
_OFFLINE_TURN_INFERENCE_SETTING = "game/offline_turn_inference_pending"
_BOARD_ORIENTATION_SETTING = "board/automatic_clockwise_quarter_turns"
_INTERACTION_FONT_POINT_SIZE = 12
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_INTEGRATED_CAMERA_TERMS = (
    "built-in",
    "builtin",
    "front camera",
    "integrated",
    "internal",
    "user facing",
)


def _turn_indicator_state(
    game_mode: GameMode,
    legal_game: LegalGame,
) -> tuple[str, str] | None:
    if game_mode != "offline-game":
        return None
    status = legal_game.status
    if status.is_game_over:
        return "#808080", f"Game over: {status.result}"
    if status.turn == "white":
        return "#ffffff", "White to move"
    return "#000000", "Black to move"


def _inferred_offline_games(
    position: Position,
) -> tuple[LegalGame, LegalGame]:
    return (
        LegalGame(
            position_fen_with_inferred_rights(position, turn="white")
        ),
        LegalGame(
            position_fen_with_inferred_rights(position, turn="black")
        ),
    )


def _inferred_offline_game_for_source(
    position: Position,
    source: tuple[int, int],
) -> LegalGame | None:
    piece = position.get(source)
    if piece is None:
        return None
    turn: Literal["white", "black"] = (
        "white" if piece.isupper() else "black"
    )
    return LegalGame(
        position_fen_with_inferred_rights(position, turn=turn)
    )


def _illegal_correction_message(
    move: tuple[tuple[int, int], tuple[int, int]],
    legal_game: LegalGame,
) -> str:
    source, destination = move
    source_name = square_name(source)
    destination_name = square_name(destination)
    turn = legal_game.status.turn.capitalize()
    if source == destination:
        return (
            f"Illegal physical removal from {source_name} rejected. "
            f"The accepted position requires the piece on {source_name}. "
            f"{turn} to move. Return the piece to {source_name}, then make "
            f"a {turn} move."
        )
    legal_destinations = legal_game.legal_destinations(source)
    prefix = (
        f"Illegal physical move {source_name}-{destination_name} rejected. "
        f"The physical piece is on {destination_name}; the accepted position "
        f"requires it on {source_name}. {turn} to move. "
    )
    if not legal_destinations:
        return (
            f"{prefix}This piece has no legal move now. Return it from "
            f"{destination_name} to {source_name}, then make a {turn} move."
        )
    destination_list = ", ".join(
        square_name(square) for square in legal_destinations
    )
    return (
        f"{prefix}Move it from {destination_name} back to {source_name}, "
        f"or directly to a legal destination: {destination_list}."
    )


def _accepted_position_message(legal_game: LegalGame) -> str:
    status = legal_game.status
    if status.is_game_over:
        return f"Board matches the accepted position. Game over: {status.result}."
    return (
        "Board matches the accepted position. "
        f"{status.turn.capitalize()} to move."
    )


def _reference_tracking_status_message(
    update: ReferenceTrackerUpdate,
) -> str | None:
    if update.state in ("activity", "settling", "candidate"):
        return None
    if update.state == "waiting-for-clear":
        return "Move your hand and other objects away from the board."
    if update.state == "ambiguous":
        return (
            "Board change is unclear. Restore the accepted position, then "
            "move one piece at a time."
        )
    return update.message


def _format_lichess_clock(milliseconds: int) -> str:
    total_seconds = max(0, milliseconds) // 1000
    minutes, seconds = divmod(total_seconds, 60)
    return f"{minutes}:{seconds:02d}"


def _format_projected_lichess_clock(milliseconds: int) -> str:
    clock = _format_lichess_clock(milliseconds)
    minutes, separator, seconds = clock.partition(":")
    return f"{minutes.zfill(2)}{separator}{seconds}"


def _lichess_display_clock_times(
    state: LichessGameState,
) -> tuple[int, int]:
    white_time_ms = state.white_time_ms
    black_time_ms = state.black_time_ms
    if state.status == "outoftime":
        if state.winner == "white":
            black_time_ms = 0
        elif state.winner == "black":
            white_time_ms = 0
    return white_time_ms, black_time_ms


def _displayed_lichess_clock_state(
    state: LichessGameState,
    *,
    elapsed_ms: int,
    clocks_available: bool,
) -> LichessGameState:
    if (
        not clocks_available
        or state.result != "*"
        or state.status not in ("created", "started")
    ):
        return state
    elapsed_ms = max(0, elapsed_ms)
    if state.turn == "white":
        return replace(
            state,
            white_time_ms=max(0, state.white_time_ms - elapsed_ms),
        )
    return replace(
        state,
        black_time_ms=max(0, state.black_time_ms - elapsed_ms),
    )


def _lichess_terminal_status(state: LichessGameState) -> str:
    if state.status == "outoftime":
        if state.winner == "white":
            return f"Black lost on time; result {state.result}"
        if state.winner == "black":
            return f"White lost on time; result {state.result}"
        return f"Game ended on time; result {state.result}"
    return f"{state.status}; result {state.result}"


def _lichess_game_information(
    game: LichessGameFull,
    *,
    clocks_available: bool,
) -> tuple[str, str, str, str, str]:
    def player_text(player: LichessPlayer) -> str:
        color = player.color.capitalize()
        rating = f" ({player.rating})" if player.rating is not None else ""
        return f"{color}: {player.name}{rating}"

    state = game.state
    white_time_ms, black_time_ms = _lichess_display_clock_times(state)
    white_clock = (
        _format_lichess_clock(white_time_ms)
        if clocks_available
        else "--:--"
    )
    black_clock = (
        _format_lichess_clock(black_time_ms)
        if clocks_available
        else "--:--"
    )
    if state.result == "*":
        status = (
            f"Status: {state.status}; {state.turn.capitalize()} to move"
        )
    else:
        status = f"Status: {_lichess_terminal_status(state)}"
    if state.white_draw_offer:
        status += "; White offers a draw"
    if state.black_draw_offer:
        status += "; Black offers a draw"
    return (
        player_text(game.black),
        f"Black clock: {black_clock}",
        player_text(game.white),
        f"White clock: {white_clock}",
        status,
    )


def _current_lichess_game_text(
    game: LichessGameFull,
    *,
    clocks_available: bool,
    synchronization_status: str,
) -> str:
    information = _lichess_game_information(
        game,
        clocks_available=clocks_available,
    )
    return "\n".join(
        (
            f"Game ID: {game.game_id}",
            *information,
            f"Physical synchronization: {synchronization_status}",
        )
    )


def _can_start_new_lichess_game(
    state: LichessGameState | None,
) -> bool:
    return (
        state is None
        or state.result != "*"
        or state.status not in ("created", "started")
    )


def _lichess_draw_offer_flags(
    state: LichessGameState,
    player_color: PlayerColor,
) -> tuple[bool, bool]:
    if player_color == "white":
        return state.white_draw_offer, state.black_draw_offer
    return state.black_draw_offer, state.white_draw_offer


def _can_abort_lichess_game(game: LichessGameFull | None) -> bool:
    return (
        game is not None
        and game.initial_fen == LegalGame().fen
        and len(game.state.moves) < 2
        and not _can_start_new_lichess_game(game.state)
    )


def _physical_promotion_edge(
    move: tuple[tuple[int, int], tuple[int, int]],
    board_quarter_turns: int,
) -> PromotionEdge:
    straight_destination = rotate_square(
        move[1][0],
        move[0][1],
        board_quarter_turns,
    )
    if straight_destination[0] == 0:
        return "top"
    if straight_destination[0] == 7:
        return "bottom"
    if straight_destination[1] == 0:
        return "left"
    if straight_destination[1] == 7:
        return "right"
    raise ValueError("Promotion destination is not on a physical board edge.")


def _physical_player_edge(
    player_color: PlayerColor,
    board_quarter_turns: int,
) -> PromotionEdge:
    player_row = 7 if player_color == "white" else 0
    physical_row, physical_column = rotate_square(
        player_row,
        4,
        board_quarter_turns,
    )
    if physical_row == 0:
        return "top"
    if physical_row == 7:
        return "bottom"
    if physical_column == 0:
        return "left"
    return "right"


def _projected_turn_corner(
    enabled: bool,
    game_mode: GameMode,
    offline_game: LegalGame,
    lichess_state: LichessGameState | None,
    board_quarter_turns: int,
) -> TurnIndicatorCorner | None:
    if not enabled:
        return None
    if lichess_state is not None:
        if lichess_state.result != "*":
            return None
        turn = lichess_state.turn
    else:
        if game_mode != "offline-game" or offline_game.status.result != "*":
            return None
        turn = offline_game.status.turn
    if turn == "white":
        return (
            "bottom-left",
            "top-left",
            "top-right",
            "bottom-right",
        )[board_quarter_turns % 4]
    return (
        "top-left",
        "top-right",
        "bottom-right",
        "bottom-left",
    )[board_quarter_turns % 4]


def _lichess_game_state_message(state: LichessGameState) -> str:
    return (
        f"Lichess: {state.turn.capitalize()} to move; "
        f"White {_format_lichess_clock(state.white_time_ms)}, "
        f"Black {_format_lichess_clock(state.black_time_ms)}; "
        f"status {state.status}; result {state.result}."
    )


def _lichess_turn_indicator_state(
    state: LichessGameState,
) -> tuple[str, str]:
    if state.result != "*":
        return "#808080", f"Lichess game over: {state.result}"
    if state.turn == "white":
        return "#ffffff", "White to move (Lichess)"
    return "#000000", "Black to move (Lichess)"


def _lichess_polling_snapshot_message(
    snapshot: LichessPollingSnapshot,
) -> str:
    game = snapshot.game
    state = game.state
    clock_text = (
        f"White {_format_lichess_clock(state.white_time_ms)}, "
        f"Black {_format_lichess_clock(state.black_time_ms)}; "
        if snapshot.clocks_available
        else "clocks unavailable for this game; "
    )
    if state.result != "*":
        return (
            f"Lichess game {game.game_id}: "
            f"{game.white.name} (White) vs {game.black.name} (Black). "
            f"{clock_text}{_lichess_terminal_status(state)}. "
            "Connected by polling."
        )
    return (
        f"Lichess game {game.game_id}: "
        f"{game.white.name} (White) vs {game.black.name} (Black). "
        f"{state.turn.capitalize()} to move; {clock_text}"
        f"status {state.status}; result {state.result}. "
        "Connected by polling."
    )


def _ongoing_game_label(game: OngoingGame) -> str:
    turn = "your turn" if game.is_my_turn else "opponent's turn"
    return (
        f"{game.opponent_name} - playing {game.color.capitalize()}, "
        f"{turn} - {game.game_id}"
    )


def _ongoing_game_choice_label(game: OngoingGame) -> str:
    turn = "your turn" if game.is_my_turn else "opponent's turn"
    return (
        f"{game.opponent_name} - playing {game.color.capitalize()}, {turn}"
    )


def _lichess_board_state(
    state: LichessGameState,
) -> tuple[
    Position,
    tuple[tuple[int, int], tuple[int, int]] | None,
]:
    position = LegalGame(state.fen).position
    if not state.moves:
        return position, None
    uci = state.moves[-1]
    if len(uci) < 4:
        raise ValueError("A Lichess move must contain two squares.")
    return position, (
        _ui_square_from_name(uci[:2]),
        _ui_square_from_name(uci[2:4]),
    )


def _ui_square_from_name(name: str) -> tuple[int, int]:
    if (
        len(name) != 2
        or name[0] not in "abcdefgh"
        or name[1] not in "12345678"
    ):
        raise ValueError(f"Invalid chess square: {name}")
    return 8 - int(name[1]), ord(name[0]) - ord("a")


def _diagnostic_position(position: Position) -> dict[str, PieceCode]:
    return {
        square_name(square): piece
        for square, piece in sorted(position.items())
    }


def _is_external_camera(
    description: str,
    position: QCameraDevice.Position,
) -> bool:
    normalized_description = description.casefold()
    return (
        position == QCameraDevice.Position.UnspecifiedPosition
        and not any(
            term in normalized_description
            for term in _INTEGRATED_CAMERA_TERMS
        )
    )


def _resolve_camera_index(
    device_names: Sequence[str],
    device_keys: Sequence[str],
    saved_key: str,
    selector: str | None,
) -> int | None:
    if not device_names or len(device_names) != len(device_keys):
        raise RuntimeError("No usable camera devices were found.")

    if selector is not None:
        if selector.isdecimal():
            index = int(selector)
            if 0 <= index < len(device_names):
                return index
            raise RuntimeError(
                f"Camera index {index} is outside the available range "
                f"0-{len(device_names) - 1}."
            )

        matching_indices = [
            index
            for index, name in enumerate(device_names)
            if selector.casefold() in name.casefold()
        ]
        if len(matching_indices) == 1:
            return matching_indices[0]
        if not matching_indices:
            raise RuntimeError(f'No camera name contains "{selector}".')
        raise RuntimeError(f'Camera name "{selector}" is ambiguous.')

    if saved_key in device_keys:
        return device_keys.index(saved_key)
    return None


def _snapshot_board_detection(
    detector: BoardDetector,
    snapshot: ImageArray,
    calibration_result: CalibrationResult | None,
) -> BoardDetection | None:
    detection = detector.detect(snapshot)
    if detection is not None or calibration_result is None:
        return detection
    return replace(
        calibration_result.board_detection,
        grid_points=calibration_result.mapping.camera_grid.copy(),
        reprojection_error=None,
        method="verified calibrated camera grid",
    )


def _lichess_token_configuration_message() -> str:
    token = os.environ.get("LICHESS_TOKEN", "")
    return (
        "LICHESS_TOKEN is configured in the environment."
        if token.strip()
        else "LICHESS_TOKEN is not configured in the environment."
    )


def _stored_boolean(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in ("true", "1", "yes", "on"):
            return True
        if normalized in ("false", "0", "no", "off"):
            return False
    return default


def _stored_integer(value: object, default: int) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return default


def _persistent_choice_options(
    available: Sequence[tuple[str, str]],
    remembered_id: str,
    remembered_label: str,
) -> tuple[tuple[tuple[str, str], ...], int]:
    options = list(available)
    for index, (_label, identity) in enumerate(options):
        if identity == remembered_id:
            return tuple(options), index
    if remembered_id:
        label = remembered_label or remembered_id
        options.append((f"{label} (unavailable)", remembered_id))
        return tuple(options), len(options) - 1
    return tuple(options), -1


def _set_readable_interaction_font(widget: QWidget) -> None:
    font = widget.font()
    font.setPointSize(
        max(font.pointSize(), _INTERACTION_FONT_POINT_SIZE)
    )
    widget.setFont(font)


def _configure_lichess_control_button(
    button: QToolButton,
    symbol: str,
    accessible_name: str,
    tooltip: str,
    *,
    enabled: bool,
) -> None:
    button.setText(symbol)
    button.setAccessibleName(accessible_name)
    button.setToolTip(tooltip)
    button.setEnabled(enabled)
    button.setAutoRaise(True)
    button.setMinimumSize(40, 36)
    font = button.font()
    font.setPointSize(max(font.pointSize(), 18))
    button.setFont(font)


def _restore_window_geometry(
    window: QMainWindow,
    stored_geometry: object,
) -> bool:
    if isinstance(stored_geometry, QByteArray):
        geometry = stored_geometry
    elif isinstance(stored_geometry, (bytes, bytearray)):
        geometry = QByteArray(bytes(stored_geometry))
    else:
        return False
    return not geometry.isEmpty() and window.restoreGeometry(geometry)


def _save_window_placement(
    window: QMainWindow,
    settings: QSettings,
) -> None:
    settings.setValue(_WINDOW_GEOMETRY_SETTING, window.saveGeometry())
    screen = window.screen()
    if screen is not None:
        geometry = (
            window.normalGeometry()
            if window.isMaximized()
            else window.geometry()
        )
        available = screen.availableGeometry()
        settings.setValue(_WINDOW_SCREEN_SETTING, screen.name())
        settings.setValue(
            _WINDOW_X_OFFSET_SETTING,
            geometry.x() - available.x(),
        )
        settings.setValue(
            _WINDOW_Y_OFFSET_SETTING,
            geometry.y() - available.y(),
        )
        settings.setValue(_WINDOW_WIDTH_SETTING, geometry.width())
        settings.setValue(_WINDOW_HEIGHT_SETTING, geometry.height())
        settings.setValue(_WINDOW_MAXIMIZED_SETTING, window.isMaximized())
    settings.sync()


def _restore_window_placement(
    window: QMainWindow,
    settings: QSettings,
) -> bool:
    screen_name = str(settings.value(_WINDOW_SCREEN_SETTING, ""))
    screen = next(
        (
            candidate
            for candidate in QGuiApplication.screens()
            if candidate.name() == screen_name
        ),
        None,
    )
    width = settings.value(_WINDOW_WIDTH_SETTING)
    height = settings.value(_WINDOW_HEIGHT_SETTING)
    x_offset = settings.value(_WINDOW_X_OFFSET_SETTING)
    y_offset = settings.value(_WINDOW_Y_OFFSET_SETTING)
    if (
        screen is not None
        and isinstance(width, int)
        and isinstance(height, int)
        and isinstance(x_offset, int)
        and isinstance(y_offset, int)
        and width > 0
        and height > 0
    ):
        available = screen.availableGeometry()
        restored_width = min(width, available.width())
        restored_height = min(height, available.height())
        restored_x = available.x() + max(
            0,
            min(x_offset, available.width() - restored_width),
        )
        restored_y = available.y() + max(
            0,
            min(y_offset, available.height() - restored_height),
        )
        window.setGeometry(
            restored_x,
            restored_y,
            restored_width,
            restored_height,
        )
        if _stored_boolean(
            settings.value(_WINDOW_MAXIMIZED_SETTING, False),
            False,
        ):
            window.setWindowState(
                window.windowState() | Qt.WindowState.WindowMaximized
            )
        return True
    return _restore_window_geometry(
        window,
        settings.value(_WINDOW_GEOMETRY_SETTING),
    )


class LichessGameSelectionDialog(QDialog):
    def __init__(
        self,
        games: tuple[OngoingGame, ...],
        parent: QWidget | None = None,
    ) -> None:
        if len(games) < 2:
            raise ValueError(
                "The Lichess game chooser requires multiple games."
            )
        super().__init__(parent)
        self._games = games
        self.setWindowTitle("Select Lichess game")
        self._selector = QComboBox()
        for game in games:
            self._selector.addItem(
                _ongoing_game_choice_label(game),
                game.game_id,
            )
        self._details = QLabel()
        self._details.setWordWrap(True)
        self._board = ChessboardWidget()
        self._board.setMinimumSize(360, 360)
        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        self._selector.currentIndexChanged.connect(self._update_preview)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Choose the Lichess game to follow:"))
        layout.addWidget(self._selector)
        layout.addWidget(self._details)
        layout.addWidget(self._board, 1)
        layout.addWidget(self._buttons)
        self.setLayout(layout)
        self.resize(560, 620)
        self._update_preview(0)

    @property
    def selected_game_id(self) -> str:
        return self._games[self._selector.currentIndex()].game_id

    def _update_preview(self, index: int) -> None:
        if not 0 <= index < len(self._games):
            return
        game = self._games[index]
        position = LegalGame(game.current_fen).position
        self._board.set_pieces(position)
        self._board.set_occupied_squares(position)
        last_move = (
            None
            if game.last_move is None
            else (
                _ui_square_from_name(game.last_move[:2]),
                _ui_square_from_name(game.last_move[2:4]),
            )
        )
        self._board.set_last_move(last_move)
        turn_owner = "your" if game.is_my_turn else "the opponent's"
        self._details.setText(
            f"Opponent: {game.opponent_name}. "
            f"You play {game.color.capitalize()}. "
            f"It is {turn_owner} turn."
        )


class LichessAiGameDialog(QDialog):
    _CLOCKS: tuple[
        tuple[str, str, int | None, int | None],
        ...,
    ] = (
        ("Unlimited", "unlimited", None, None),
        ("10 minutes + 0 seconds", "10+0", 600, 0),
        ("10 minutes + 5 seconds", "10+5", 600, 5),
        ("15 minutes + 10 seconds", "15+10", 900, 10),
    )

    def __init__(
        self,
        *,
        level: int = 4,
        color: AiChallengeColor = "white",
        clock_key: str = "10+5",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("New Lichess AI game")
        self._level = QSpinBox()
        self._level.setRange(1, 8)
        self._level.setValue(min(8, max(1, level)))
        self._color = QComboBox()
        for label, value in (
            ("White", "white"),
            ("Black", "black"),
            ("Random", "random"),
        ):
            self._color.addItem(label, value)
        color_index = self._color.findData(color)
        self._color.setCurrentIndex(max(0, color_index))
        self._clock = QComboBox()
        for label, key, limit, increment in self._CLOCKS:
            self._clock.addItem(label, (key, limit, increment))
        clock_index = next(
            (
                index
                for index, (_label, key, _limit, _increment)
                in enumerate(self._CLOCKS)
                if key == clock_key
            ),
            2,
        )
        self._clock.setCurrentIndex(clock_index)
        form = QFormLayout()
        form.addRow("AI level:", self._level)
        form.addRow("Your color:", self._color)
        form.addRow("Clock:", self._clock)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout()
        layout.addLayout(form)
        layout.addWidget(buttons)
        self.setLayout(layout)

    @property
    def configuration(
        self,
    ) -> tuple[int, AiChallengeColor, str, int | None, int | None]:
        color = cast(AiChallengeColor, self._color.currentData())
        clock_key, limit, increment = cast(
            tuple[str, int | None, int | None],
            self._clock.currentData(),
        )
        return self._level.value(), color, clock_key, limit, increment


class CameraWindow(QMainWindow):
    def __init__(self, camera_selector: str | None) -> None:
        super().__init__()
        self._detector = BoardDetector()
        self._diagnostic_store = DiagnosticStore(
            _PROJECT_ROOT / "diagnostics",
            max_captures=100,
        )
        self._runtime_diagnostics = RuntimeDiagnosticLog(
            _PROJECT_ROOT / "diagnostics"
        )
        self._calibration_store = CalibrationStore(
            _PROJECT_ROOT / "calibrations"
        )
        self._reference_store = ReferenceImageStore(
            _PROJECT_ROOT / "calibrations" / "move_reference.npz"
        )
        self._settings = QSettings(
            _SETTINGS_ORGANIZATION,
            _SETTINGS_APPLICATION,
        )
        self._media_devices = QMediaDevices(self)
        self._requested_camera_selector = camera_selector
        self._devices = self._external_camera_devices()
        device_names = [device.description() for device in self._devices]
        device_keys = [self._device_key(device) for device in self._devices]
        saved_key = str(self._settings.value(_CAMERA_SETTING, ""))
        selected_index = (
            _resolve_camera_index(
                device_names,
                device_keys,
                saved_key,
                camera_selector,
            )
            if self._devices
            else None
        )

        self._camera: QCamera | None = None
        self._active_device_index = -1
        self._projector_window = ProjectorWindow()
        self._calibrator = ProjectorCalibrationController(
            self._projector_window,
            self._detector,
        )
        self._calibrator.status_changed.connect(
            self._projector_calibration_status
        )
        self._calibrator.progress_changed.connect(
            self._projector_calibration_progress
        )
        self._calibrator.active_changed.connect(
            self._projector_calibration_active
        )
        self._calibrator.completed.connect(self._projector_calibration_completed)
        self._calibrator.failed.connect(self._projector_calibration_failed)
        self._alignment_monitor = AlignmentMonitor(
            self._detector,
            interval_milliseconds=200,
        )
        self._alignment_monitor.status_changed.connect(
            self._alignment_monitor_status
        )
        self._alignment_monitor.calibration_error.connect(
            self._calibration_error
        )
        self._alignment_monitor.mapping_updated.connect(
            self._alignment_mapping_updated
        )
        self._alignment_monitor.mapping_verified.connect(
            self._alignment_mapping_verified
        )
        self._alignment_monitor.camera_grid_checked.connect(
            self._camera_grid_checked
        )
        self._occupancy_detector = SquareOccupancyDetector()
        self._automatic_start_occupancy_detector = SquareOccupancyDetector(
            canonical_size=400,
            required_observations=2,
            required_agreement=2,
            settling_observations=3,
        )
        self._board_edge_activity_detector = BoardEdgeActivityDetector()
        self._board_edge_indicator_active = False
        self._board_edge_indicator_active_since: float | None = None
        self._board_edge_indicator_last_stuck_diagnostic_at: float | None = None
        self._projector_window.projection_changed.connect(
            self._reset_board_edge_activity_indicator
        )
        self._lichess_blackout_settling_detector = (
            BlackoutSettlingDetector()
        )
        self._illegal_correction_activity_detector = (
            GuidanceActivityDetector(
                confirmation_observations=1,
                minimum_square_fraction=0.5,
                rolling_baseline_before_activity=True,
                require_halo_activity=True,
            )
        )
        self._promotion_activity_detector = (
            ManipulationSettlingDetector()
        )
        self._reference_move_tracker = ReferenceMoveTracker()
        self._piece_synchronizer = PiecePositionSynchronizer()
        self._legal_game = LegalGame()
        self._game_mode: GameMode = "diagnostic"
        self._offline_turn_inference_pending = False
        self._pending_offline_games: tuple[LegalGame, LegalGame] | None = None
        self._pending_offline_restore_game: LegalGame | None = None
        self._offline_mode_recovery_error: str | None = None
        self._startup_offline_reference_refresh_pending = False
        self._offline_restore_requested = False
        self._activate_offline_after_reset = False
        self._automatic_start_reset_recovery = False
        self._automatic_start_reset = _stored_boolean(
            self._settings.value(_AUTOMATIC_START_RESET_SETTING, True),
            True,
        )
        self._project_turn_indicator = _stored_boolean(
            self._settings.value(_PROJECT_TURN_INDICATOR_SETTING, True),
            True,
        )
        stored_promotion_mode = str(
            self._settings.value(_PROMOTION_MODE_SETTING, "auto-queen")
        )
        self._promotion_mode = (
            stored_promotion_mode
            if stored_promotion_mode
            in ("auto-queen", "select-promoted-piece")
            else "auto-queen"
        )
        self._pending_promotion_selection: (
            _PendingPromotionSelection | None
        ) = None
        self._lichess_thread: QThread | None = None
        self._lichess_worker: LichessConnectionWorker | None = None
        self._lichess_client: LichessReadOnlyClient | None = None
        self._lichess_submission_thread: QThread | None = None
        self._lichess_submission_worker: (
            LichessMoveSubmissionWorker | None
        ) = None
        self._lichess_game_action_thread: QThread | None = None
        self._lichess_game_action_worker: LichessGameActionWorker | None = None
        self._lichess_ai_creation_thread: QThread | None = None
        self._lichess_ai_creation_worker: (
            LichessAiGameCreationWorker | None
        ) = None
        self._pending_created_lichess_game_id: str | None = None
        self._pending_lichess_game_action: LichessGameAction | None = None
        self._draw_offer_game_id: str | None = None
        self._responded_draw_offer_game_id: str | None = None
        self._lichess_draw_offer_state_available = False
        self._abort_requested_game_id: str | None = None
        self._resign_requested_game_id: str | None = None
        self._pending_lichess_local_move: (
            tuple[tuple[int, int], tuple[int, int]] | None
        ) = None
        self._pending_lichess_local_uci: str | None = None
        self._pending_lichess_local_result: LegalMoveResult | None = None
        self._lichess_disconnect_requested = False
        self._close_pending = False
        self._lichess_game_id: str | None = None
        self._preferred_lichess_game_id = str(
            self._settings.value(_LICHESS_GAME_SETTING, "")
        )
        self._last_lichess_game_prompt_ids: frozenset[str] | None = None
        self._lichess_player_color: PlayerColor | None = None
        self._lichess_opponent_name = ""
        self._lichess_game_controller: LichessGameController | None = None
        self._last_projected_lichess_uci: str | None = None
        self._lichess_guidance_projection_active = False
        self._lichess_guidance_blackout_pending = False
        self._lichess_guidance_reprojection_pending = False
        self._lichess_guidance_squares: (
            tuple[tuple[int, int], ...]
        ) = ()
        self._lichess_guidance_edge_entered = False
        self._lichess_guidance_activity_logged = False
        self._offline_move_marking_active = False
        self._offline_move_marking_blackout_pending = False
        self._offline_move_marking_generation = 0
        self._illegal_correction_projection_active = False
        self._illegal_correction_blackout_pending = False
        self._illegal_correction_generation = 0
        self._illegal_correction_physical_squares: (
            tuple[tuple[int, int], ...]
        ) = ()
        self._illegal_correction_activity_logged = False
        self._illegal_correction_last_projected_move: (
            tuple[tuple[int, int], tuple[int, int]] | None
        ) = None
        self._lichess_guidance_generation = 0
        self._latest_lichess_state: LichessGameState | None = None
        self._latest_lichess_game: LichessGameFull | None = None
        self._lichess_clocks_available = False
        self._lichess_clock_snapshot_monotonic: float | None = None
        self._last_runtime_tracker_state: str | None = None
        self._runtime_diagnostics_error: str | None = None
        self._latest_stable_occupied_ui: (
            frozenset[tuple[int, int]] | None
        ) = None
        self._reset_to_start_requested = False
        self._reconciliation_requested = False
        self._reconciliation_blackout_pending = False
        self._reconciliation_reference_acceptance_pending = False
        self._reconciliation_blackout_generation = 0
        self._last_rejected_move: (
            tuple[tuple[int, int], tuple[int, int]] | None
        ) = None
        self._camera_only_calibration = False
        self._camera_geometry_ready = False
        self._calibration_result: CalibrationResult | None = None
        self._verified_calibration_result: CalibrationResult | None = None
        self._pending_alignment_mapping: (
            tuple[MappingResult, float] | None
        ) = None
        self._pending_alignment_mapping_generation = 0
        self._pending_alignment_mapping_verified = False
        self._remembered_board_square: tuple[int, int] | None = None
        self._selected_board_square: tuple[int, int] | None = None
        self._pending_board_square: tuple[int, int] | None = None
        self._latest_frame: ImageArray | None = None
        self._snapshot_visible = False
        self._current_pixmap: QPixmap | None = None

        self._capture_session = QMediaCaptureSession(self)
        self._video_sink = QVideoSink(self)
        self._video_sink.videoFrameChanged.connect(self._receive_video_frame)
        self._capture_session.setVideoSink(self._video_sink)

        self._preview = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self._preview.setMinimumSize(480, 240)
        self._preview.setStyleSheet("background: #202020;")
        self._board_widget = ChessboardWidget()
        self._board_widget.setMinimumSize(240, 240)
        self._lichess_board_widget = ChessboardWidget()
        self._lichess_board_widget.setMinimumSize(280, 280)
        self._board_widget.copy_fen_requested.connect(
            self._copy_physical_board_fen
        )
        self._board_widget.copy_board_requested.connect(
            self._copy_physical_board_image
        )
        self._lichess_board_widget.copy_fen_requested.connect(
            self._copy_lichess_board_fen
        )
        self._lichess_board_widget.copy_board_requested.connect(
            self._copy_lichess_board_image
        )
        self._current_lichess_game_action = QAction(
            "Current Lichess game...",
            self._lichess_board_widget,
        )
        self._current_lichess_game_action.setEnabled(False)
        self._current_lichess_game_action.triggered.connect(
            self._show_current_lichess_game
        )
        self._lichess_board_widget.addAction(
            self._current_lichess_game_action
        )
        self._restore_piece_position()
        self._restore_game_mode()
        self._board_widget.setEnabled(False)
        self._board_widget.square_clicked.connect(self._board_square_clicked)
        self._board_quarter_turns = int(
            self._settings.value(_BOARD_ORIENTATION_SETTING, 0)
        ) % 4
        self._board_orientation_needs_detection = (
            not self._piece_synchronizer.initialized
        )
        self._piece_side_orientation_needs_detection = (
            not self._piece_synchronizer.initialized
        )

        self._camera_selector = QComboBox()
        self._camera_selector.setMaximumWidth(260)
        self._camera_selector.addItems(device_names)
        if selected_index is not None:
            self._camera_selector.setCurrentIndex(selected_index)
        else:
            self._camera_selector.setCurrentIndex(-1)
        self._camera_selector.setEnabled(bool(self._devices))
        self._game_mode_selector = QComboBox()
        self._game_mode_selector.setMaximumWidth(260)
        self._game_mode_selector.addItem(
            "Board setup / diagnostic",
            "diagnostic",
        )
        self._game_mode_selector.addItem("Offline legal game", "offline-game")
        self._game_mode_selector.setCurrentIndex(
            1 if self._game_mode == "offline-game" else 0
        )
        self._game_mode_selector.currentIndexChanged.connect(
            self._game_mode_changed
        )
        self._turn_indicator = QLabel()
        self._turn_indicator.setFixedSize(22, 22)
        self._turn_text = QLabel()

        camera_layout = QHBoxLayout()
        camera_layout.addWidget(QLabel("Turn:"))
        camera_layout.addWidget(self._turn_indicator)
        camera_layout.addWidget(self._turn_text)
        camera_layout.addStretch(1)
        self._update_turn_indicator()

        self._camera_status = QLabel()
        self._camera_status.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self._detection_status = QLabel(
            "Live preview. Take a snapshot to detect the board."
        )
        self._detection_status.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self._instructions = QLabel(
            "Keep the entire board visible. Move one piece at a time, then "
            "remove your hand. Press Esc or Q to exit."
        )

        self._snapshot_button = QAction("Take snapshot", self)
        self._snapshot_button.triggered.connect(self._take_snapshot)
        self._resume_button = QAction("Resume live camera", self)
        self._resume_button.triggered.connect(self._resume_live_camera)
        self._resume_button.setEnabled(False)
        self._save_diagnostics = QCheckBox("Save diagnostic snapshots")
        self._save_diagnostics.setChecked(
            _stored_boolean(
                self._settings.value(_SAVE_DIAGNOSTICS_SETTING, True),
                True,
            )
        )
        self._save_diagnostics.toggled.connect(
            self._save_diagnostics_preference_changed
        )
        self._reset_pieces_button = QAction(
            "Reset pieces to starting position",
            self,
        )
        self._reset_pieces_button.triggered.connect(
            self._reset_pieces_to_start
        )
        self._reset_pieces_button.setEnabled(False)
        self._reset_pieces_button.setVisible(
            not self._automatic_start_reset
        )
        self._reconcile_button = QAction("Reconcile physical board", self)
        self._reconcile_button.triggered.connect(
            self._toggle_physical_reconciliation
        )
        self._reconcile_button.setEnabled(False)
        self._lichess_connection_button = QPushButton(
            "Connect",
            self,
        )
        self._lichess_connection_button.clicked.connect(
            self._toggle_lichess_connection
        )
        self._lichess_takeback_button = QToolButton(self)
        _configure_lichess_control_button(
            self._lichess_takeback_button,
            "↶",
            "Request takeback",
            "Request takeback (not implemented)",
            enabled=False,
        )
        self._lichess_draw_button = QToolButton(self)
        _configure_lichess_control_button(
            self._lichess_draw_button,
            "½",
            "Offer draw",
            "Offer draw",
            enabled=False,
        )
        self._lichess_draw_button.clicked.connect(self._offer_lichess_draw)
        self._lichess_accept_draw_button = QToolButton(self)
        _configure_lichess_control_button(
            self._lichess_accept_draw_button,
            "✓",
            "Accept draw offer",
            "Accept draw offer",
            enabled=False,
        )
        self._lichess_accept_draw_button.setVisible(False)
        self._lichess_accept_draw_button.clicked.connect(
            self._accept_lichess_draw
        )
        self._lichess_decline_draw_button = QToolButton(self)
        _configure_lichess_control_button(
            self._lichess_decline_draw_button,
            "✕",
            "Decline draw offer",
            "Decline draw offer",
            enabled=False,
        )
        self._lichess_decline_draw_button.setVisible(False)
        self._lichess_decline_draw_button.clicked.connect(
            self._decline_lichess_draw
        )
        self._lichess_resign_button = QToolButton(self)
        _configure_lichess_control_button(
            self._lichess_resign_button,
            "⚑",
            "Resign game",
            "Resign game",
            enabled=False,
        )
        self._lichess_resign_button.clicked.connect(
            self._confirm_lichess_resignation
        )
        self._lichess_abort_button = QToolButton(self)
        _configure_lichess_control_button(
            self._lichess_abort_button,
            "×",
            "Abort game",
            "Abort game",
            enabled=False,
        )
        self._lichess_abort_button.clicked.connect(
            self._confirm_lichess_abort
        )
        self._start_new_lichess_game_button = QToolButton(self)
        _configure_lichess_control_button(
            self._start_new_lichess_game_button,
            "+",
            "Start new Lichess game",
            "Start new Lichess game",
            enabled=True,
        )
        self._start_new_lichess_game_button.clicked.connect(
            self._start_new_lichess_game
        )
        self._lichess_game_selector = QComboBox()
        self._lichess_game_selector.setMaximumWidth(520)
        self._lichess_game_selector.addItem(
            "Waiting for ongoing games...",
            None,
        )
        self._lichess_game_selector.setEnabled(False)
        self._lichess_game_selector.currentIndexChanged.connect(
            self._lichess_game_selection_changed
        )
        self._screens = list(QGuiApplication.screens())
        self._projector_selector = QComboBox()
        self._projector_selector.setMaximumWidth(360)
        primary_screen = QGuiApplication.primaryScreen()
        for screen in self._screens:
            self._projector_selector.addItem(
                self._projector_screen_label(screen, primary_screen)
            )
        saved_screen_name = str(
            self._settings.value(_PROJECTOR_SETTING, "")
        )
        projector_index = self._default_projector_index(
            self._screens,
            saved_screen_name,
        )
        self._projector_selector.setCurrentIndex(projector_index)

        self._open_projector_button = QAction(
            "Open projector black",
            self,
        )
        self._open_projector_button.triggered.connect(self._open_projector)
        self._focus_pattern_button = QAction("Show focus pattern", self)
        self._focus_pattern_button.triggered.connect(
            self._show_focus_pattern
        )
        self._black_projector_button = QAction("Black projector", self)
        self._black_projector_button.triggered.connect(
            self._black_projector
        )
        self._close_projector_button = QAction("Close projector", self)
        self._close_projector_button.triggered.connect(
            self._close_projector
        )

        self._start_calibration_button = QAction(
            "Full calibration - empty board",
            self,
        )
        self._start_calibration_button.setToolTip(
            "Remove all pieces before running the accepted Gray-code "
            "calibration."
        )
        self._start_calibration_button.triggered.connect(
            self._start_projector_calibration
        )
        self._coded_calibration_button = QAction(
            "Populated calibration - experimental",
            self,
        )
        self._coded_calibration_button.setToolTip(
            "Experimental hardware path for recalibration without removing "
            "the pieces. A failure retains the previous verified calibration."
        )
        self._coded_calibration_button.triggered.connect(
            self._start_coded_projector_calibration
        )
        self._quick_verify_button = QAction(
            "Quick realign / assess",
            self,
        )
        self._quick_verify_button.triggered.connect(
            self._start_quick_verification
        )
        self._quick_verify_button.setEnabled(False)
        self._validate_calibration_button = QAction(
            "Validate calibration",
            self,
        )
        self._validate_calibration_button.triggered.connect(
            self._start_validation_only
        )
        self._validate_calibration_button.setEnabled(False)
        self._cancel_calibration_button = QAction(
            "Cancel calibration",
            self,
        )
        self._cancel_calibration_button.triggered.connect(
            self._cancel_projector_calibration
        )
        self._cancel_calibration_button.setEnabled(False)
        self._show_grid_button = QAction("Project 8x8 grid", self)
        self._show_grid_button.triggered.connect(self._show_calibrated_grid)
        self._show_grid_button.setEnabled(False)
        self._show_squares_button = QAction(
            "Project all 64 squares",
            self,
        )
        self._show_squares_button.triggered.connect(
            self._show_calibrated_squares
        )
        self._show_squares_button.setEnabled(False)
        self._create_application_menus()

        self._calibration_progress = QProgressBar()
        self._calibration_progress.setRange(0, 1)
        self._calibration_progress.setValue(0)
        self._calibration_progress.setVisible(False)

        self._camera_calibration_indicator = QLabel()
        self._projector_calibration_indicator = QLabel()
        self._set_calibration_indicator(
            self._camera_calibration_indicator,
            "Camera calibration",
            "unknown",
        )
        self._set_calibration_indicator(
            self._projector_calibration_indicator,
            "Projector calibration",
            "unknown",
        )
        calibration_indicator_layout = QHBoxLayout()
        calibration_indicator_layout.addWidget(
            self._camera_calibration_indicator
        )
        calibration_indicator_layout.addWidget(
            self._projector_calibration_indicator
        )
        calibration_indicator_layout.addStretch()

        self._projector_status = QLabel(
            "Projector closed. Open it black before showing the square."
        )
        self._projector_status.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self._alignment_status = QLabel("Alignment monitor inactive.")
        self._alignment_status.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self._occupancy_status = QLabel(
            "Occupancy unavailable: projector/camera calibration is not verified."
        )
        self._occupancy_status.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self._piece_status = QLabel(
            "Piece position restored from the previous session."
            if self._piece_synchronizer.initialized
            else "Piece position is not synchronized."
        )
        self._piece_status.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self._lichess_status = QLabel(
            f"Lichess disconnected. "
            f"{_lichess_token_configuration_message()}"
        )
        self._lichess_status.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self._lichess_sync_status = QLabel(
            "Lichess physical synchronization is inactive."
        )
        self._lichess_sync_status.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        for status_label in (
            self._camera_status,
            self._detection_status,
            self._projector_status,
            self._alignment_status,
            self._occupancy_status,
            self._piece_status,
            self._lichess_status,
            self._lichess_sync_status,
            self._instructions,
        ):
            status_label.setWordWrap(True)

        camera_panel = QGroupBox("Camera")
        camera_panel_layout = QVBoxLayout()
        camera_panel_layout.addWidget(self._preview)
        camera_panel.setLayout(camera_panel_layout)

        physical_panel = QGroupBox("Physical position")
        physical_panel_layout = QVBoxLayout()
        physical_panel_layout.addWidget(self._board_widget)
        physical_panel.setLayout(physical_panel_layout)

        self._lichess_black_player = QLabel("Black: --")
        self._lichess_black_clock = QLabel("Black clock: --:--")
        self._lichess_white_player = QLabel("White: --")
        self._lichess_white_clock = QLabel("White clock: --:--")
        self._lichess_game_information_status = QLabel(
            "No active Lichess game information."
        )
        for label in (
            self._lichess_black_player,
            self._lichess_black_clock,
            self._lichess_white_player,
            self._lichess_white_clock,
            self._lichess_game_information_status,
        ):
            label.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
            )
        self._lichess_game_information_status.setWordWrap(True)

        black_information_layout = QHBoxLayout()
        black_information_layout.addWidget(self._lichess_black_player)
        black_information_layout.addStretch(1)
        black_information_layout.addWidget(
            self._lichess_connection_button
        )
        black_information_layout.addStretch(1)
        black_information_layout.addWidget(self._lichess_black_clock)
        white_information_layout = QHBoxLayout()
        white_information_layout.addWidget(self._lichess_white_player)
        white_information_layout.addStretch(1)
        white_information_layout.addWidget(self._lichess_white_clock)
        lichess_controls_layout = QHBoxLayout()
        lichess_controls_layout.addStretch(1)
        lichess_controls_layout.addWidget(self._lichess_takeback_button)
        lichess_controls_layout.addWidget(self._lichess_draw_button)
        lichess_controls_layout.addWidget(
            self._lichess_accept_draw_button
        )
        lichess_controls_layout.addWidget(
            self._lichess_decline_draw_button
        )
        lichess_controls_layout.addWidget(self._lichess_abort_button)
        lichess_controls_layout.addWidget(self._lichess_resign_button)
        lichess_controls_layout.addWidget(
            self._start_new_lichess_game_button
        )
        lichess_controls_layout.addStretch(1)

        lichess_panel = QGroupBox("Lichess position")
        _set_readable_interaction_font(lichess_panel)
        lichess_panel_layout = QVBoxLayout()
        lichess_panel_layout.addLayout(black_information_layout)
        lichess_panel_layout.addWidget(self._lichess_board_widget)
        lichess_panel_layout.addLayout(white_information_layout)
        lichess_panel_layout.addWidget(
            self._lichess_game_information_status
        )
        lichess_panel_layout.addLayout(lichess_controls_layout)
        lichess_panel.setLayout(lichess_panel_layout)

        interaction_content = QWidget()
        _set_readable_interaction_font(interaction_content)
        interaction_layout = QVBoxLayout()
        interaction_layout.addLayout(camera_layout)
        interaction_layout.addWidget(self._calibration_progress)
        interaction_layout.addLayout(calibration_indicator_layout)
        interaction_layout.addWidget(self._camera_status)
        interaction_layout.addWidget(self._detection_status)
        interaction_layout.addWidget(self._projector_status)
        interaction_layout.addWidget(self._alignment_status)
        interaction_layout.addWidget(self._occupancy_status)
        interaction_layout.addWidget(self._piece_status)
        interaction_layout.addWidget(self._lichess_status)
        interaction_layout.addWidget(self._lichess_sync_status)
        interaction_layout.addWidget(self._instructions)
        interaction_layout.addStretch(1)
        interaction_content.setLayout(interaction_layout)

        interaction_scroll = QScrollArea()
        interaction_scroll.setWidgetResizable(True)
        interaction_scroll.setWidget(interaction_content)
        interaction_panel = QGroupBox("Interaction")
        interaction_panel_layout = QVBoxLayout()
        interaction_panel_layout.addWidget(interaction_scroll)
        interaction_panel.setLayout(interaction_panel_layout)

        quadrant_layout = QGridLayout()
        quadrant_layout.addWidget(camera_panel, 0, 0)
        quadrant_layout.addWidget(physical_panel, 0, 1)
        quadrant_layout.addWidget(interaction_panel, 1, 0)
        quadrant_layout.addWidget(lichess_panel, 1, 1)
        quadrant_layout.setColumnStretch(0, 3)
        quadrant_layout.setColumnStretch(1, 2)
        quadrant_layout.setRowStretch(0, 2)
        quadrant_layout.setRowStretch(1, 3)

        layout = QVBoxLayout()
        layout.addLayout(quadrant_layout)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.setWindowTitle("Chessboard Camera Detection")
        self.resize(1500, 900)
        _restore_window_placement(self, self._settings)
        self._projector_selector.currentIndexChanged.connect(
            self._projector_selection_changed
        )
        application = QGuiApplication.instance()
        if isinstance(application, QGuiApplication):
            application.screenAdded.connect(
                self._available_screens_changed
            )
            application.screenRemoved.connect(
                self._available_screens_changed
            )
        self._camera_selector.currentIndexChanged.connect(
            self._camera_selection_changed
        )
        self._media_devices.videoInputsChanged.connect(
            self._camera_devices_changed
        )
        self._occupancy_timer = QTimer(self)
        self._occupancy_timer.setInterval(100)
        self._occupancy_timer.timeout.connect(self._update_occupancy)
        self._occupancy_timer.start()
        self._lichess_clock_timer = QTimer(self)
        self._lichess_clock_timer.setInterval(250)
        self._lichess_clock_timer.timeout.connect(
            self._refresh_lichess_clock_display
        )
        self._lichess_clock_timer.start()
        QTimer.singleShot(0, self._black_projector)
        if selected_index is None:
            self._wait_for_external_camera()
        else:
            self._start_camera(selected_index)
        self._record_runtime_diagnostic(
            "application-started",
            "Chessboard camera application started.",
        )
        if (
            self._pending_offline_restore_game is not None
            or self._offline_mode_recovery_error is not None
        ) and not self._automatic_start_reset_recovery:
            QTimer.singleShot(0, self._show_offline_mode_recovery_dialog)

    def _create_application_menus(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        self._settings_action = file_menu.addAction("Settings...")
        self._settings_action.triggered.connect(self._show_settings_dialog)
        file_menu.addSeparator()
        exit_action = file_menu.addAction("Exit")
        exit_action.triggered.connect(self.close)

        camera_menu = self.menuBar().addMenu("&Camera")
        camera_menu.addAction(self._snapshot_button)
        camera_menu.addAction(self._resume_button)
        camera_menu.addSeparator()
        camera_menu.addAction(self._reset_pieces_button)
        camera_menu.addAction(self._reconcile_button)

        game_menu = self.menuBar().addMenu("&Game")
        mode_menu = game_menu.addMenu("Mode")
        self._game_mode_action_group = QActionGroup(self)
        self._game_mode_action_group.setExclusive(True)
        self._diagnostic_mode_action = mode_menu.addAction(
            "Board setup / diagnostic"
        )
        self._diagnostic_mode_action.setCheckable(True)
        self._diagnostic_mode_action.triggered.connect(
            lambda: self._game_mode_selector.setCurrentIndex(0)
        )
        self._game_mode_action_group.addAction(
            self._diagnostic_mode_action
        )
        self._offline_mode_action = mode_menu.addAction("Offline legal game")
        self._offline_mode_action.setCheckable(True)
        self._offline_mode_action.triggered.connect(
            lambda: self._game_mode_selector.setCurrentIndex(1)
        )
        self._game_mode_action_group.addAction(self._offline_mode_action)
        self._sync_game_mode_actions()

        lichess_menu = game_menu.addMenu("Lichess")
        self._lichess_games_menu = lichess_menu.addMenu("Active game")
        self._lichess_game_action_group = QActionGroup(self)
        self._lichess_game_action_group.setExclusive(True)
        self._rebuild_lichess_games_menu()

        projector_menu = self.menuBar().addMenu("&Projector")
        projector_menu.addAction(self._open_projector_button)
        projector_menu.addAction(self._focus_pattern_button)
        projector_menu.addAction(self._black_projector_button)
        projector_menu.addAction(self._close_projector_button)
        projector_menu.addSeparator()
        patterns_menu = projector_menu.addMenu("Test patterns")
        patterns_menu.addAction(self._show_grid_button)
        patterns_menu.addAction(self._show_squares_button)

        calibration_menu = self.menuBar().addMenu("&Calibration")
        calibration_menu.addAction(self._start_calibration_button)
        calibration_menu.addAction(self._coded_calibration_button)
        calibration_menu.addAction(self._quick_verify_button)
        calibration_menu.addAction(self._validate_calibration_button)
        calibration_menu.addSeparator()
        calibration_menu.addAction(self._cancel_calibration_button)

    def _sync_game_mode_actions(self) -> None:
        diagnostic = self._game_mode_selector.currentData() == "diagnostic"
        self._diagnostic_mode_action.setChecked(diagnostic)
        self._offline_mode_action.setChecked(not diagnostic)

    def _rebuild_lichess_games_menu(self) -> None:
        self._lichess_games_menu.clear()
        self._lichess_game_action_group = QActionGroup(self)
        self._lichess_game_action_group.setExclusive(True)
        current_game_id = self._lichess_game_selector.currentData()
        selectable_games = 0
        for index in range(self._lichess_game_selector.count()):
            label = self._lichess_game_selector.itemText(index)
            game_id = self._lichess_game_selector.itemData(index)
            action = self._lichess_games_menu.addAction(label)
            if not isinstance(game_id, str):
                action.setEnabled(False)
                continue
            selectable_games += 1
            action.setCheckable(True)
            action.setChecked(game_id == current_game_id)
            action.triggered.connect(
                lambda _checked=False, selected=game_id: (
                    self._select_lichess_game_from_menu(selected)
                )
            )
            self._lichess_game_action_group.addAction(action)
        self._lichess_games_menu.setEnabled(selectable_games > 0)

    def _select_lichess_game_from_menu(self, game_id: str) -> None:
        index = self._lichess_game_selector.findData(game_id)
        if index >= 0:
            self._lichess_game_selector.setCurrentIndex(index)
        self._rebuild_lichess_games_menu()

    def _show_settings_dialog(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Settings")
        dialog.setModal(True)
        tabs = QTabWidget(dialog)

        devices_tab = QWidget()
        devices_layout = QFormLayout()
        camera_selector = QComboBox()
        camera_options, camera_index = _persistent_choice_options(
            tuple(
                (
                    device.description(),
                    self._device_key(device),
                )
                for device in self._devices
            ),
            str(self._settings.value(_CAMERA_SETTING, "")),
            str(self._settings.value(_CAMERA_NAME_SETTING, "")),
        )
        for label, identity in camera_options:
            camera_selector.addItem(label, identity)
        camera_selector.setCurrentIndex(camera_index)

        projector_selector = QComboBox()

        def refresh_projector_selector(
            _screen: QScreen | None = None,
        ) -> None:
            selected_identity = projector_selector.currentData()
            selected_label = projector_selector.currentText()
            self._refresh_projector_screens(
                list(QGuiApplication.screens())
            )
            remembered_id = (
                selected_identity
                if isinstance(selected_identity, str)
                else str(self._settings.value(_PROJECTOR_SETTING, ""))
            )
            remembered_label = (
                selected_label.removesuffix(" (unavailable)")
                if isinstance(selected_identity, str)
                else str(
                    self._settings.value(_PROJECTOR_LABEL_SETTING, "")
                )
            )
            primary_screen = QGuiApplication.primaryScreen()
            projector_options, projector_index = (
                _persistent_choice_options(
                    tuple(
                        (
                            CameraWindow._projector_screen_label(
                                screen,
                                primary_screen,
                            ),
                            screen.name(),
                        )
                        for screen in self._screens
                    ),
                    remembered_id,
                    remembered_label,
                )
            )
            projector_selector.blockSignals(True)
            projector_selector.clear()
            for label, identity in projector_options:
                projector_selector.addItem(label, identity)
            projector_selector.setCurrentIndex(projector_index)
            projector_selector.blockSignals(False)

        refresh_projector_selector()
        devices_layout.addRow("Camera:", camera_selector)
        devices_layout.addRow("Projector display:", projector_selector)
        devices_tab.setLayout(devices_layout)
        tabs.addTab(devices_tab, "Devices")

        diagnostics_tab = QWidget()
        diagnostics_layout = QVBoxLayout()
        save_diagnostics = QCheckBox("Save diagnostic snapshots")
        save_diagnostics.setChecked(self._save_diagnostics.isChecked())
        diagnostics_layout.addWidget(save_diagnostics)
        diagnostics_layout.addStretch(1)
        diagnostics_tab.setLayout(diagnostics_layout)
        tabs.addTab(diagnostics_tab, "Diagnostics")

        chess_tab = QWidget()
        chess_layout = QFormLayout()
        promotion_mode = QComboBox()
        promotion_mode.addItem("Auto-queen", "auto-queen")
        promotion_mode.addItem(
            "Select promoted piece",
            "select-promoted-piece",
        )
        promotion_index = promotion_mode.findData(self._promotion_mode)
        promotion_mode.setCurrentIndex(max(0, promotion_index))
        chess_layout.addRow("Promotion:", promotion_mode)
        automatic_start_reset = QCheckBox(
            "Automatically reset a detected starting position"
        )
        automatic_start_reset.setChecked(self._automatic_start_reset)
        chess_layout.addRow(automatic_start_reset)
        project_turn_indicator = QCheckBox(
            "Project turn indicator beside the board"
        )
        project_turn_indicator.setChecked(self._project_turn_indicator)
        chess_layout.addRow(project_turn_indicator)
        chess_tab.setLayout(chess_layout)
        tabs.addTab(chess_tab, "Chess")

        lichess_tab = QWidget()
        lichess_layout = QVBoxLayout()
        lichess_layout.addWidget(
            QLabel(
                f"{_lichess_token_configuration_message()} Authentication "
                "continues to use the environment value. The token is not "
                "displayed or stored in application settings."
            )
        )
        ai_defaults = QGroupBox("New AI game defaults")
        ai_defaults_layout = QFormLayout()
        ai_level = QSpinBox()
        ai_level.setRange(1, 8)
        ai_level.setValue(
            min(
                8,
                max(
                    1,
                    _stored_integer(
                        self._settings.value(
                            _LICHESS_AI_LEVEL_SETTING,
                            4,
                        ),
                        4,
                    ),
                ),
            )
        )
        ai_color = QComboBox()
        for label, value in (
            ("White", "white"),
            ("Black", "black"),
            ("Random", "random"),
        ):
            ai_color.addItem(label, value)
        stored_ai_color = str(
            self._settings.value(_LICHESS_AI_COLOR_SETTING, "white")
        )
        ai_color_index = ai_color.findData(stored_ai_color)
        ai_color.setCurrentIndex(max(0, ai_color_index))
        ai_clock = QComboBox()
        for label, key, _limit, _increment in LichessAiGameDialog._CLOCKS:
            ai_clock.addItem(label, key)
        stored_ai_clock = str(
            self._settings.value(_LICHESS_AI_CLOCK_SETTING, "10+5")
        )
        ai_clock_index = ai_clock.findData(stored_ai_clock)
        ai_clock.setCurrentIndex(
            ai_clock_index if ai_clock_index >= 0 else 2
        )
        ai_defaults_layout.addRow("AI level:", ai_level)
        ai_defaults_layout.addRow("Your color:", ai_color)
        ai_defaults_layout.addRow("Clock:", ai_clock)
        ai_defaults.setLayout(ai_defaults_layout)
        lichess_layout.addWidget(ai_defaults)
        lichess_layout.addStretch(1)
        lichess_tab.setLayout(lichess_layout)
        tabs.addTab(lichess_tab, "Lichess")

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.Apply
        )

        def apply_values() -> None:
            selected_camera = camera_selector.currentData()
            camera_keys = [
                self._device_key(device) for device in self._devices
            ]
            if (
                isinstance(selected_camera, str)
                and selected_camera in camera_keys
            ):
                self._camera_selector.setCurrentIndex(
                    camera_keys.index(selected_camera)
                )
            selected_projector = projector_selector.currentData()
            projector_names = [screen.name() for screen in self._screens]
            if (
                isinstance(selected_projector, str)
                and selected_projector in projector_names
            ):
                self._projector_selector.setCurrentIndex(
                    projector_names.index(selected_projector)
                )
            self._save_diagnostics.setChecked(
                save_diagnostics.isChecked()
            )
            selected_promotion_mode = promotion_mode.currentData()
            if not isinstance(selected_promotion_mode, str):
                raise RuntimeError("Promotion mode has no stored identity.")
            self._set_promotion_mode(selected_promotion_mode)
            self._set_automatic_start_reset(
                automatic_start_reset.isChecked()
            )
            self._set_project_turn_indicator(
                project_turn_indicator.isChecked()
            )
            selected_ai_color = ai_color.currentData()
            selected_ai_clock = ai_clock.currentData()
            if not isinstance(selected_ai_color, str):
                raise RuntimeError("AI color has no stored identity.")
            if not isinstance(selected_ai_clock, str):
                raise RuntimeError("AI clock has no stored identity.")
            self._settings.setValue(
                _LICHESS_AI_LEVEL_SETTING,
                ai_level.value(),
            )
            self._settings.setValue(
                _LICHESS_AI_COLOR_SETTING,
                selected_ai_color,
            )
            self._settings.setValue(
                _LICHESS_AI_CLOCK_SETTING,
                selected_ai_clock,
            )
            self._settings.sync()

        def accept_values() -> None:
            apply_values()
            dialog.accept()

        buttons.accepted.connect(accept_values)
        buttons.rejected.connect(dialog.reject)
        apply_button = buttons.button(
            QDialogButtonBox.StandardButton.Apply
        )
        if apply_button is None:
            raise RuntimeError("Settings dialog has no Apply button.")
        apply_button.clicked.connect(apply_values)

        layout = QVBoxLayout()
        layout.addWidget(tabs)
        layout.addWidget(buttons)
        dialog.setLayout(layout)
        dialog.resize(520, 320)
        application = QGuiApplication.instance()
        if isinstance(application, QGuiApplication):
            application.screenAdded.connect(refresh_projector_selector)
            application.screenRemoved.connect(refresh_projector_selector)
        try:
            dialog.exec()
        finally:
            if isinstance(application, QGuiApplication):
                application.screenAdded.disconnect(
                    refresh_projector_selector
                )
                application.screenRemoved.disconnect(
                    refresh_projector_selector
                )

    def _save_diagnostics_preference_changed(self, checked: bool) -> None:
        self._settings.setValue(_SAVE_DIAGNOSTICS_SETTING, checked)
        self._settings.sync()

    def _set_promotion_mode(self, mode: str) -> None:
        if mode not in ("auto-queen", "select-promoted-piece"):
            raise ValueError(f"Unsupported promotion mode: {mode}")
        self._promotion_mode = mode
        self._settings.setValue(_PROMOTION_MODE_SETTING, mode)
        self._settings.sync()

    def _set_automatic_start_reset(self, enabled: bool) -> None:
        self._automatic_start_reset = enabled
        automatic_detector = getattr(
            self,
            "_automatic_start_occupancy_detector",
            None,
        )
        if automatic_detector is not None:
            automatic_detector.reset()
        reset_action = getattr(self, "_reset_pieces_button", None)
        if reset_action is not None:
            reset_action.setVisible(not enabled)
        self._settings.setValue(_AUTOMATIC_START_RESET_SETTING, enabled)
        self._settings.sync()

    def _set_project_turn_indicator(self, enabled: bool) -> None:
        self._project_turn_indicator = enabled
        self._settings.setValue(_PROJECT_TURN_INDICATOR_SETTING, enabled)
        self._settings.sync()
        if hasattr(self, "_projector_window"):
            self._update_projected_turn_indicator()

    @staticmethod
    def _default_projector_index(
        screens: Sequence[QScreen],
        saved_screen_name: str,
    ) -> int:
        for index, screen in enumerate(screens):
            if screen.name() == saved_screen_name:
                return index
        return -1

    @staticmethod
    def _projector_screen_label(
        screen: QScreen,
        primary_screen: QScreen | None,
    ) -> str:
        geometry = screen.geometry()
        primary_suffix = " (primary)" if screen is primary_screen else ""
        hardware_parts = [
            value
            for value in (screen.manufacturer(), screen.model())
            if value
        ]
        hardware = (
            f" [{' '.join(hardware_parts)}]" if hardware_parts else ""
        )
        return (
            f"{screen.name()}{hardware} - "
            f"{geometry.width()}x{geometry.height()} at "
            f"({geometry.x()}, {geometry.y()}){primary_suffix}"
        )

    def _available_screens_changed(self, _screen: QScreen) -> None:
        self._refresh_projector_screens(
            list(QGuiApplication.screens())
        )

    def _refresh_projector_screens(
        self,
        screens: Sequence[QScreen],
    ) -> None:
        primary_screen = QGuiApplication.primaryScreen()
        saved_screen_name = str(
            self._settings.value(_PROJECTOR_SETTING, "")
        )
        selected_index = CameraWindow._default_projector_index(
            screens,
            saved_screen_name,
        )
        self._projector_selector.blockSignals(True)
        self._projector_selector.clear()
        for screen in screens:
            self._projector_selector.addItem(
                CameraWindow._projector_screen_label(
                    screen,
                    primary_screen,
                )
            )
        self._screens = list(screens)
        self._projector_selector.setCurrentIndex(selected_index)
        self._projector_selector.setEnabled(bool(screens))
        self._projector_selector.blockSignals(False)
        self._restore_saved_calibration()

    @staticmethod
    def _device_key(device: QCameraDevice) -> str:
        return bytes(device.id().toHex().data()).decode("ascii")

    @staticmethod
    def _external_camera_devices() -> list[QCameraDevice]:
        return [
            device
            for device in QMediaDevices.videoInputs()
            if _is_external_camera(
                device.description(),
                device.position(),
            )
        ]

    @staticmethod
    def _preferred_format(device: QCameraDevice) -> QCameraFormat | None:
        formats = list(device.videoFormats())
        if not formats:
            return None

        target_area = 1920 * 1080
        return min(
            formats,
            key=lambda camera_format: (
                abs(
                    camera_format.resolution().width()
                    * camera_format.resolution().height()
                    - target_area
                ),
                -camera_format.maxFrameRate(),
            ),
        )

    def _start_camera(self, device_index: int) -> None:
        if not 0 <= device_index < len(self._devices):
            raise RuntimeError(f"Camera index {device_index} is not available.")

        if self._camera is not None:
            self._camera.stop()

        device = self._devices[device_index]
        camera = QCamera(device, self)
        preferred_format = self._preferred_format(device)
        if preferred_format is not None:
            camera.setCameraFormat(preferred_format)
        camera.errorOccurred.connect(self._camera_error)
        self._capture_session.setCamera(camera)
        camera.start()
        self._camera = camera
        self._active_device_index = device_index

        self._latest_frame = None
        self._snapshot_visible = False
        self._snapshot_button.setEnabled(True)
        self._start_calibration_button.setEnabled(True)
        self._coded_calibration_button.setEnabled(True)
        self._resume_button.setEnabled(False)
        self._settings.setValue(_CAMERA_SETTING, self._device_key(device))
        self._settings.setValue(
            _CAMERA_NAME_SETTING,
            device.description(),
        )
        self._settings.sync()

        active_format = camera.cameraFormat()
        resolution = active_format.resolution()
        self._camera_status.setText(
            f"{device.description()}: {resolution.width()}x{resolution.height()} "
            f"up to {active_format.maxFrameRate():.1f} FPS"
        )
        self._detection_status.setText(
            "Starting live preview. Take a snapshot to detect the board."
        )
        self._restore_saved_calibration()
        self._restore_move_reference()
        self._start_pending_offline_reference_refresh()
        self._record_runtime_diagnostic(
            "camera-started",
            f"Camera started: {device.description()}.",
            details={
                "selected_index": device_index,
                "resolution": [resolution.width(), resolution.height()],
                "maximum_fps": active_format.maxFrameRate(),
            },
        )

    def _camera_selection_changed(self, selected_index: int) -> None:
        previous_index = self._active_device_index
        try:
            self._start_camera(selected_index)
        except RuntimeError as error:
            self._camera_selector.blockSignals(True)
            self._camera_selector.setCurrentIndex(previous_index)
            self._camera_selector.blockSignals(False)
            self._record_runtime_diagnostic(
                "camera-selection-failed",
                str(error),
                details={"requested_index": selected_index},
            )
            QMessageBox.critical(self, "Camera error", str(error))

    def _camera_devices_changed(self) -> None:
        active_key = (
            self._device_key(self._devices[self._active_device_index])
            if 0 <= self._active_device_index < len(self._devices)
            else ""
        )
        devices = self._external_camera_devices()
        device_keys = [self._device_key(device) for device in devices]
        self._devices = devices

        self._camera_selector.blockSignals(True)
        self._camera_selector.clear()
        self._camera_selector.addItems(
            [device.description() for device in devices]
        )
        self._camera_selector.setEnabled(
            bool(devices) and not self._calibrator.active
        )

        if active_key in device_keys:
            self._active_device_index = device_keys.index(active_key)
            self._camera_selector.setCurrentIndex(self._active_device_index)
            self._camera_selector.blockSignals(False)
            return

        if self._camera is not None:
            self._camera.stop()
            self._camera = None
        self._active_device_index = -1
        self._latest_frame = None
        self._current_pixmap = None
        self._preview.clear()
        self._invalidate_calibration()

        if not devices:
            self._camera_selector.blockSignals(False)
            self._wait_for_external_camera()
            return

        saved_key = str(self._settings.value(_CAMERA_SETTING, ""))
        try:
            selected_index = _resolve_camera_index(
                [device.description() for device in devices],
                device_keys,
                saved_key,
                self._requested_camera_selector,
            )
        except RuntimeError as error:
            self._camera_selector.blockSignals(False)
            self._wait_for_external_camera(str(error))
            return

        if selected_index is None:
            self._camera_selector.setCurrentIndex(-1)
            self._camera_selector.blockSignals(False)
            self._wait_for_external_camera(
                "The previously selected camera is unavailable."
            )
            return
        self._camera_selector.setCurrentIndex(selected_index)
        self._camera_selector.blockSignals(False)
        self._start_camera(selected_index)

    def _wait_for_external_camera(self, detail: str | None = None) -> None:
        self._snapshot_button.setEnabled(False)
        self._resume_button.setEnabled(False)
        self._start_calibration_button.setEnabled(False)
        self._coded_calibration_button.setEnabled(False)
        message = "Waiting for an external USB camera."
        if detail is not None:
            message = f"{message} {detail}"
        self._camera_status.setText(message)
        self._detection_status.setText(
            "The integrated webcam is not used."
        )

    def _camera_error(
        self,
        error: QCamera.Error,
        error_string: str,
    ) -> None:
        self._detection_status.setText(
            f"Camera error ({error.name}): {error_string}"
        )
        self._record_runtime_diagnostic(
            "camera-error",
            error_string,
            details={"error": error.name},
            include_images=True,
        )

    def _open_projector(self) -> bool:
        selected_index = self._projector_selector.currentIndex()
        if not 0 <= selected_index < len(self._screens):
            self._record_runtime_diagnostic(
                "projector-open-failed",
                "The selected projector display is no longer available.",
                details={"selected_index": selected_index},
            )
            QMessageBox.critical(
                self,
                "Projector error",
                "The selected projector display is no longer available.",
            )
            return False

        screen = self._screens[selected_index]
        self._projector_window.show_black()
        try:
            self._projector_window.show_on_screen(screen)
        except RuntimeError as error:
            self._record_runtime_diagnostic(
                "projector-open-failed",
                str(error),
                details={"screen": screen.name()},
            )
            QMessageBox.critical(self, "Projector error", str(error))
            return False

        self._alignment_monitor.set_black()
        self._settings.setValue(_PROJECTOR_SETTING, screen.name())
        self._settings.setValue(
            _PROJECTOR_LABEL_SETTING,
            CameraWindow._projector_screen_label(
                screen,
                QGuiApplication.primaryScreen(),
            ),
        )
        self._settings.sync()
        geometry = screen.geometry()
        self._projector_status.setText(
            f"Projector open black on {screen.name()} at "
            f"{geometry.width()}x{geometry.height()}."
        )
        self._record_runtime_diagnostic(
            "projector-opened",
            self._projector_status.text(),
            details={"screen": screen.name()},
        )
        return True

    def _start_projector_calibration(self) -> None:
        if self._latest_frame is None:
            QMessageBox.warning(
                self,
                "Projector calibration",
                "No camera frame is available yet.",
            )
            return
        if not self._open_projector():
            return

        selected_index = self._projector_selector.currentIndex()
        screen_geometry = self._screens[selected_index].geometry()
        previous_calibration_available = self._calibration_result is not None
        self._record_runtime_diagnostic(
            "calibration-start-requested",
            "Full projector calibration requested.",
            details={
                "operation": "full",
                "previous_calibration_available": (
                    previous_calibration_available
                ),
            },
            include_images=True,
        )
        self._suspend_selected_square()
        try:
            self._calibrator.start(
                screen_geometry.width(),
                screen_geometry.height(),
            )
        except (RuntimeError, ValueError) as error:
            self._projector_window.show_black()
            QMessageBox.critical(self, "Projector calibration", str(error))

    def _start_coded_projector_calibration(self) -> None:
        if self._latest_frame is None:
            QMessageBox.warning(
                self,
                "Coded-pattern calibration",
                "No camera frame is available yet.",
            )
            return
        if not self._open_projector():
            return

        self._suspend_selected_square()
        selected_index = self._projector_selector.currentIndex()
        screen_geometry = self._screens[selected_index].geometry()
        self._record_runtime_diagnostic(
            "calibration-start-requested",
            "Coded-pattern calibration requested.",
            details={
                "operation": "coded-pattern",
                "previous_calibration_available": (
                    self._calibration_result is not None
                ),
            },
            include_images=True,
        )
        try:
            self._calibrator.start_coded_pattern(
                screen_geometry.width(),
                screen_geometry.height(),
            )
        except (RuntimeError, ValueError) as error:
            self._projector_window.show_black()
            QMessageBox.critical(
                self,
                "Coded-pattern calibration",
                str(error),
            )

    def _start_quick_verification(self) -> None:
        result = self._calibration_result
        if result is None:
            QMessageBox.warning(
                self,
                "Quick calibration verification",
                "No matching saved calibration is available.",
            )
            return
        if self._latest_frame is None:
            QMessageBox.warning(
                self,
                "Quick calibration verification",
                "No camera frame is available yet.",
            )
            return
        if not self._open_projector():
            return

        self._suspend_selected_square()
        selected_index = self._projector_selector.currentIndex()
        screen_geometry = self._screens[selected_index].geometry()
        self._record_runtime_diagnostic(
            "calibration-start-requested",
            "Quick calibration verification requested.",
            details={"operation": "quick-verification"},
            include_images=True,
        )
        try:
            self._calibrator.start_verification(
                result,
                screen_geometry.width(),
                screen_geometry.height(),
            )
        except (RuntimeError, ValueError) as error:
            self._projector_window.show_black()
            QMessageBox.critical(
                self,
                "Quick calibration verification",
                str(error),
            )

    def _start_validation_only(self) -> None:
        result = self._calibration_result
        if result is None:
            QMessageBox.warning(
                self,
                "Projector calibration validation",
                "No matching saved calibration is available.",
            )
            return
        if self._latest_frame is None:
            QMessageBox.warning(
                self,
                "Projector calibration validation",
                "No camera frame is available yet.",
            )
            return
        if not self._open_projector():
            return

        self._suspend_selected_square()
        selected_index = self._projector_selector.currentIndex()
        screen_geometry = self._screens[selected_index].geometry()
        self._record_runtime_diagnostic(
            "calibration-start-requested",
            "Calibration validation requested.",
            details={"operation": "validation"},
            include_images=True,
        )
        try:
            self._calibrator.start_validation_only(
                result,
                screen_geometry.width(),
                screen_geometry.height(),
            )
        except (RuntimeError, ValueError) as error:
            self._projector_window.show_black()
            QMessageBox.critical(
                self,
                "Projector calibration validation",
                str(error),
            )

    def _black_projector(self) -> None:
        lichess_guidance_active = self._lichess_guidance_projection_active
        self._offline_move_marking_active = False
        self._offline_move_marking_blackout_pending = False
        self._offline_move_marking_generation += 1
        self._illegal_correction_projection_active = False
        self._illegal_correction_blackout_pending = False
        self._illegal_correction_generation += 1
        self._illegal_correction_physical_squares = ()
        self._illegal_correction_activity_logged = False
        self._illegal_correction_last_projected_move = None
        self._illegal_correction_activity_detector.reset()
        if self._calibrator.active:
            self._calibrator.cancel()
        if not self._open_projector():
            return
        self._suspend_selected_square()
        if lichess_guidance_active:
            self._begin_lichess_guidance_observation()
            return
        self._projector_status.setText(
            "Projector is black; the selected board square is remembered."
        )

    def _show_focus_pattern(self) -> None:
        if not self._open_projector():
            return
        selected_index = self._projector_selector.currentIndex()
        geometry = self._screens[selected_index].geometry()
        self._projector_window.show_pattern(
            focus_pattern(geometry.width(), geometry.height())
        )
        self._suspend_selected_square()
        self._alignment_monitor.set_paused("focus pattern")
        self._projector_status.setText(
            "Showing the projector focus pattern. Select Black when finished."
        )

    def _close_projector(self) -> None:
        if self._calibrator.active:
            self._calibrator.cancel()
        self._projector_window.show_black()
        self._projector_window.close()
        self._suspend_selected_square()
        self._alignment_monitor.set_black()
        self._projector_status.setText("Projector closed.")

    def _show_calibrated_grid(self) -> None:
        if not self._open_projector():
            return
        try:
            self._projector_window.show_grid()
        except RuntimeError as error:
            QMessageBox.warning(self, "Projector calibration", str(error))
            return
        self._suspend_selected_square()
        self._alignment_monitor.set_paused("full-board grid preview")
        self._projector_status.setText(
            "Projecting the calibrated 8x8 board grid."
        )

    def _show_calibrated_squares(self) -> None:
        if not self._open_projector():
            return
        try:
            self._projector_window.show_all_squares()
        except RuntimeError as error:
            QMessageBox.warning(self, "Projector calibration", str(error))
            return
        self._suspend_selected_square()
        self._alignment_monitor.set_paused("all-squares preview")
        self._projector_status.setText(
            "Projecting inset red fills over all 64 squares."
        )

    def _projector_calibration_status(self, message: str) -> None:
        self._projector_status.setText(message)

    def _projector_calibration_progress(
        self,
        value: int,
        maximum: int,
    ) -> None:
        self._calibration_progress.setRange(0, maximum)
        self._calibration_progress.setValue(value)

    def _projector_calibration_active(self, active: bool) -> None:
        self._calibration_progress.setVisible(active)
        self._start_calibration_button.setEnabled(not active)
        self._coded_calibration_button.setEnabled(not active)
        self._quick_verify_button.setEnabled(
            not active and self._calibration_result is not None
        )
        self._validate_calibration_button.setEnabled(
            not active and self._calibration_result is not None
        )
        self._cancel_calibration_button.setEnabled(active)
        self._camera_selector.setEnabled(not active)
        self._projector_selector.setEnabled(not active)
        self._open_projector_button.setEnabled(not active)
        self._focus_pattern_button.setEnabled(not active)
        self._close_projector_button.setEnabled(not active)
        self._snapshot_button.setEnabled(not active and not self._snapshot_visible)
        self._board_widget.setEnabled(
            not active and self._calibration_result is not None
        )
        if active:
            self._alignment_monitor.set_paused("projector calibration")

    def _cancel_projector_calibration(self) -> None:
        self._calibrator.cancel()
        self._restore_remembered_square()
        self._record_runtime_diagnostic(
            "calibration-cancelled",
            "Projector calibration was cancelled.",
        )

    def _projector_calibration_completed(
        self,
        result: CalibrationResult,
    ) -> None:
        self._activate_calibration_result(result)
        self._alignment_monitor.set_paused(
            "calibration grid preview; select Black or a square to resume"
        )
        self._calibration_progress.setVisible(True)
        if self._latest_frame is None:
            return
        selected_screen = self._screens[
            self._projector_selector.currentIndex()
        ]
        active_device = self._devices[self._active_device_index]
        camera_height, camera_width = self._latest_frame.shape[:2]
        projector_geometry = selected_screen.geometry()
        try:
            saved_path = self._calibration_store.save(
                result,
                camera_name=active_device.description(),
                camera_id=self._device_key(active_device),
                camera_resolution=(camera_width, camera_height),
                projector_name=selected_screen.name(),
                projector_resolution=(
                    projector_geometry.width(),
                    projector_geometry.height(),
                ),
            )
        except OSError as error:
            QMessageBox.warning(
                self,
                "Calibration save failed",
                str(error),
            )
            self._projector_status.setText(
                f"{self._projector_status.text()} Saving failed."
            )
        else:
            self._projector_status.setText(
                f"{self._projector_status.text()} "
                f"Saved to calibrations/{saved_path.name}."
            )
        self._record_runtime_diagnostic(
            "calibration-completed",
            self._projector_status.text(),
            details={
                "board_confidence": result.board_detection.confidence,
                "board_method": result.board_detection.method,
                "board_reprojection_error_pixels": (
                    result.board_detection.reprojection_error
                ),
                "mapping_coverage_ratio": result.mapping.coverage_ratio,
                "mapping_inlier_ratio": result.mapping.inlier_ratio,
                "mapping_rms_error_projector_pixels": (
                    result.mapping.rms_error
                ),
                "validation_rms_error_camera_pixels": (
                    result.validation.rms_error
                ),
                "validation_maximum_error_camera_pixels": (
                    result.validation.maximum_error
                ),
            },
            include_images=True,
        )
        self._restore_remembered_square()

    def _projector_calibration_failed(self, message: str) -> None:
        self._projector_window.show_black()
        if self._calibrator.quick_verification:
            operation = "Quick realignment"
        elif self._calibrator.validation_only:
            operation = "Validation"
        elif self._calibrator.coded_pattern_mode:
            operation = "Populated calibration"
        else:
            operation = "Full calibration"
        if (
            self._calibration_result is not None
            and self._calibrator.coded_pattern_mode
        ):
            self._alignment_monitor.set_black()
            self._set_calibration_indicator(
                self._projector_calibration_indicator,
                "Projector calibration",
                "ok",
                f"{operation} failed: {message} "
                "The previous verified mapping was retained.",
            )
            self._projector_status.setText(
                f"{operation} failed: {message} The previous verified "
                "mapping remains active."
            )
            self._record_runtime_diagnostic(
                "calibration-failed",
                message,
                details={
                    "operation": operation,
                    "previous_calibration_retained": True,
                    "previous_calibration_invalidated": False,
                },
                include_images=True,
                additional_images={
                    f"calibration-{name}": image
                    for name, image in (
                        self._calibrator.diagnostic_captures.items()
                    )
                },
            )
            return
        if self._calibration_result is not None:
            invalidation_error = self._invalidate_saved_calibration()
            self._invalidate_calibration()
            self._set_calibration_indicator(
                self._projector_calibration_indicator,
                "Projector calibration",
                "error",
                message,
            )
            self._projector_status.setText(
                f"{operation} failed: {message} The previous mapping was "
                "invalidated; recalibration is required."
            )
            if invalidation_error is not None:
                self._projector_status.setText(
                    f"{self._projector_status.text()} {invalidation_error}"
                )
            if self._calibrator.validation_only:
                QMessageBox.warning(
                    self,
                    "Projector calibration validation failed",
                    message,
                )
            self._record_runtime_diagnostic(
                "calibration-failed",
                message,
                details={
                    "operation": operation,
                    "previous_calibration_retained": False,
                    "previous_calibration_invalidated": True,
                },
                include_images=True,
                additional_images={
                    f"calibration-{name}": image
                    for name, image in (
                        self._calibrator.diagnostic_captures.items()
                    )
                },
            )
            return
        self._invalidate_calibration()
        self._set_calibration_indicator(
            self._projector_calibration_indicator,
            "Projector calibration",
            "error",
            message,
        )
        self._record_runtime_diagnostic(
            "calibration-failed",
            message,
            details={
                "operation": operation,
                "previous_calibration_retained": False,
            },
            include_images=True,
            additional_images={
                f"calibration-{name}": image
                for name, image in self._calibrator.diagnostic_captures.items()
            },
        )
        QMessageBox.warning(
            self,
            "Projector calibration failed",
            message,
        )

    def _copy_physical_board_fen(self) -> None:
        fen = position_fen_with_inferred_rights(self._board_widget.pieces)
        QGuiApplication.clipboard().setText(fen)
        self.statusBar().showMessage("Physical position FEN copied.", 3000)

    def _copy_physical_board_image(self) -> None:
        QGuiApplication.clipboard().setPixmap(
            self._board_widget.rendered_board_pixmap()
        )
        self.statusBar().showMessage("Physical board image copied.", 3000)

    def _copy_lichess_board_fen(self) -> None:
        fen = (
            self._lichess_game_controller.server_fen
            if self._lichess_game_controller is not None
            else position_fen_with_inferred_rights(
                self._lichess_board_widget.pieces
            )
        )
        QGuiApplication.clipboard().setText(fen)
        self.statusBar().showMessage("Lichess position FEN copied.", 3000)

    def _copy_lichess_board_image(self) -> None:
        QGuiApplication.clipboard().setPixmap(
            self._lichess_board_widget.rendered_board_pixmap()
        )
        self.statusBar().showMessage("Lichess board image copied.", 3000)

    def _show_current_lichess_game(self) -> None:
        game = self._displayed_lichess_game()
        if game is None:
            return
        message_box = QMessageBox(self)
        message_box.setIcon(QMessageBox.Icon.Information)
        message_box.setWindowTitle("Current Lichess game")
        message_box.setText(
            _current_lichess_game_text(
                game,
                clocks_available=self._lichess_clocks_available,
                synchronization_status=self._lichess_sync_status.text(),
            )
        )
        message_box.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        message_box.exec()

    def _board_square_clicked(self, row: int, column: int) -> None:
        clicked_square = f"{chr(ord('a') + column)}{8 - row}"
        if (
            self._calibration_result is None
            or self._camera_only_calibration
        ):
            self._projector_status.setText(
                "No matching projector calibration is available."
            )
            self._record_runtime_diagnostic(
                "board-square-projection-rejected",
                self._projector_status.text(),
                details={
                    "square": clicked_square,
                    "calibration_available": (
                        self._calibration_result is not None
                    ),
                    "camera_only_calibration": self._camera_only_calibration,
                },
                include_images=True,
            )
            return
        if self._piece_side_orientation_needs_detection:
            self._projector_status.setText(
                "Square projection unavailable until White-side orientation "
                "is resolved."
            )
            self._record_runtime_diagnostic(
                "board-square-projection-rejected",
                self._projector_status.text(),
                details={
                    "square": clicked_square,
                    "reason": "white-side-orientation-unresolved",
                },
                include_images=True,
            )
            return
        self._record_runtime_diagnostic(
            "board-square-clicked",
            f"Physical board square {clicked_square} was clicked.",
            details={"square": clicked_square},
        )
        self._remembered_board_square = (row, column)
        self._board_widget.set_selected_square((row, column))
        if self._selected_board_square == (row, column):
            self._black_projector()
            return
        if not self._open_projector():
            return
        self._pending_board_square = (row, column)
        self._projector_status.setText(
            "Capturing a black reference before projecting the selected square."
        )
        QTimer.singleShot(
            250,
            lambda: self._project_board_square_after_black(row, column),
        )

    def _project_board_square_after_black(
        self,
        row: int,
        column: int,
    ) -> None:
        clicked_square = f"{chr(ord('a') + column)}{8 - row}"
        if (
            self._pending_board_square != (row, column)
            or self._calibration_result is None
        ):
            return
        if self._latest_frame is not None:
            self._alignment_monitor.capture_black_reference(
                self._latest_frame
            )
            self._detect_current_board_orientation()
        try:
            physical_row, physical_column = self._physical_board_square(
                row,
                column,
            )
            self._projector_window.show_square(
                physical_row,
                physical_column,
                light_square=(row + column) % 2 == 0,
            )
        except (RuntimeError, ValueError) as error:
            self._pending_board_square = None
            self._record_runtime_diagnostic(
                "board-square-projection-failed",
                str(error),
                details={"square": clicked_square},
                include_images=True,
            )
            QMessageBox.warning(self, "Square projection", str(error))
            return

        self._pending_board_square = None
        self._selected_board_square = (row, column)
        self._remembered_board_square = (row, column)
        self._board_widget.set_selected_square((row, column))
        square_name = f"{chr(ord('a') + column)}{8 - row}"
        verify_projection = (
            (row, column) not in self._piece_synchronizer.position
        )
        self._alignment_monitor.set_selected_square(
            physical_row,
            physical_column,
            verify_projection=verify_projection,
        )
        self._projector_status.setText(
            f"Projecting {square_name} (row {row + 1}, column {column + 1})."
        )
        self._record_runtime_diagnostic(
            "board-square-projected",
            self._projector_status.text(),
            details={
                "square": square_name,
                "physical_row": physical_row,
                "physical_column": physical_column,
                "optical_alignment_check": verify_projection,
            },
            include_images=True,
        )
        QTimer.singleShot(
            1000,
            lambda: self._finish_selected_square_projection(row, column),
        )

    def _finish_selected_square_projection(
        self,
        row: int,
        column: int,
    ) -> None:
        if (
            self._selected_board_square != (row, column)
            or self._projector_window.output_mode != "selected_square"
        ):
            return
        self._black_projector()
        self._projector_status.setText(
            "Square illumination finished; projector returned to black."
        )

    def _activate_calibration_result(
        self,
        result: CalibrationResult,
    ) -> None:
        CameraWindow._cancel_pending_alignment_mapping(self)
        self._camera_only_calibration = False
        self._camera_geometry_ready = True
        self._calibration_result = result
        self._verified_calibration_result = result
        self._board_orientation_needs_detection = True
        self._piece_side_orientation_needs_detection = True
        self._suspend_selected_square()
        self._reset_occupancy(
            "Occupancy stabilizing with the verified calibration."
        )
        self._projector_window.set_projector_grid(
            result.mapping.projector_grid
        )
        self._board_widget.setEnabled(True)
        self._reset_pieces_button.setEnabled(True)
        self._reconcile_button.setEnabled(
            self._piece_synchronizer.initialized
        )
        self._show_grid_button.setEnabled(True)
        self._show_squares_button.setEnabled(True)
        self._quick_verify_button.setEnabled(True)
        self._validate_calibration_button.setEnabled(True)
        selected_screen = self._screens[
            self._projector_selector.currentIndex()
        ]
        geometry = selected_screen.geometry()
        self._alignment_monitor.configure(
            result,
            (geometry.width(), geometry.height()),
        )
        self._set_calibration_indicator(
            self._camera_calibration_indicator,
            "Camera calibration",
            "ok",
        )
        self._set_calibration_indicator(
            self._projector_calibration_indicator,
            "Projector calibration",
            "ok",
        )

    def _activate_camera_only_calibration(
        self,
        result: CalibrationResult,
    ) -> None:
        CameraWindow._cancel_pending_alignment_mapping(self)
        self._camera_only_calibration = True
        self._camera_geometry_ready = False
        self._calibration_result = result
        self._verified_calibration_result = None
        self._board_orientation_needs_detection = True
        self._piece_side_orientation_needs_detection = True
        self._suspend_selected_square()
        self._reset_occupancy(
            "Occupancy stabilizing from saved camera board geometry."
        )
        self._projector_window.clear_projector_grid()
        self._board_widget.setEnabled(True)
        self._reset_pieces_button.setEnabled(True)
        self._reconcile_button.setEnabled(
            self._piece_synchronizer.initialized
        )
        self._show_grid_button.setEnabled(False)
        self._show_squares_button.setEnabled(False)
        self._quick_verify_button.setEnabled(False)
        self._validate_calibration_button.setEnabled(False)
        self._alignment_monitor.configure(result, None)
        self._set_calibration_indicator(
            self._camera_calibration_indicator,
            "Camera calibration",
            "ok",
            "Loaded saved camera board geometry.",
        )
        self._set_calibration_indicator(
            self._projector_calibration_indicator,
            "Projector calibration",
            "unknown",
            "The calibrated projector display is unavailable.",
        )

    def _invalidate_calibration(self) -> None:
        CameraWindow._cancel_pending_alignment_mapping(self)
        self._camera_only_calibration = False
        self._camera_geometry_ready = False
        self._calibration_result = None
        self._verified_calibration_result = None
        self._suspend_selected_square()
        self._reset_occupancy(
            "Occupancy unavailable: projector/camera calibration is not verified."
        )
        self._projector_window.clear_projector_grid()
        self._board_widget.setEnabled(False)
        self._reset_pieces_button.setEnabled(False)
        self._reconciliation_requested = False
        self._reconcile_button.setText("Reconcile physical board")
        self._reconcile_button.setEnabled(False)
        self._show_grid_button.setEnabled(False)
        self._show_squares_button.setEnabled(False)
        self._quick_verify_button.setEnabled(False)
        self._validate_calibration_button.setEnabled(False)
        self._alignment_monitor.clear()
        self._set_calibration_indicator(
            self._camera_calibration_indicator,
            "Camera calibration",
            "unknown",
        )
        self._set_calibration_indicator(
            self._projector_calibration_indicator,
            "Projector calibration",
            "unknown",
        )

    @staticmethod
    def _set_calibration_indicator(
        label: QLabel,
        title: str,
        state: str,
        detail: str = "",
    ) -> None:
        appearances = {
            "ok": ("OK", "#2e7d32"),
            "error": ("ERROR", "#c62828"),
            "unknown": ("NOT CALIBRATED", "#616161"),
        }
        text, background = appearances[state]
        label.setText(f"{title}: {text}")
        label.setStyleSheet(
            f"background: {background}; color: white; font-weight: bold; "
            "border-radius: 4px; padding: 5px 9px;"
        )
        label.setToolTip(detail)

    def _alignment_monitor_status(self, message: str) -> None:
        self._alignment_status.setText(message)

    def _alignment_mapping_updated(
        self,
        mapping: MappingResult,
        displacement: float,
    ) -> None:
        if self._calibration_result is None:
            return
        if self._reference_move_tracker.initialized:
            self._pending_alignment_mapping = (mapping, displacement)
            self._pending_alignment_mapping_generation += 1
            self._pending_alignment_mapping_verified = False
            generation = self._pending_alignment_mapping_generation
            self._projector_status.setText(
                f"Board movement of {displacement:.1f} camera pixels detected; "
                "waiting for physical move evaluation before remapping."
            )
            QTimer.singleShot(
                _ALIGNMENT_REMAP_DELAY_MILLISECONDS,
                lambda: self._finish_pending_alignment_mapping(generation),
            )
            return
        self._apply_alignment_mapping_update(mapping, displacement)

    def _finish_pending_alignment_mapping(
        self,
        generation: int,
    ) -> None:
        if (
            generation != self._pending_alignment_mapping_generation
            or self._pending_alignment_mapping is None
            or self._calibration_result is None
        ):
            return
        if self._alignment_mapping_conflicts_with_move_evaluation():
            QTimer.singleShot(
                _ALIGNMENT_REMAP_RETRY_MILLISECONDS,
                lambda: self._finish_pending_alignment_mapping(generation),
            )
            return
        mapping, displacement = self._pending_alignment_mapping
        self._pending_alignment_mapping = None
        self._apply_alignment_mapping_update(mapping, displacement)
        if self._pending_alignment_mapping_verified:
            self._verified_calibration_result = self._calibration_result
            self._pending_alignment_mapping_verified = False

    def _alignment_mapping_conflicts_with_move_evaluation(self) -> bool:
        return (
            self._pending_lichess_local_move is not None
            or self._pending_promotion_selection is not None
            or self._last_rejected_move is not None
            or self._last_runtime_tracker_state
            in _ALIGNMENT_REMAP_PROTECTED_TRACKER_STATES
        )

    def _cancel_pending_alignment_mapping(self) -> None:
        self._pending_alignment_mapping = None
        self._pending_alignment_mapping_generation = (
            getattr(self, "_pending_alignment_mapping_generation", 0) + 1
        )
        self._pending_alignment_mapping_verified = False

    def _apply_alignment_mapping_update(
        self,
        mapping: MappingResult,
        displacement: float,
    ) -> None:
        if self._calibration_result is None:
            return
        self._calibration_result = replace(
            self._calibration_result,
            mapping=mapping,
        )
        self._reset_occupancy(
            "Occupancy stabilizing after board movement."
        )
        self._reference_move_tracker.note_camera_grid_remap()
        if not self._camera_only_calibration:
            self._projector_window.set_projector_grid(mapping.projector_grid)
        if (
            not self._camera_only_calibration
            and self._selected_board_square is not None
        ):
            row, column = self._selected_board_square
            physical_row, physical_column = self._physical_board_square(
                row,
                column,
            )
            self._projector_window.show_square(
                physical_row,
                physical_column,
                light_square=(row + column) % 2 == 0,
            )
        if self._camera_only_calibration:
            self._projector_status.setText(
                f"Board moved {displacement:.1f} camera pixels; camera "
                "coordinates were updated."
            )
        else:
            self._projector_status.setText(
                f"Board moved {displacement:.1f} camera pixels; projected "
                "coordinates were updated without full recalibration."
            )
        self._refresh_pending_lichess_guidance()

    def _alignment_mapping_verified(self, _mapping: MappingResult) -> None:
        if self._calibration_result is None:
            return
        if self._pending_alignment_mapping is not None:
            self._pending_alignment_mapping_verified = True
            if self._alignment_mapping_conflicts_with_move_evaluation():
                return
            mapping, displacement = self._pending_alignment_mapping
            self._pending_alignment_mapping = None
            self._apply_alignment_mapping_update(mapping, displacement)
            self._pending_alignment_mapping_verified = False
        self._verified_calibration_result = self._calibration_result

    def _camera_grid_checked(self) -> None:
        if (
            not self._camera_only_calibration
            or self._camera_geometry_ready
        ):
            return
        self._camera_geometry_ready = True
        self._reset_occupancy(
            "Occupancy stabilizing after camera-grid verification."
        )

    def _suspend_selected_square(self) -> None:
        self._selected_board_square = None
        self._pending_board_square = None
        self._board_widget.set_selected_square(None)

    def _reset_occupancy(self, status: str) -> None:
        self._occupancy_detector.reset()
        self._automatic_start_occupancy_detector.reset()
        self._reference_move_tracker.reset_activity()
        self._latest_stable_occupied_ui = None
        self._board_widget.set_occupied_squares(())
        self._occupancy_status.setText(status)
        if self._last_rejected_move is not None:
            self._piece_status.setText(
                _illegal_correction_message(
                    self._last_rejected_move,
                    self._legal_game,
                )
            )
        elif self._piece_synchronizer.initialized:
            self._piece_status.setText(
                "Piece position retained while occupancy restabilizes."
            )
        else:
            self._piece_status.setText(
                "Piece position is not synchronized."
            )

    def _reset_board_edge_activity_indicator(self) -> None:
        self._board_edge_activity_detector.reset()
        self._board_edge_indicator_active = False
        self._board_edge_indicator_active_since = None
        self._board_edge_indicator_last_stuck_diagnostic_at = None
        self._projector_window.set_edge_activity_indicator(False)

    def _update_board_edge_activity_indicator(self) -> None:
        if self._latest_frame is None or self._calibration_result is None:
            self._reset_board_edge_activity_indicator()
            return
        try:
            active = self._board_edge_activity_detector.observe(
                self._latest_frame,
                self._calibration_result.mapping.camera_grid,
            )
        except (ValueError, RuntimeError, cv2.error) as error:
            self._reset_board_edge_activity_indicator()
            self._projector_status.setText(
                f"Board-edge indicator unavailable: {error}"
            )
            return
        previous_active = self._board_edge_indicator_active
        self._board_edge_indicator_active = active
        self._projector_window.set_edge_activity_indicator(active)
        observed_at = monotonic()
        if active:
            if (
                not previous_active
                or self._board_edge_indicator_active_since is None
            ):
                self._board_edge_indicator_active_since = observed_at
                self._board_edge_indicator_last_stuck_diagnostic_at = None
            active_seconds = max(
                0.0,
                observed_at - self._board_edge_indicator_active_since,
            )
        else:
            active_seconds = (
                0.0
                if self._board_edge_indicator_active_since is None
                else max(
                    0.0,
                    observed_at - self._board_edge_indicator_active_since,
                )
            )
            self._board_edge_indicator_active_since = None
            self._board_edge_indicator_last_stuck_diagnostic_at = None
        detector = self._board_edge_activity_detector
        bounds = detector.largest_component_bounds
        details = {
            "active": active,
            "active_seconds": round(active_seconds, 3),
            "crossing_detected": detector.crossing_detected,
            "component_pixels": detector.largest_component_pixels,
            "inside_pixels": detector.largest_component_inside_pixels,
            "outside_pixels": detector.largest_component_outside_pixels,
            "outer_halo_pixels": (
                detector.largest_component_outer_halo_pixels
            ),
            "component_bounds": (
                None if bounds is None else list(bounds)
            ),
        }
        if active == previous_active:
            last_stuck = self._board_edge_indicator_last_stuck_diagnostic_at
            if (
                not active
                or active_seconds < _BOARD_EDGE_STUCK_DIAGNOSTIC_SECONDS
                or (
                    last_stuck is not None
                    and observed_at - last_stuck
                    < _BOARD_EDGE_STUCK_REPEAT_SECONDS
                )
            ):
                return
            self._board_edge_indicator_last_stuck_diagnostic_at = observed_at
            self._record_runtime_diagnostic(
                "board-edge-indicator-still-active",
                (
                    "Board-edge crossing indicator remains active after "
                    f"{active_seconds:.1f} seconds."
                ),
                details=details,
                include_images=True,
            )
            return
        message = (
            "Board-edge crossing indicator activated."
            if active
            else "Board-edge crossing indicator cleared."
        )
        self._record_runtime_diagnostic(
            (
                "board-edge-indicator-activated"
                if active
                else "board-edge-indicator-cleared"
            ),
            message,
            details=details,
            include_images=True,
        )

    def _update_occupancy(self) -> None:
        if (
            self._latest_frame is None
            or self._calibration_result is None
            or self._calibrator.active
        ):
            return
        if (
            self._camera_only_calibration
            and not self._camera_geometry_ready
        ):
            self._occupancy_status.setText(
                "Occupancy waiting for camera-grid verification."
            )
            return
        self._update_board_edge_activity_indicator()
        if (
            self._activate_offline_after_reset
            and not self._reset_to_start_requested
        ):
            self._reset_pieces_to_start()
            return
        if self._pending_promotion_selection is not None:
            self._update_promotion_selection()
            return
        if self._illegal_correction_projection_active:
            self._observe_illegal_correction_activity()
            return
        if self._illegal_correction_blackout_pending:
            self._occupancy_status.setText(
                "Red correction removed; waiting for projector black to "
                "settle."
            )
            return
        if self._offline_move_marking_active:
            self._occupancy_status.setText(
                "Accepted move is marked for one second."
            )
            return
        if self._offline_move_marking_blackout_pending:
            self._occupancy_status.setText(
                "Accepted-move marking removed; waiting for projector black "
                "to settle."
            )
            return
        if self._lichess_guidance_projection_active:
            self._observe_lichess_guidance_activity()
            return
        if self._lichess_guidance_blackout_pending:
            self._observe_lichess_guidance_blackout()
            return
        if self._reconciliation_blackout_pending:
            self._occupancy_status.setText(
                "Reconciliation marks removed; waiting for projector black "
                "to settle."
            )
            return
        if (
            not self._camera_only_calibration
            and self._alignment_monitor.output_mode != "black"
        ):
            self._occupancy_status.setText(
                "Occupancy paused until the projector is black."
            )
            return
        self._detect_current_board_orientation()
        if self._board_orientation_needs_detection:
            self._occupancy_status.setText(
                "Occupancy unavailable: board orientation is unresolved."
            )
            return
        if self._detect_automatic_starting_position():
            return
        if (
            self._reference_move_tracker.initialized
            and not self._piece_side_orientation_needs_detection
            and not self._reset_to_start_requested
            and not self._reconciliation_requested
            and not self._offline_restore_requested
        ):
            self._update_reference_move_tracking()
            return
        try:
            update = self._occupancy_detector.observe(
                self._latest_frame,
                self._calibration_result.mapping.camera_grid,
            )
        except (ValueError, cv2.error) as error:
            self._occupancy_status.setText(
                f"Occupancy unavailable: {error}"
            )
            return
        if update.state == "motion":
            self._occupancy_status.setText(
                "Occupancy frozen while motion or occlusion is present."
            )
            return
        if update.state == "settling":
            self._occupancy_status.setText(
                f"Waiting for the hand to leave the board "
                f"({update.observation_count}/"
                f"{update.required_observations} quiet observations)."
            )
            return
        if update.result is None:
            self._occupancy_status.setText(
                f"Occupancy stabilizing: {update.observation_count}/"
                f"{update.required_observations} observations."
            )
            return
        occupied_ui = {
            rotate_square(row, column, -self._board_quarter_turns)
            for row, column in update.result.occupied_squares
        }
        if (
            update.state == "stable"
            and self._piece_side_orientation_needs_detection
        ):
            try:
                self._board_quarter_turns = (
                    detect_starting_position_orientation(
                        self._latest_frame,
                        self._calibration_result.mapping.camera_grid,
                        update.result.occupied_squares,
                        self._board_quarter_turns,
                    )
                )
            except ValueError:
                pass
            else:
                self._piece_side_orientation_needs_detection = False
                self._settings.setValue(
                    _BOARD_ORIENTATION_SETTING,
                    self._board_quarter_turns,
                )
                self._settings.sync()
                self._refresh_pending_lichess_guidance()
                occupied_ui = {
                    rotate_square(
                        row,
                        column,
                        -self._board_quarter_turns,
                    )
                    for row, column in update.result.occupied_squares
                }
        self._board_widget.set_occupied_squares(occupied_ui)
        if update.state == "stable":
            stable_occupied = frozenset(occupied_ui)
            self._latest_stable_occupied_ui = stable_occupied
            if self._reset_to_start_requested:
                self._complete_starting_position_reset(stable_occupied)
                return
            if self._reconciliation_requested:
                self._update_physical_reconciliation(stable_occupied)
                return
            if self._offline_restore_requested:
                self._update_offline_position_restore(stable_occupied)
                return
            position_update = self._piece_synchronizer.observe(
                stable_occupied,
            )
            self._board_widget.set_pieces(position_update.position)
            self._persist_piece_position(position_update.position)
            self._reconcile_button.setEnabled(
                self._piece_synchronizer.initialized
            )
            self._piece_status.setText(position_update.message)
            if (
                self._piece_synchronizer.initialized
                and stable_occupied
                == frozenset(self._piece_synchronizer.position)
            ):
                self._reference_move_tracker.set_reference(
                    self._latest_frame,
                    self._calibration_result.mapping.camera_grid,
                    self._board_quarter_turns,
                )
                self._persist_move_reference()
                self._occupancy_status.setText(
                    "Reference image established; watching for board activity."
                )
                self._board_widget.set_occupied_squares(stable_occupied)
                return
        if update.state == "stabilizing":
            self._occupancy_status.setText(
                f"Occupancy provisional "
                f"({update.observation_count}/"
                f"{update.required_observations}): "
                f"{update.result.occupied_count} occupied, "
                f"{update.result.empty_count} empty."
            )
        else:
            self._occupancy_status.setText(
                f"Occupancy stable: {update.result.occupied_count} occupied, "
                f"{update.result.empty_count} empty."
            )

    def _detect_automatic_starting_position(self) -> bool:
        if (
            not self._automatic_start_reset
            or self._lichess_game_controller is not None
            or self._latest_frame is None
            or self._calibration_result is None
        ):
            return False
        starting_game = LegalGame()
        if (
            self._game_mode == "offline-game"
            and self._legal_game.fen == starting_game.fen
            and self._piece_synchronizer.position == starting_game.position
        ):
            return False
        try:
            update = self._automatic_start_occupancy_detector.observe(
                self._latest_frame,
                self._calibration_result.mapping.camera_grid,
            )
        except (ValueError, cv2.error) as error:
            self._occupancy_status.setText(
                f"Automatic starting-position detection unavailable: {error}"
            )
            return False
        if update.result is None:
            return False
        occupied_ui = frozenset(
            rotate_square(row, column, -self._board_quarter_turns)
            for row, column in update.result.occupied_squares
        )
        if occupied_ui != starting_occupied_squares():
            return False
        if update.state != "stable":
            self._piece_status.setText(
                "Starting position detected; confirming stability "
                f"({update.observation_count}/"
                f"{update.required_observations})."
            )
            return True
        self._activate_offline_after_reset = True
        self._automatic_start_reset_recovery = False
        self._complete_starting_position_reset(occupied_ui)
        self._piece_status.setText(
            "Starting position detected automatically; Offline game ready. "
            "White to move."
        )
        return True

    def _update_reference_move_tracking(self) -> None:
        if self._latest_frame is None or self._calibration_result is None:
            return
        if self._pending_lichess_local_move is not None:
            return
        lichess_controller = self._lichess_game_controller
        lichess_signature = (
            None
            if lichess_controller is None
            else lichess_controller.expected_signature
        )
        tracking_position = (
            self._piece_synchronizer.position
            if lichess_controller is None
            else lichess_controller.physical_position
        )
        offline_signatures = self._legal_game.physical_move_signatures
        if (
            lichess_controller is None
            and self._game_mode == "offline-game"
            and getattr(self, "_offline_turn_inference_pending", False)
        ):
            pending_games = self._pending_offline_games
            if pending_games is None:
                pending_games = _inferred_offline_games(tracking_position)
                self._pending_offline_games = pending_games
            offline_signatures = tuple(
                signature
                for game in pending_games
                for signature in game.physical_move_signatures
            )
        try:
            update = self._reference_move_tracker.observe(
                self._latest_frame,
                self._calibration_result.mapping.camera_grid,
                self._board_quarter_turns,
                tracking_position,
                (
                    (lichess_signature,)
                    if lichess_signature is not None
                    else lichess_controller.local_move_signatures
                    if lichess_controller is not None
                    else offline_signatures
                    if self._game_mode == "offline-game"
                    else ()
                ),
                required_signature=lichess_signature,
            )
        except (ValueError, cv2.error) as error:
            self._occupancy_status.setText(
                f"Reference tracking unavailable: {error}"
            )
            return
        status_message = _reference_tracking_status_message(update)
        if status_message is not None:
            self._occupancy_status.setText(status_message)
        if update.state != self._last_runtime_tracker_state:
            self._last_runtime_tracker_state = update.state
            if update.state in {
                "accepted",
                "ambiguous",
                "illegal-removal",
                "illegal-relocation",
                "pending-capture",
                "pending-castling",
                "pending-en-passant",
            }:
                self._record_runtime_diagnostic(
                    "reference-tracker",
                    update.message,
                    tracker_update=update,
                    include_images=True,
                )
        self._restore_pending_lichess_guidance_after_no_change(update)
        rejected_move = self._last_rejected_move
        correction_restored = (
            rejected_move is not None
            and self._reference_move_tracker.rejected_move_is_restored(
                update,
                rejected_move,
            )
        )
        if correction_restored:
            self._board_widget.set_illegal_source(None)
            restored_move = self._last_rejected_move
            self._last_rejected_move = None
            self._illegal_correction_last_projected_move = None
            self._occupancy_status.setText(
                "Board matches the accepted reference."
            )
            self._record_runtime_diagnostic(
                "illegal-correction-restored",
                "Physical board restored to the accepted reference.",
                details={
                    "move": (
                        None
                        if restored_move is None
                        else [
                            square_name(restored_move[0]),
                            square_name(restored_move[1]),
                        ]
                    ),
                    "generation": self._illegal_correction_generation,
                },
                tracker_update=update,
                include_images=True,
            )
            if (
                lichess_controller is not None
                and lichess_controller.state == "correction"
            ):
                synchronization = (
                    lichess_controller.note_correction_restored()
                )
                self._piece_status.setText(synchronization.message)
                self._lichess_sync_status.setText(
                    synchronization.message
                )
                self._last_projected_lichess_uci = None
                self._update_lichess_synchronization_projection()
                self._record_runtime_diagnostic(
                    "lichess-correction-restored",
                    synchronization.message,
                    include_images=True,
                )
            else:
                self._piece_status.setText(
                    _accepted_position_message(self._legal_game)
                )
        if update.state == "illegal-removal" and update.move is not None:
            if lichess_controller is not None:
                if lichess_signature is not None:
                    lichess_controller.confirm_expected_move(update.move)
                self._reject_unconfirmed_lichess_move(update.move)
                return
            square = update.move[0]
            self._reference_move_tracker.reset_activity()
            self._last_rejected_move = update.move
            self._board_widget.set_illegal_source(square)
            correction_game = CameraWindow._offline_correction_game(
                self,
                update.move,
            )
            self._piece_status.setText(
                _illegal_correction_message(update.move, correction_game)
            )
            self._project_illegal_move_correction(update.move)
            return
        if update.state == "illegal-relocation" and update.move is not None:
            self._reference_move_tracker.reset_activity()
            self._last_rejected_move = update.move
            self._board_widget.set_illegal_source(update.move[0])
            correction_game = CameraWindow._offline_correction_game(
                self,
                update.move,
            )
            self._piece_status.setText(
                _illegal_correction_message(update.move, correction_game)
            )
            self._project_illegal_move_correction(update.move)
            return
        if update.state != "accepted" or update.move is None:
            return
        source, destination = update.move
        legal_move: LegalMoveResult | None = None
        offline_move_game: LegalGame | None = None
        inferred_offline_turn = False
        lichess_pending = (
            lichess_controller is not None
            and lichess_signature is not None
        )
        if lichess_controller is not None and not lichess_pending:
            local_signature = next(
                (
                    signature
                    for signature in lichess_controller.local_move_signatures
                    if (
                        signature.source,
                        signature.destination,
                    )
                    == update.move
                ),
                None,
            )
            if (
                self._promotion_mode == "select-promoted-piece"
                and local_signature is not None
                and local_signature.kind
                in ("promotion", "capture-promotion")
            ):
                self._start_promotion_selection(update.move)
                return
            try:
                uci, legal_move = lichess_controller.prepare_local_move(
                    update.move
                )
            except ValueError:
                self._reject_unconfirmed_lichess_move(update.move)
                return
            self._begin_lichess_local_move_submission(
                update.move,
                uci,
                legal_move,
            )
            return
        if lichess_pending:
            expected_move = (
                lichess_signature.source,
                lichess_signature.destination,
            )
            if update.move != expected_move:
                lichess_controller.confirm_expected_move(update.move)
                self._reject_unconfirmed_lichess_move(update.move)
                return
            legal_move = lichess_controller.pending_result
            if legal_move is None:
                self._lichess_sync_status.setText(
                    "The pending Lichess move cannot be applied physically."
                )
                return
        elif self._game_mode == "offline-game":
            offline_move_game = self._legal_game
            if getattr(self, "_offline_turn_inference_pending", False):
                inferred_game = _inferred_offline_game_for_source(
                    tracking_position,
                    source,
                )
                if inferred_game is not None:
                    offline_move_game = inferred_game
                    inferred_offline_turn = True
            local_signature = next(
                (
                    signature
                    for signature in offline_move_game.legal_move_signatures
                    if (
                        signature.source,
                        signature.destination,
                    )
                    == update.move
                ),
                None,
            )
            if (
                self._promotion_mode == "select-promoted-piece"
                and local_signature is not None
                and local_signature.kind
                in ("promotion", "capture-promotion")
            ):
                if inferred_offline_turn:
                    self._legal_game = offline_move_game
                self._start_promotion_selection(update.move)
                return
            try:
                legal_move = offline_move_game.apply_move(
                    source,
                    destination,
                )
            except ValueError as error:
                self._reference_move_tracker.reset_activity()
                self._last_rejected_move = update.move
                self._board_widget.set_illegal_source(source)
                self._piece_status.setText(
                    _illegal_correction_message(
                        update.move,
                        offline_move_game,
                    )
                )
                self._project_illegal_move_correction(update.move)
                return
        previous_position = self._piece_synchronizer.position
        try:
            if legal_move is None:
                position_update = self._piece_synchronizer.accept_move(
                    source,
                    destination,
                )
            else:
                position_update = (
                    self._piece_synchronizer.accept_authoritative_move(
                        source,
                        destination,
                        legal_move.position,
                    )
                )
            self._reference_move_tracker.commit_accepted(update.move)
        except ValueError as error:
            self._piece_synchronizer.restore(previous_position)
            if offline_move_game is not None and legal_move is not None:
                offline_move_game.rollback()
            self._reference_move_tracker.reset_activity()
            self._piece_status.setText(
                f"Reference move rejected: {error}"
            )
            return
        if inferred_offline_turn and offline_move_game is not None:
            self._legal_game = offline_move_game
            self._offline_turn_inference_pending = False
            self._pending_offline_games = None
        self._board_widget.set_pieces(position_update.position)
        self._board_widget.set_occupied_squares(position_update.position)
        self._board_widget.set_illegal_source(None)
        self._board_widget.set_guidance_move(None)
        self._last_rejected_move = None
        self._illegal_correction_last_projected_move = None
        self._board_widget.set_last_move(update.move)
        self._persist_piece_position(position_update.position)
        self._persist_move_reference()
        if lichess_pending and lichess_controller is not None:
            synchronization = lichess_controller.confirm_expected_move(
                update.move
            )
            self._lichess_sync_status.setText(synchronization.message)
            self._piece_status.setText(synchronization.message)
            self._last_projected_lichess_uci = None
            self._update_lichess_synchronization_projection()
            self._record_runtime_diagnostic(
                "lichess-physical-move-confirmed",
                synchronization.message,
                include_images=True,
            )
        elif legal_move is None:
            self._piece_status.setText(position_update.message)
        else:
            self._persist_game_state()
            self._update_turn_indicator()
            self._piece_status.setText(
                f"Legal move {legal_move.san} accepted; "
                f"{legal_move.status.turn} to move."
            )
        if not lichess_pending:
            if legal_move is None:
                self._project_accepted_move(update.move)
            else:
                self._project_offline_accepted_move(update.move)

    def _offline_correction_game(
        self,
        move: tuple[tuple[int, int], tuple[int, int]],
    ) -> LegalGame:
        if not getattr(self, "_offline_turn_inference_pending", False):
            return self._legal_game
        return (
            _inferred_offline_game_for_source(
                self._piece_synchronizer.position,
                move[0],
            )
            or self._legal_game
        )

    def _start_promotion_selection(
        self,
        move: tuple[tuple[int, int], tuple[int, int]],
    ) -> None:
        if self._latest_frame is None or self._calibration_result is None:
            self._fail_promotion_selection(
                "Promotion selection requires a calibrated camera frame."
            )
            return
        try:
            edge = _physical_promotion_edge(
                move,
                self._board_quarter_turns,
            )
        except ValueError as error:
            self._fail_promotion_selection(
                str(error)
            )
            return
        polygons = promotion_choice_polygons(
            self._calibration_result.mapping.camera_grid,
            edge,
        )
        try:
            validate_promotion_polygons(
                polygons,
                self._latest_frame.shape[1],
                self._latest_frame.shape[0],
            )
        except ValueError as error:
            self._fail_promotion_selection(str(error))
            return
        self._pending_promotion_selection = _PendingPromotionSelection(
            move,
            edge,
            polygons,
            self._latest_frame.copy(),
            viewer_edge=_physical_player_edge(
                (
                    self._lichess_player_color
                    if self._lichess_game_controller is not None
                    and self._lichess_player_color is not None
                    else "white"
                ),
                self._board_quarter_turns,
            ),
        )
        self._show_promotion_choices()

    def _update_promotion_selection(self) -> None:
        selection = self._pending_promotion_selection
        if (
            selection is None
            or self._latest_frame is None
            or self._calibration_result is None
        ):
            return
        if selection.stage == "verifying":
            self._verify_promotion_replacement()
            return
        if selection.stage.endswith("-blackout"):
            self._occupancy_status.setText(
                "Promotion projector is black; waiting for clean evidence."
            )
            return
        activity_polygons = selection.camera_polygons
        if selection.piece is not None:
            physical_destination = self._physical_board_square(
                *selection.move[1]
            )
            destination_polygon = square_polygons(
                self._calibration_result.mapping.camera_grid
            )[
                physical_destination[0] * 8
                + physical_destination[1]
            ]
            activity_polygons = np.concatenate(
                (
                    activity_polygons,
                    destination_polygon[np.newaxis, :, :],
                ),
                axis=0,
            )
        try:
            settled = self._promotion_activity_detector.observe(
                self._latest_frame,
                activity_polygons,
            )
        except (ValueError, cv2.error) as error:
            self._fail_promotion_selection(
                f"Promotion hand detection failed: {error}"
            )
            return
        if not settled:
            self._occupancy_status.setText(
                "Promotion choices remain projected during manipulation."
            )
            return
        selection.generation += 1
        generation = selection.generation
        selection.stage = f"{selection.stage}-blackout"
        self._projector_window.show_black()
        self._alignment_monitor.set_black()
        self._projector_status.setText(
            "Promotion manipulation settled; projector is black."
        )
        QTimer.singleShot(
            1000,
            lambda: self._finish_promotion_blackout(generation),
        )

    def _finish_promotion_blackout(self, generation: int) -> None:
        selection = self._pending_promotion_selection
        if (
            selection is None
            or selection.generation != generation
            or self._latest_frame is None
        ):
            return
        if selection.stage == "choices-blackout":
            try:
                observation = observe_promotion_choices(
                    selection.reference_frame,
                    self._latest_frame,
                    selection.camera_polygons,
                )
            except (ValueError, cv2.error) as error:
                self._fail_promotion_selection(
                    f"Promotion choice detection failed: {error}"
                )
                return
            piece = observation.selected_piece
            if piece is None:
                self._show_promotion_choices(
                    "No unambiguous promotion choice was detected."
                )
                return
            selection.piece = piece
            self._show_promotion_confirmation()
            return
        if selection.stage == "confirmation-blackout":
            try:
                observation = observe_promotion_choices(
                    selection.reference_frame,
                    self._latest_frame,
                    selection.camera_polygons,
                )
            except (ValueError, cv2.error) as error:
                self._fail_promotion_selection(
                    f"Promotion choice detection failed: {error}"
                )
                return
            if observation.selected_piece is not None:
                selection.piece = observation.selected_piece
                self._show_promotion_confirmation()
                return
            if observation.occupied_choices:
                self._show_promotion_confirmation(
                    "Multiple promotion choices were detected."
                )
                return
            selection.stage = "verifying"
            self._promotion_activity_detector.reset()
            self._reference_move_tracker.reset_activity()
            self._projector_status.setText(
                "Projector is black; verifying promotion replacement."
            )

    def _show_promotion_choices(self, reason: str | None = None) -> None:
        selection = self._pending_promotion_selection
        if selection is None:
            return
        if self._camera_only_calibration or not self._open_projector():
            self._fail_promotion_selection(
                "Promotion selection requires a calibrated projector."
            )
            return
        try:
            self._projector_window.show_promotion_choices(
                selection.edge,
                selection.viewer_edge,
            )
        except (RuntimeError, ValueError) as error:
            self._fail_promotion_selection(
                f"Promotion choices cannot be projected: {error}"
            )
            return
        selection.stage = "choices"
        selection.piece = None
        self._promotion_activity_detector.reset()
        self._alignment_monitor.set_paused("promotion choices")
        message = (
            "Move the pawn onto the projected promotion piece."
            if reason is None
            else f"{reason} Move the pawn onto one projected piece."
        )
        self._piece_status.setText(message)
        self._projector_status.setText(message)

    def _show_promotion_confirmation(
        self,
        reason: str | None = None,
    ) -> None:
        selection = self._pending_promotion_selection
        if selection is None or selection.piece is None:
            return
        if not self._open_projector():
            self._fail_promotion_selection(
                "Promotion confirmation requires a calibrated projector."
            )
            return
        destination = self._physical_board_square(*selection.move[1])
        try:
            self._projector_window.show_promotion_confirmation(
                selection.edge,
                selection.viewer_edge,
                selection.piece,
                destination,
            )
        except (RuntimeError, ValueError) as error:
            self._fail_promotion_selection(
                f"Promotion confirmation cannot be projected: {error}"
            )
            return
        selection.stage = "confirmation"
        self._promotion_activity_detector.reset()
        self._alignment_monitor.set_paused("promotion replacement")
        piece_name = {
            "q": "queen",
            "r": "rook",
            "b": "bishop",
            "n": "knight",
        }[selection.piece]
        instruction = (
            f"{piece_name.capitalize()} selected. Move the pawn to another "
            "option or place that replacement piece on the promotion square."
        )
        message = (
            instruction
            if reason is None
            else f"{reason} {instruction}"
        )
        self._piece_status.setText(message)
        self._projector_status.setText(message)

    def _verify_promotion_replacement(self) -> None:
        selection = self._pending_promotion_selection
        controller = self._lichess_game_controller
        if (
            selection is None
            or selection.piece is None
            or self._latest_frame is None
            or self._calibration_result is None
        ):
            self._fail_promotion_selection(
                "Promotion replacement lost its active game state."
            )
            return
        if controller is None and self._game_mode != "offline-game":
            self._fail_promotion_selection(
                "Promotion replacement lost its active game state."
            )
            return
        tracking_position = (
            controller.physical_position
            if controller is not None
            else self._piece_synchronizer.position
        )
        signatures = (
            controller.local_move_signatures
            if controller is not None
            else self._legal_game.legal_move_signatures
        )
        try:
            update = self._reference_move_tracker.observe(
                self._latest_frame,
                self._calibration_result.mapping.camera_grid,
                self._board_quarter_turns,
                tracking_position,
                signatures,
            )
        except (ValueError, cv2.error) as error:
            self._fail_promotion_selection(
                f"Promotion replacement verification failed: {error}"
            )
            return
        if update.state == "accepted" and update.move == selection.move:
            if controller is None:
                self._accept_offline_promotion(
                    selection.move,
                    selection.piece,
                )
                return
            try:
                uci, result = controller.prepare_local_move(
                    selection.move,
                    selection.piece,
                )
            except ValueError as error:
                self._fail_promotion_selection(str(error))
                return
            move = selection.move
            self._pending_promotion_selection = None
            self._begin_lichess_local_move_submission(move, uci, result)
            return
        if update.state in ("watching", "ambiguous", "illegal-removal"):
            self._show_promotion_confirmation()

    def _accept_offline_promotion(
        self,
        move: tuple[tuple[int, int], tuple[int, int]],
        piece: PromotionPiece,
    ) -> None:
        previous_position = self._piece_synchronizer.position
        game_advanced = False
        try:
            legal_move = self._legal_game.apply_move(
                move[0],
                move[1],
                piece,
            )
            game_advanced = True
            position_update = (
                self._piece_synchronizer.accept_authoritative_move(
                    move[0],
                    move[1],
                    legal_move.position,
                )
            )
            self._reference_move_tracker.commit_accepted(move)
        except ValueError as error:
            self._piece_synchronizer.restore(previous_position)
            if game_advanced:
                self._legal_game.rollback()
            self._reference_move_tracker.reset_activity()
            self._fail_promotion_selection(
                f"Promotion replacement was rejected: {error}"
            )
            return
        self._pending_promotion_selection = None
        if getattr(self, "_offline_turn_inference_pending", False):
            self._offline_turn_inference_pending = False
            self._pending_offline_games = None
        self._board_widget.set_pieces(position_update.position)
        self._board_widget.set_occupied_squares(position_update.position)
        self._board_widget.set_illegal_source(None)
        self._board_widget.set_guidance_move(None)
        self._board_widget.set_last_move(move)
        self._last_rejected_move = None
        self._persist_piece_position(position_update.position)
        self._persist_move_reference()
        self._persist_game_state()
        self._update_turn_indicator()
        self._piece_status.setText(
            f"Legal move {legal_move.san} accepted; "
            f"{legal_move.status.turn} to move."
        )
        self._project_offline_accepted_move(move)

    def _fail_promotion_selection(self, message: str) -> None:
        self._pending_promotion_selection = None
        self._promotion_activity_detector.reset()
        self._projector_window.show_black()
        self._alignment_monitor.set_black()
        controller = self._lichess_game_controller
        if controller is not None:
            synchronization = controller.require_reconciliation(message)
            self._lichess_sync_status.setText(synchronization.message)
            self._piece_status.setText(synchronization.message)
        else:
            self._piece_status.setText(message)
        self._projector_status.setText(message)

    def _begin_lichess_local_move_submission(
        self,
        move: tuple[tuple[int, int], tuple[int, int]],
        uci: str,
        legal_move: LegalMoveResult,
    ) -> None:
        controller = self._lichess_game_controller
        client = self._lichess_client
        if (
            controller is None
            or client is None
            or self._lichess_submission_thread is not None
        ):
            self._reference_move_tracker.reset_activity()
            self._piece_status.setText(
                "Cannot submit the physical move because the Lichess "
                "connection is not ready."
            )
            return
        thread = QThread(self)
        worker = LichessMoveSubmissionWorker(
            client,
            controller.game_id,
            uci,
            controller.server_moves,
            confirmed_fen=controller.server_fen,
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.accepted.connect(self._lichess_local_move_accepted)
        worker.rejected.connect(self._lichess_local_move_rejected)
        worker.uncertain.connect(self._lichess_local_move_uncertain)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(self._lichess_submission_finished)
        thread.finished.connect(thread.deleteLater)
        self._pending_lichess_local_move = move
        self._pending_lichess_local_uci = uci
        self._pending_lichess_local_result = legal_move
        self._lichess_submission_thread = thread
        self._lichess_submission_worker = worker
        message = (
            f"Submitting physical move {square_name(move[0])}-"
            f"{square_name(move[1])} to Lichess."
        )
        self._piece_status.setText(message)
        self._lichess_sync_status.setText(message)
        self._record_runtime_diagnostic(
            "lichess-local-move-submitting",
            message,
            details={"uci": uci, "game_id": controller.game_id},
            include_images=True,
        )
        thread.start()

    def _lichess_local_move_accepted(
        self,
        game_id: str,
        uci: str,
    ) -> None:
        controller = self._lichess_game_controller
        move = self._pending_lichess_local_move
        legal_move = self._pending_lichess_local_result
        if (
            controller is None
            or move is None
            or legal_move is None
            or game_id != controller.game_id
            or uci != self._pending_lichess_local_uci
        ):
            return
        previous_position = self._piece_synchronizer.position
        try:
            position_update = (
                self._piece_synchronizer.accept_authoritative_move(
                    move[0],
                    move[1],
                    legal_move.position,
                )
            )
            promotion: PromotionPiece | None = (
                cast(PromotionPiece, uci[4])
                if len(uci) == 5
                else None
            )
            committed = controller.commit_local_move(move, promotion)
            self._reference_move_tracker.commit_accepted(move)
        except ValueError as error:
            self._piece_synchronizer.restore(previous_position)
            self._reference_move_tracker.reset_activity()
            synchronization = controller.require_reconciliation(
                f"Lichess accepted {uci}, but the local commit failed: "
                f"{error}"
            )
            self._piece_status.setText(synchronization.message)
            self._lichess_sync_status.setText(synchronization.message)
            self._clear_pending_lichess_local_move()
            return
        if self._lichess_worker is not None:
            self._lichess_worker.note_local_move(
                game_id,
                uci,
                controller.server_fen,
            )
        self._board_widget.set_pieces(position_update.position)
        self._board_widget.set_occupied_squares(position_update.position)
        self._board_widget.set_illegal_source(None)
        self._board_widget.set_last_move(move)
        self._last_rejected_move = None
        self._persist_piece_position(position_update.position)
        self._persist_move_reference()
        if self._latest_lichess_state is not None:
            remote_status = "started"
            winner: PlayerColor | None = None
            if committed.status.is_game_over:
                if committed.status.is_checkmate:
                    remote_status = "mate"
                    winner = (
                        "black"
                        if committed.status.turn == "white"
                        else "white"
                    )
                elif committed.status.is_stalemate:
                    remote_status = "stalemate"
                else:
                    remote_status = "draw"
            self._update_lichess_board(
                replace(
                    self._latest_lichess_state,
                    moves=controller.server_moves,
                    fen=controller.server_fen,
                    turn=committed.status.turn,
                    status=remote_status,
                    winner=winner,
                    result=committed.status.result,
                )
            )
        message = (
            f"Lichess accepted {square_name(move[0])}-"
            f"{square_name(move[1])}; {committed.status.turn} to move."
        )
        self._piece_status.setText(message)
        self._lichess_sync_status.setText(message)
        self._record_runtime_diagnostic(
            "lichess-local-move-accepted",
            message,
            details={"uci": uci, "game_id": game_id},
            include_images=True,
        )
        self._clear_pending_lichess_local_move()
        self._project_accepted_move(move)

    def _lichess_local_move_rejected(
        self,
        game_id: str,
        uci: str,
        reason: str,
    ) -> None:
        controller = self._lichess_game_controller
        move = self._pending_lichess_local_move
        if (
            controller is None
            or move is None
            or game_id != controller.game_id
            or uci != self._pending_lichess_local_uci
        ):
            return
        self._reference_move_tracker.reset_activity()
        self._last_rejected_move = move
        self._board_widget.set_illegal_source(move[0])
        message = (
            f"Lichess did not accept {square_name(move[0])}-"
            f"{square_name(move[1])}: {reason} Restore the physical move."
        )
        self._piece_status.setText(message)
        self._lichess_sync_status.setText(message)
        self._record_runtime_diagnostic(
            "lichess-local-move-rejected",
            message,
            details={"uci": uci, "game_id": game_id},
            include_images=True,
        )
        self._clear_pending_lichess_local_move()
        self._project_illegal_move_correction(move)

    def _lichess_local_move_uncertain(
        self,
        game_id: str,
        uci: str,
        reason: str,
    ) -> None:
        controller = self._lichess_game_controller
        move = self._pending_lichess_local_move
        if (
            controller is None
            or move is None
            or game_id != controller.game_id
            or uci != self._pending_lichess_local_uci
        ):
            return
        self._reference_move_tracker.reset_activity()
        synchronization = controller.require_reconciliation(
            f"The outcome of Lichess move {uci} is uncertain. {reason}"
        )
        self._last_rejected_move = move
        self._board_widget.set_illegal_source(move[0])
        self._piece_status.setText(synchronization.message)
        self._lichess_sync_status.setText(synchronization.message)
        self._record_runtime_diagnostic(
            "lichess-local-move-uncertain",
            synchronization.message,
            details={"uci": uci, "game_id": game_id},
            include_images=True,
        )
        self._clear_pending_lichess_local_move()
        self._project_illegal_move_correction(move)

    def _clear_pending_lichess_local_move(self) -> None:
        self._pending_lichess_local_move = None
        self._pending_lichess_local_uci = None
        self._pending_lichess_local_result = None

    def _reset_pieces_to_start(self) -> None:
        if self._latest_frame is None or self._calibration_result is None:
            self._piece_status.setText(
                "Cannot reset pieces until the camera board is calibrated."
            )
            return
        self._black_projector()
        self._reset_to_start_requested = True
        self._piece_side_orientation_needs_detection = True
        self._occupancy_detector.reset()
        self._latest_stable_occupied_ui = None
        self._piece_status.setText(
            "Checking the physical board before resetting pieces."
        )
        self._occupancy_status.setText(
            "Reset verification: detecting the starting position."
        )

    def _complete_starting_position_reset(
        self,
        occupied: frozenset[tuple[int, int]],
    ) -> None:
        self._reset_to_start_requested = False
        if occupied != starting_occupied_squares():
            self._piece_status.setText(
                "Cannot reset pieces: the physical board is not in the "
                "standard starting position."
            )
            self._occupancy_status.setText(
                "Reset verification finished without changing the "
                "logical position."
            )
            if self._automatic_start_reset_recovery:
                self._automatic_start_reset_recovery = False
                self._activate_offline_after_reset = False
                QTimer.singleShot(
                    0,
                    self._show_offline_mode_recovery_dialog,
                )
            elif self._activate_offline_after_reset:
                self._reset_to_start_requested = True
                self._piece_status.setText(
                    "Arrange all pieces in the standard starting position "
                    "to begin an Offline legal game."
                )
            return
        self._piece_synchronizer.reset()
        self._legal_game.reset()
        self._offline_turn_inference_pending = False
        self._pending_offline_games = None
        if self._activate_offline_after_reset:
            self._automatic_start_reset_recovery = False
            self._activate_offline_after_reset = False
            self._pending_offline_restore_game = None
            self._offline_mode_recovery_error = None
            self._game_mode = "offline-game"
            self._game_mode_selector.blockSignals(True)
            self._game_mode_selector.setCurrentIndex(1)
            self._game_mode_selector.blockSignals(False)
            self._sync_game_mode_actions()
        self._update_turn_indicator()
        update = self._piece_synchronizer.observe(
            occupied,
        )
        self._board_widget.set_pieces(update.position)
        self._board_widget.set_occupied_squares(occupied)
        self._persist_piece_position(update.position)
        self._persist_game_state()
        self._board_widget.set_illegal_source(None)
        self._last_rejected_move = None
        self._illegal_correction_last_projected_move = None
        self._board_widget.set_last_move(None)
        self._piece_status.setText(update.message)
        if self._latest_frame is None or self._calibration_result is None:
            raise RuntimeError(
                "Reset verification completed without a calibrated frame."
            )
        self._reference_move_tracker.set_reference(
            self._latest_frame,
            self._calibration_result.mapping.camera_grid,
            self._board_quarter_turns,
        )
        self._persist_move_reference()
        self._occupancy_status.setText(
            "Starting position confirmed; move reference replaced."
        )

    def _project_accepted_move(
        self,
        move: tuple[tuple[int, int], tuple[int, int]],
    ) -> None:
        if (
            self._board_orientation_needs_detection
            or self._piece_side_orientation_needs_detection
        ):
            self._projector_status.setText(
                "Move accepted; physical projection is waiting for "
                "White-side orientation."
            )
            return
        if self._camera_only_calibration:
            self._projector_status.setText(
                "Move accepted; physical projection is unavailable."
            )
            return
        if self._calibration_result is None or not self._open_projector():
            self._projector_status.setText(
                "Move accepted, but no calibrated projector is available."
            )
            return
        source, destination = move
        try:
            physical_source = self._physical_board_square(*source)
            physical_destination = self._physical_board_square(*destination)
            self._projector_window.show_move(
                physical_source,
                physical_destination,
                source_is_light=(sum(source) % 2 == 0),
                destination_is_light=(sum(destination) % 2 == 0),
            )
        except (RuntimeError, ValueError) as error:
            self._projector_status.setText(
                f"Move accepted, but projection failed: {error}"
            )
            return
        self._alignment_monitor.set_paused("accepted move indication")
        source_name = f"{chr(ord('a') + source[1])}{8 - source[0]}"
        destination_name = (
            f"{chr(ord('a') + destination[1])}{8 - destination[0]}"
        )
        self._projector_status.setText(
            f"Projecting move {source_name}-{destination_name} for 2 seconds."
        )
        QTimer.singleShot(2000, self._finish_move_projection)

    def _project_offline_accepted_move(
        self,
        move: tuple[tuple[int, int], tuple[int, int]],
    ) -> None:
        if (
            self._board_orientation_needs_detection
            or self._piece_side_orientation_needs_detection
        ):
            self._projector_status.setText(
                "Accepted Offline move; physical marking is waiting for "
                "White-side orientation."
            )
            return
        if self._camera_only_calibration:
            return
        if self._calibration_result is None or not self._open_projector():
            return
        source, destination = move
        try:
            self._projector_window.show_move(
                self._physical_board_square(*source),
                self._physical_board_square(*destination),
                source_is_light=(sum(source) % 2 == 0),
                destination_is_light=(sum(destination) % 2 == 0),
                include_turn_indicator=True,
            )
        except (RuntimeError, ValueError) as error:
            self._projector_status.setText(
                f"Accepted-move projection failed: {error}"
            )
            return
        self._offline_move_marking_generation += 1
        generation = self._offline_move_marking_generation
        self._offline_move_marking_active = True
        self._offline_move_marking_blackout_pending = False
        self._alignment_monitor.set_paused("Offline accepted move marking")
        self._projector_status.setText(
            "Showing the accepted Offline move for 1 second."
        )
        QTimer.singleShot(
            1000,
            lambda: self._finish_offline_move_marking(generation),
        )

    def _finish_offline_move_marking(self, generation: int) -> None:
        if (
            generation != self._offline_move_marking_generation
            or not self._offline_move_marking_active
        ):
            return
        self._offline_move_marking_active = False
        self._offline_move_marking_blackout_pending = True
        self._projector_window.show_black()
        self._alignment_monitor.set_black()
        self._reference_move_tracker.reset_activity()
        self._projector_status.setText(
            "Accepted-move marking removed; projector black is settling."
        )
        QTimer.singleShot(
            1000,
            lambda: self._finish_offline_move_marking_blackout(generation),
        )

    def _finish_offline_move_marking_blackout(
        self,
        generation: int,
    ) -> None:
        if (
            generation != self._offline_move_marking_generation
            or self._offline_move_marking_active
            or not self._offline_move_marking_blackout_pending
        ):
            return
        self._offline_move_marking_blackout_pending = False
        self._reference_move_tracker.reset_activity()
        self._projector_status.setText(
            "Projector black settled; watching for the next Offline move."
        )

    def _update_lichess_synchronization_projection(self) -> None:
        controller = self._lichess_game_controller
        if controller is None:
            return
        signature = controller.expected_signature
        if signature is None:
            self._last_projected_lichess_uci = None
            self._board_widget.set_guidance_move(None)
            self._cancel_lichess_guidance_activity()
            return
        projected = self._project_lichess_move_guidance(
            (signature.source, signature.destination),
            ()
            if signature.kind == "castling"
            else tuple(
                change.square
                for change in signature.changes
                if change.square
                not in (signature.source, signature.destination)
            ),
        )
        self._last_projected_lichess_uci = (
            signature.uci if projected else None
        )

    def _refresh_pending_lichess_guidance(self) -> None:
        controller = self._lichess_game_controller
        if controller is None or controller.expected_signature is None:
            return
        self._last_projected_lichess_uci = None
        self._update_lichess_synchronization_projection()

    def _project_lichess_move_guidance(
        self,
        move: tuple[tuple[int, int], tuple[int, int]],
        additional_squares: tuple[tuple[int, int], ...] = (),
    ) -> bool:
        source, destination = move
        guidance_squares = tuple(
            dict.fromkeys((source, destination, *additional_squares))
        )
        move_name = f"{square_name(source)}-{square_name(destination)}"
        self._board_widget.set_illegal_source(None)
        self._board_widget.set_reconciliation_squares(())
        if additional_squares:
            self._board_widget.set_guidance_squares(guidance_squares)
        else:
            self._board_widget.set_guidance_move(move)
        if (
            self._board_orientation_needs_detection
            or self._piece_side_orientation_needs_detection
        ):
            self._cancel_lichess_guidance_activity()
            self._projector_window.show_black()
            self._alignment_monitor.set_black()
            self._projector_status.setText(
                f"Lichess move {move_name} is pending until White-side "
                "orientation is resolved."
            )
            return False
        if self._camera_only_calibration:
            self._projector_status.setText(
                f"Lichess move {move_name} is pending; physical projection "
                "is unavailable."
            )
            return False
        if self._calibration_result is None or not self._open_projector():
            self._projector_status.setText(
                f"Lichess move {move_name} is pending, but no calibrated "
                "projector is available."
            )
            return False
        try:
            if additional_squares:
                self._projector_window.show_move_squares(
                    tuple(
                        (
                            *self._physical_board_square(*square),
                            sum(square) % 2 == 0,
                        )
                        for square in guidance_squares
                    )
                )
            else:
                self._projector_window.show_move(
                    self._physical_board_square(*source),
                    self._physical_board_square(*destination),
                    source_is_light=(sum(source) % 2 == 0),
                    destination_is_light=(sum(destination) % 2 == 0),
                )
        except (RuntimeError, ValueError) as error:
            self._projector_status.setText(
                f"Lichess move guidance failed: {error}"
            )
            return False
        reprojection = self._lichess_guidance_reprojection_pending
        self._alignment_monitor.set_paused("Lichess move guidance")
        self._lichess_guidance_generation += 1
        self._lichess_guidance_projection_active = True
        self._lichess_guidance_blackout_pending = False
        self._lichess_guidance_reprojection_pending = False
        self._lichess_guidance_squares = guidance_squares
        self._lichess_guidance_edge_entered = False
        self._lichess_guidance_activity_logged = False
        status_message = (
            f"Projecting pending Lichess move {move_name} until the hand "
            "leaves the board."
        )
        self._projector_status.setText(status_message)
        self._record_runtime_diagnostic(
            (
                "lichess-guidance-reprojected"
                if reprojection
                else "lichess-guidance-projected"
            ),
            status_message,
            details={
                "move": move_name,
                "squares": [
                    square_name(square)
                    for square in guidance_squares
                ],
                "generation": self._lichess_guidance_generation,
            },
            include_images=True,
        )
        return True

    def _observe_lichess_guidance_activity(self) -> None:
        if not self._lichess_guidance_projection_active:
            return
        if not self._board_edge_indicator_active:
            if self._lichess_guidance_edge_entered:
                self._begin_lichess_guidance_observation()
            return
        self._lichess_guidance_edge_entered = True
        status_message = (
            "Opponent move in progress; guidance remains projected until "
            "the hand leaves the board."
        )
        self._projector_status.setText(status_message)
        if not self._lichess_guidance_activity_logged:
            self._lichess_guidance_activity_logged = True
            self._record_runtime_diagnostic(
                "lichess-guidance-hand-entered",
                status_message,
                details={
                    "generation": self._lichess_guidance_generation,
                },
                include_images=True,
            )

    def _begin_lichess_guidance_observation(self) -> None:
        if not self._lichess_guidance_projection_active:
            return
        generation = self._lichess_guidance_generation
        self._lichess_guidance_projection_active = False
        self._lichess_guidance_blackout_pending = True
        self._lichess_guidance_reprojection_pending = True
        self._lichess_guidance_edge_entered = False
        self._projector_window.show_black()
        self._alignment_monitor.set_black()
        self._lichess_blackout_settling_detector.reset()
        self._reference_move_tracker.reset_activity()
        status_message = (
            "Hand left the board; projector returned to black and is settling "
            "before move verification."
        )
        self._projector_status.setText(status_message)
        self._record_runtime_diagnostic(
            "lichess-guidance-blackout-started",
            status_message,
            details={"generation": generation},
            include_images=True,
        )
        QTimer.singleShot(
            1000,
            lambda: self._finish_lichess_guidance_blackout(
                generation,
                "fallback",
            ),
        )

    def _observe_lichess_guidance_blackout(self) -> None:
        if (
            self._latest_frame is None
            or self._calibration_result is None
            or not self._lichess_guidance_blackout_pending
        ):
            return
        try:
            update = self._lichess_blackout_settling_detector.observe(
                self._latest_frame,
                self._calibration_result.mapping.camera_grid,
            )
        except (ValueError, RuntimeError, cv2.error) as error:
            self._projector_status.setText(
                "Adaptive projector-black settling unavailable; waiting for "
                f"the one-second fallback: {error}"
            )
            return
        if update.state in ("stable", "timeout"):
            self._finish_lichess_guidance_blackout(
                self._lichess_guidance_generation,
                update.state,
            )
            return
        self._occupancy_status.setText(
            "Opponent-move guidance removed; waiting for projector black "
            f"to settle ({update.stable_observations}/"
            f"{update.required_observations} stable observations after "
            "400 ms)."
        )

    def _finish_lichess_guidance_blackout(
        self,
        generation: int,
        reason: Literal["stable", "timeout", "fallback"] = "fallback",
    ) -> None:
        if (
            generation != self._lichess_guidance_generation
            or self._lichess_guidance_projection_active
            or not self._lichess_guidance_blackout_pending
        ):
            return
        self._lichess_guidance_blackout_pending = False
        self._reference_move_tracker.reset_activity()
        status_message = (
            "Projector black exposure stabilized; verifying the pending "
            "physical move."
            if reason == "stable"
            else "Projector blackout reached the one-second fallback; "
            "verifying the pending physical move."
        )
        self._projector_status.setText(status_message)
        self._record_runtime_diagnostic(
            "lichess-guidance-verification-ready",
            status_message,
            details={
                "generation": generation,
                "reason": reason,
            },
            include_images=True,
        )

    def _cancel_lichess_guidance_activity(self) -> None:
        self._lichess_guidance_generation += 1
        self._lichess_guidance_projection_active = False
        self._lichess_guidance_blackout_pending = False
        self._lichess_guidance_reprojection_pending = False
        self._lichess_guidance_squares = ()
        self._lichess_guidance_edge_entered = False
        self._lichess_guidance_activity_logged = False
        self._lichess_blackout_settling_detector.reset()

    def _restore_pending_lichess_guidance_after_no_change(
        self,
        update: ReferenceTrackerUpdate,
    ) -> None:
        controller = self._lichess_game_controller
        if (
            update.state != "watching"
            or not self._lichess_guidance_reprojection_pending
            or controller is None
            or controller.expected_signature is None
        ):
            return
        self._update_lichess_synchronization_projection()
        if self._lichess_guidance_projection_active:
            self._occupancy_status.setText(
                "No physical move detected; pending Lichess guidance "
                "restored."
            )

    def _reject_unconfirmed_lichess_move(
        self,
        move: tuple[tuple[int, int], tuple[int, int]],
    ) -> None:
        controller = self._lichess_game_controller
        self._reference_move_tracker.reset_activity()
        self._board_widget.set_guidance_move(None)
        self._last_rejected_move = move
        self._board_widget.set_illegal_source(move[0])
        expected = (
            None if controller is None else controller.expected_signature
        )
        if expected is None:
            message = (
                f"Physical move {square_name(move[0])}-"
                f"{square_name(move[1])} is not yet present in Lichess. "
                "Restore the physical board and make the move in the "
                "Lichess browser first."
            )
        else:
            message = (
                f"Physical move {square_name(move[0])}-"
                f"{square_name(move[1])} does not match pending Lichess move "
                f"{square_name(expected.source)}-"
                f"{square_name(expected.destination)}. Restore the board or "
                "complete the pending move."
            )
        self._piece_status.setText(message)
        self._lichess_sync_status.setText(message)
        self._record_runtime_diagnostic(
            "lichess-physical-move-rejected",
            message,
            include_images=True,
        )
        self._project_illegal_move_correction(move)

    def _project_illegal_move_correction(
        self,
        move: tuple[tuple[int, int], tuple[int, int]],
    ) -> None:
        if (
            self._illegal_correction_projection_active
            or self._illegal_correction_blackout_pending
        ):
            return
        if self._camera_only_calibration:
            self._projector_status.setText(
                "Illegal move marked in red on the UI; physical projection "
                "is unavailable."
            )
            return
        if self._calibration_result is None or not self._open_projector():
            self._projector_status.setText(
                "Illegal move marked in red on the UI, but no calibrated "
                "projector is available."
            )
            return
        source, destination = move
        physical_source = self._physical_board_square(*source)
        physical_destination = self._physical_board_square(*destination)
        try:
            if source == destination:
                self._projector_window.show_illegal_square(*physical_source)
                correction_squares = (physical_source,)
            else:
                self._projector_window.show_correction(
                    physical_source,
                    physical_destination,
                )
                correction_squares = tuple(
                    dict.fromkeys((physical_source, physical_destination))
                )
        except (RuntimeError, ValueError) as error:
            self._projector_status.setText(
                f"Illegal move marked on the UI, but projection failed: "
                f"{error}"
            )
            return
        self._alignment_monitor.set_paused("illegal move indication")
        self._illegal_correction_generation += 1
        self._illegal_correction_projection_active = True
        self._illegal_correction_blackout_pending = False
        self._illegal_correction_physical_squares = correction_squares
        self._illegal_correction_activity_logged = False
        self._illegal_correction_activity_detector.reset()
        status_message = (
            "Projecting the illegally removed square in red until the "
            "restoring hand leaves the board."
            if source == destination
            else "Projecting both illegal move squares in red until the "
            "restoring hand leaves the board."
        )
        self._projector_status.setText(status_message)
        reprojection = self._illegal_correction_last_projected_move == move
        self._illegal_correction_last_projected_move = move
        self._record_runtime_diagnostic(
            (
                "illegal-correction-reprojected"
                if reprojection
                else "illegal-correction-projected"
            ),
            status_message,
            details={
                "generation": self._illegal_correction_generation,
                "logical_move": [
                    square_name(source),
                    square_name(destination),
                ],
                "physical_squares": [
                    [row, column] for row, column in correction_squares
                ],
            },
            include_images=True,
        )

    def _observe_illegal_correction_activity(self) -> None:
        if (
            self._latest_frame is None
            or self._calibration_result is None
            or not self._illegal_correction_projection_active
        ):
            return
        try:
            manipulation_settled = (
                self._illegal_correction_activity_detector.observe(
                    self._latest_frame,
                    self._calibration_result.mapping.camera_grid,
                    self._illegal_correction_physical_squares,
                )
            )
        except (ValueError, RuntimeError, cv2.error) as error:
            self._projector_status.setText(
                f"Correction hand detection unavailable: {error}"
            )
            return
        if manipulation_settled:
            self._begin_illegal_correction_observation()
        elif self._illegal_correction_activity_detector.manipulation_active:
            status_message = (
                "Correction in progress; both red squares remain projected "
                "until the hand leaves the board."
            )
            self._projector_status.setText(status_message)
            if not self._illegal_correction_activity_logged:
                self._illegal_correction_activity_logged = True
                self._record_runtime_diagnostic(
                    "illegal-correction-hand-entered",
                    status_message,
                    details={
                        "generation": self._illegal_correction_generation,
                        "physical_squares": [
                            [row, column]
                            for row, column
                            in self._illegal_correction_physical_squares
                        ],
                    },
                    include_images=True,
                )

    def _begin_illegal_correction_observation(self) -> None:
        if not self._illegal_correction_projection_active:
            return
        generation = self._illegal_correction_generation
        self._illegal_correction_projection_active = False
        self._illegal_correction_blackout_pending = True
        self._illegal_correction_activity_detector.reset()
        self._projector_window.show_black()
        self._alignment_monitor.set_black()
        self._reference_move_tracker.reset_activity()
        status_message = (
            "Hand left the board; red correction removed while projector "
            "black settles."
        )
        self._projector_status.setText(status_message)
        self._record_runtime_diagnostic(
            "illegal-correction-blackout-started",
            status_message,
            details={"generation": generation},
            include_images=True,
        )
        QTimer.singleShot(
            1000,
            lambda: self._finish_illegal_correction_blackout(generation),
        )

    def _finish_illegal_correction_blackout(self, generation: int) -> None:
        if (
            generation != self._illegal_correction_generation
            or not self._illegal_correction_blackout_pending
        ):
            return
        self._illegal_correction_blackout_pending = False
        self._reference_move_tracker.reset_activity()
        status_message = (
            "Projector black settled; verifying the physical correction."
        )
        self._projector_status.setText(status_message)
        self._record_runtime_diagnostic(
            "illegal-correction-verification-ready",
            status_message,
            details={"generation": generation},
            include_images=True,
        )

    def _finish_move_projection(self) -> None:
        if self._projector_window.output_mode != "move":
            return
        self._projector_window.show_black()
        self._alignment_monitor.set_black()
        self._projector_status.setText(
            "Move indication finished; projector returned to black."
        )

    def _toggle_physical_reconciliation(self) -> None:
        if self._reconciliation_requested:
            self._accept_current_physical_reconciliation()
            return
        if (
            self._calibration_result is None
            or not self._piece_synchronizer.initialized
        ):
            QMessageBox.warning(
                self,
                "Physical-board reconciliation",
                "A calibrated board and accepted position are required.",
            )
            return
        self._begin_physical_reconciliation(
            "Reconciliation started; checking stable physical occupancy."
        )

    def _accept_current_physical_reconciliation(self) -> None:
        _target_position, lichess_game = (
            self._physical_reconciliation_target()
        )
        if self._latest_frame is None or self._calibration_result is None:
            QMessageBox.warning(
                self,
                "Stop reconciliation",
                "A calibrated current camera frame is required before the "
                "board can be accepted.",
            )
            return
        if lichess_game is not None and self._lichess_player_color is None:
            QMessageBox.warning(
                self,
                "Stop reconciliation",
                "Lichess game metadata is not available yet. Wait for the "
                "connection update, then try again.",
            )
            return
        self._queue_clean_reconciliation_reference_capture()

    def _queue_clean_reconciliation_reference_capture(self) -> None:
        if not self._reconciliation_requested:
            raise RuntimeError(
                "Clean reference capture requires active reconciliation."
            )
        self._reconciliation_reference_acceptance_pending = True
        self._reconciliation_blackout_generation += 1
        generation = self._reconciliation_blackout_generation
        self._reconciliation_blackout_pending = True
        self._black_projector()
        self._occupancy_detector.reset()
        self._reference_move_tracker.reset_activity()
        self._board_widget.set_reconciliation_squares(())
        self._piece_status.setText(
            "Current position accepted; keep the board clear while a clean "
            "move reference is captured."
        )
        self._occupancy_status.setText(
            "Waiting for projector black and camera exposure to settle."
        )
        self._projector_status.setText(
            "Reconciliation marks removed; clean reference capture pending."
        )
        QTimer.singleShot(
            1000,
            lambda: self._finish_reconciliation_blackout(generation),
        )

    def _begin_physical_reconciliation(
        self,
        occupancy_message: str,
    ) -> None:
        self._reconciliation_requested = True
        self._reconciliation_blackout_pending = True
        self._reconciliation_reference_acceptance_pending = False
        self._reconciliation_blackout_generation += 1
        generation = self._reconciliation_blackout_generation
        self._reconcile_button.setText("Stop reconciliation")
        self._board_widget.set_illegal_source(None)
        self._board_widget.set_guidance_move(None)
        self._board_widget.set_reconciliation_squares(())
        self._last_rejected_move = None
        self._black_projector()
        self._occupancy_detector.reset()
        self._reference_move_tracker.reset_activity()
        self._occupancy_status.setText(occupancy_message)
        self._piece_status.setText(
            "Keep the board clear while mismatching squares are checked."
        )
        QTimer.singleShot(
            1000,
            lambda: self._finish_reconciliation_blackout(generation),
        )

    def _start_automatic_lichess_reconciliation(self) -> None:
        if (
            self._reconciliation_requested
            or self._reset_to_start_requested
            or self._calibration_result is None
            or not self._piece_synchronizer.initialized
            or not self._reference_move_tracker.initialized
        ):
            return
        self._begin_physical_reconciliation(
            "Lichess reconciliation started automatically; checking stable "
            "physical occupancy."
        )

    def _update_physical_reconciliation(
        self,
        physically_occupied: frozenset[tuple[int, int]],
    ) -> None:
        target_position, lichess_game = self._physical_reconciliation_target()
        if getattr(
            self,
            "_reconciliation_reference_acceptance_pending",
            False,
        ):
            self._complete_physical_reconciliation(
                target_position,
                lichess_game,
                "Current physical position accepted; clean move reference "
                "captured.",
            )
            return
        observed_colors: (
            dict[tuple[int, int], PieceColorLabel] | None
        ) = None
        reference = self._reference_move_tracker.reference
        if (
            reference is not None
            and self._latest_frame is not None
            and self._calibration_result is not None
        ):
            current = self._reference_move_tracker.canonical_frame(
                self._latest_frame,
                self._calibration_result.mapping.camera_grid,
                self._board_quarter_turns,
            )
            observed_colors = {
                square: classification.label
                for square, classification in classify_piece_colors(
                    reference,
                    current,
                    self._piece_synchronizer.position,
                ).items()
            }
        result = compare_physical_occupancy(
            target_position,
            physically_occupied,
            observed_colors,
        )
        self._board_widget.set_occupied_squares(physically_occupied)
        self._board_widget.set_reconciliation_squares(
            result.mismatch_squares
        )
        self._piece_status.setText(result.message)
        if result.matches:
            if self._latest_frame is None or self._calibration_result is None:
                raise RuntimeError(
                    "Reconciliation matched without a calibrated frame."
                )
            if lichess_game is not None:
                if self._lichess_player_color is None:
                    self._occupancy_status.setText(
                        "Lichess game metadata is not available yet; "
                        "reconciliation will retry."
                    )
                    return
            self._complete_physical_reconciliation(
                target_position,
                lichess_game,
                "Reconciliation complete; accepted reference replaced.",
            )
            return
        self._occupancy_status.setText(
            "Reconciliation found mismatching physical squares."
        )
        if self._camera_only_calibration:
            self._projector_status.setText(
                "Reconciliation mismatches are listed in the UI; physical "
                "projection is unavailable."
            )
            return
        if self._calibration_result is None or not self._open_projector():
            self._projector_status.setText(
                "Reconciliation mismatches are listed in the UI, but no "
                "calibrated projector is available."
            )
            return
        physical_squares = tuple(
            self._physical_board_square(*square)
            for square in result.mismatch_squares
        )
        try:
            self._projector_window.show_reconciliation(physical_squares)
        except (RuntimeError, ValueError) as error:
            self._projector_status.setText(
                f"Reconciliation projection failed: {error}"
            )
            return
        self._alignment_monitor.set_paused("physical-board reconciliation")
        self._projector_status.setText(
            "Projecting all occupancy mismatches in red for 2 seconds."
        )
        QTimer.singleShot(2000, self._finish_reconciliation_projection)

    def _complete_physical_reconciliation(
        self,
        target_position: Position,
        lichess_game: LichessGameFull | None,
        completion_message: str,
    ) -> None:
        if self._latest_frame is None or self._calibration_result is None:
            raise RuntimeError(
                "Reconciliation completed without a calibrated frame."
            )
        adopted_controller: LichessGameController | None = None
        synchronization_message: str | None = None
        if lichess_game is not None:
            if self._lichess_player_color is None:
                raise RuntimeError(
                    "Reconciliation completed without Lichess player metadata."
                )
            adopted_controller = LichessGameController(
                lichess_game.game_id,
                lichess_game.initial_fen,
                self._lichess_player_color,
                self._lichess_opponent_name,
                physical_fen=lichess_game.state.fen,
            )
            update = adopted_controller.accept_server_history(
                lichess_game.state.moves
            )
            if update.state not in ("synchronized", "game-complete"):
                raise RuntimeError(
                    "The reconciled Lichess history did not attach: "
                    f"{update.message}"
                )
            synchronization_message = update.message
        self._reference_move_tracker.set_reference(
            self._latest_frame,
            self._calibration_result.mapping.camera_grid,
            self._board_quarter_turns,
        )
        if lichess_game is not None and adopted_controller is not None:
            self._piece_synchronizer.restore(target_position)
            self._legal_game = LegalGame(lichess_game.state.fen)
            self._lichess_game_controller = adopted_controller
            self._cancel_offline_recovery_for_lichess()
            self._board_widget.set_pieces(target_position)
            self._board_widget.set_occupied_squares(target_position)
            self._persist_piece_position(target_position)
            if synchronization_message is not None:
                self._lichess_sync_status.setText(synchronization_message)
        self._persist_move_reference()
        self._reconciliation_requested = False
        self._reconciliation_blackout_pending = False
        self._reconciliation_reference_acceptance_pending = False
        self._reconciliation_blackout_generation += 1
        self._reconcile_button.setText("Reconcile physical board")
        self._black_projector()
        self._occupancy_detector.reset()
        self._reference_move_tracker.reset_activity()
        self._board_widget.set_reconciliation_squares(())
        self._occupancy_status.setText(completion_message)
        self._piece_status.setText(
            "Clean move reference captured; make the first move to determine "
            "whose turn it is."
            if (
                self._game_mode == "offline-game"
                and getattr(
                    self,
                    "_offline_turn_inference_pending",
                    False,
                )
            )
            else _accepted_position_message(self._legal_game)
            if self._game_mode == "offline-game"
            else "Physical and UI piece positions are synchronized."
        )

    def _physical_reconciliation_target(
        self,
    ) -> tuple[Position, LichessGameFull | None]:
        game = self._latest_lichess_game
        if game is not None and game.game_id == self._lichess_game_id:
            position, _ = _lichess_board_state(game.state)
            return position, game
        return self._piece_synchronizer.position, None

    def _finish_reconciliation_projection(self) -> None:
        if self._projector_window.output_mode != "reconciliation":
            return
        self._reconciliation_blackout_generation += 1
        generation = self._reconciliation_blackout_generation
        self._reconciliation_blackout_pending = True
        self._projector_window.show_black()
        self._alignment_monitor.set_black()
        self._occupancy_detector.reset()
        self._projector_status.setText(
            "Reconciliation projection finished; waiting for projector black "
            "to settle."
        )
        QTimer.singleShot(
            1000,
            lambda: self._finish_reconciliation_blackout(generation),
        )

    def _finish_reconciliation_blackout(self, generation: int) -> None:
        if (
            generation != self._reconciliation_blackout_generation
            or not self._reconciliation_requested
        ):
            return
        self._reconciliation_blackout_pending = False
        self._occupancy_detector.reset()
        self._projector_status.setText(
            "Projector is black; checking reconciliation occupancy again."
        )

    def _physical_board_square(
        self,
        row: int,
        column: int,
    ) -> tuple[int, int]:
        return rotate_square(
            row,
            column,
            self._board_quarter_turns,
        )

    def _restore_piece_position(self) -> None:
        encoded = str(self._settings.value(_PIECE_POSITION_SETTING, ""))
        if not encoded:
            return
        try:
            entries = json.loads(encoded)
            if not isinstance(entries, list):
                raise ValueError("Saved piece position must be a list.")
            position: Position = {}
            for entry in entries:
                if (
                    not isinstance(entry, list)
                    or len(entry) != 3
                    or not isinstance(entry[0], int)
                    or not isinstance(entry[1], int)
                    or not isinstance(entry[2], str)
                    or entry[2] not in "KQRBNPkqrbnp"
                ):
                    raise ValueError("Saved piece position entry is invalid.")
                position[(entry[0], entry[1])] = cast(PieceCode, entry[2])
            self._piece_synchronizer.restore(position)
            self._board_widget.set_pieces(position)
        except (json.JSONDecodeError, ValueError):
            self._settings.remove(_PIECE_POSITION_SETTING)
            self._settings.sync()

    def _persist_piece_position(self, position: Position) -> None:
        if not position:
            return
        encoded = json.dumps(
            [
                [row, column, piece]
                for (row, column), piece in sorted(position.items())
            ],
            separators=(",", ":"),
        )
        if str(self._settings.value(_PIECE_POSITION_SETTING, "")) == encoded:
            return
        self._settings.setValue(_PIECE_POSITION_SETTING, encoded)
        self._settings.sync()

    def _restore_game_mode(self) -> None:
        saved_mode = str(
            self._settings.value(_GAME_MODE_SETTING, "diagnostic")
        )
        saved_fen = str(self._settings.value(_LEGAL_GAME_FEN_SETTING, ""))
        if saved_mode != "offline-game" or not saved_fen:
            return
        try:
            restored = LegalGame(saved_fen)
        except ValueError as error:
            self._offline_mode_recovery_error = (
                f"The saved Offline legal game is invalid: {error}"
            )
            return
        if (
            not self._piece_synchronizer.initialized
            or restored.position != self._piece_synchronizer.position
        ):
            self._pending_offline_restore_game = restored
            if self._automatic_start_reset:
                self._automatic_start_reset_recovery = True
                self._activate_offline_after_reset = True
            return
        self._legal_game = restored
        self._game_mode = "offline-game"
        self._offline_turn_inference_pending = _stored_boolean(
            self._settings.value(
                _OFFLINE_TURN_INFERENCE_SETTING,
                False,
            ),
            False,
        )
        self._pending_offline_games = (
            _inferred_offline_games(self._piece_synchronizer.position)
            if self._offline_turn_inference_pending
            else None
        )
        self._startup_offline_reference_refresh_pending = True

    def _start_pending_offline_reference_refresh(self) -> None:
        if not self._startup_offline_reference_refresh_pending:
            return
        self._startup_offline_reference_refresh_pending = False
        self._begin_physical_reconciliation(
            "Restored Offline game; preparing a clean move reference."
        )
        self._queue_clean_reconciliation_reference_capture()

    def _persist_game_state(self) -> None:
        self._settings.setValue(_GAME_MODE_SETTING, self._game_mode)
        self._settings.setValue(_LEGAL_GAME_FEN_SETTING, self._legal_game.fen)
        self._settings.setValue(
            _OFFLINE_TURN_INFERENCE_SETTING,
            getattr(self, "_offline_turn_inference_pending", False),
        )
        self._settings.sync()

    def _game_mode_changed(self, index: int) -> None:
        selected = str(self._game_mode_selector.itemData(index))
        if selected == "diagnostic":
            self._activate_board_setup_mode()
            return
        if selected != "offline-game":
            raise ValueError(f"Unknown game mode: {selected}")
        position = self._piece_synchronizer.position
        if position == self._legal_game.position:
            self._game_mode = "offline-game"
            self._sync_game_mode_actions()
            self._persist_game_state()
            self._update_turn_indicator()
            self._piece_status.setText(
                f"Offline legal game enabled; "
                f"{self._legal_game.status.turn} to move."
            )
            return
        self._game_mode_selector.blockSignals(True)
        self._game_mode_selector.setCurrentIndex(
            1 if self._game_mode == "offline-game" else 0
        )
        self._game_mode_selector.blockSignals(False)
        self._sync_game_mode_actions()
        self._update_turn_indicator()
        self._pending_offline_restore_game = LegalGame(
            self._legal_game.fen
        )
        self._offline_mode_recovery_error = None
        if self._automatic_start_reset:
            self._begin_automatic_offline_start_reset()
        else:
            self._show_offline_mode_recovery_dialog()

    def _begin_automatic_offline_start_reset(self) -> None:
        self._automatic_start_reset_recovery = True
        self._activate_offline_after_reset = True
        if (
            self._latest_frame is not None
            and self._calibration_result is not None
        ):
            self._reset_pieces_to_start()
        else:
            self._piece_status.setText(
                "Waiting for a calibrated camera to check for the standard "
                "starting position."
            )

    def _show_offline_mode_recovery_dialog(self) -> None:
        target = self._pending_offline_restore_game
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Icon.Warning)
        dialog.setWindowTitle("Offline game position mismatch")
        explanation = (
            self._offline_mode_recovery_error
            or "The saved Offline legal position does not match the "
            "accepted physical position."
        )
        dialog.setText(explanation)
        dialog.setInformativeText(
            "Choose how to recover. No saved state will be changed until "
            "you select an option."
        )
        restore_button = dialog.addButton(
            "Restore saved offline position",
            QMessageBox.ButtonRole.AcceptRole,
        )
        restore_button.setEnabled(target is not None)
        reset_button = dialog.addButton(
            "Reset to starting position",
            QMessageBox.ButtonRole.ActionRole,
        )
        setup_button = dialog.addButton(
            "Start Offline game from current position",
            QMessageBox.ButtonRole.ActionRole,
        )
        cancel_button = dialog.addButton(
            QMessageBox.StandardButton.Cancel
        )
        dialog.setDefaultButton(cancel_button)
        dialog.exec()
        clicked = dialog.clickedButton()
        if clicked is restore_button and target is not None:
            self._begin_offline_position_restore(target)
        elif clicked is reset_button:
            self._pending_offline_restore_game = None
            self._offline_mode_recovery_error = None
            self._automatic_start_reset_recovery = False
            self._activate_offline_after_reset = True
            if (
                self._latest_frame is not None
                and self._calibration_result is not None
            ):
                self._reset_pieces_to_start()
            else:
                self._piece_status.setText(
                    "Waiting for a calibrated camera before verifying the "
                    "starting position."
                )
        elif clicked is setup_button:
            self._start_offline_game_from_current_position()
        else:
            self._game_mode_selector.blockSignals(True)
            self._game_mode_selector.setCurrentIndex(
                1 if self._game_mode == "offline-game" else 0
            )
            self._game_mode_selector.blockSignals(False)
            self._sync_game_mode_actions()

    def _activate_board_setup_mode(self) -> None:
        self._pending_offline_restore_game = None
        self._offline_mode_recovery_error = None
        self._offline_restore_requested = False
        self._automatic_start_reset_recovery = False
        self._activate_offline_after_reset = False
        self._offline_turn_inference_pending = False
        self._pending_offline_games = None
        self._game_mode = "diagnostic"
        self._game_mode_selector.blockSignals(True)
        self._game_mode_selector.setCurrentIndex(0)
        self._game_mode_selector.blockSignals(False)
        self._sync_game_mode_actions()
        self._board_widget.set_illegal_source(None)
        self._board_widget.set_reconciliation_squares(())
        self._last_rejected_move = None
        self._settings.setValue(_GAME_MODE_SETTING, "diagnostic")
        self._settings.setValue(_OFFLINE_TURN_INFERENCE_SETTING, False)
        self._settings.sync()
        self._update_turn_indicator()
        self._piece_status.setText(
            "Board setup / diagnostic mode accepts unambiguous relocations."
        )

    def _start_offline_game_from_current_position(self) -> None:
        position = self._piece_synchronizer.position
        if not position:
            self._piece_status.setText(
                "Cannot start an Offline game before a physical position "
                "has been accepted."
            )
            return
        is_standard_position = position == standard_position()
        pending_games = (
            None
            if is_standard_position
            else _inferred_offline_games(position)
        )
        self._pending_offline_restore_game = None
        self._offline_mode_recovery_error = None
        self._offline_restore_requested = False
        self._automatic_start_reset_recovery = False
        self._activate_offline_after_reset = False
        self._legal_game = (
            LegalGame() if pending_games is None else pending_games[0]
        )
        self._pending_offline_games = pending_games
        self._offline_turn_inference_pending = pending_games is not None
        self._game_mode = "offline-game"
        self._game_mode_selector.blockSignals(True)
        self._game_mode_selector.setCurrentIndex(1)
        self._game_mode_selector.blockSignals(False)
        self._sync_game_mode_actions()
        self._board_widget.set_illegal_source(None)
        self._board_widget.set_reconciliation_squares(())
        self._last_rejected_move = None
        self._reference_move_tracker.reset_activity()
        self._persist_game_state()
        self._update_turn_indicator()
        self._begin_physical_reconciliation(
            "Current Offline position accepted; preparing a clean move "
            "reference."
        )
        self._queue_clean_reconciliation_reference_capture()

    def _begin_offline_position_restore(self, target: LegalGame) -> None:
        self._pending_offline_restore_game = None
        self._offline_mode_recovery_error = None
        self._automatic_start_reset_recovery = False
        self._activate_offline_after_reset = False
        self._offline_turn_inference_pending = False
        self._pending_offline_games = None
        self._legal_game = target
        self._game_mode = "offline-game"
        self._offline_restore_requested = True
        self._game_mode_selector.blockSignals(True)
        self._game_mode_selector.setCurrentIndex(1)
        self._game_mode_selector.blockSignals(False)
        self._sync_game_mode_actions()
        self._persist_game_state()
        self._update_turn_indicator()
        self._black_projector()
        self._occupancy_detector.reset()
        self._piece_status.setText(
            "Restore the physical board to the saved Offline legal position."
        )
        self._occupancy_status.setText(
            "Checking physical occupancy against the saved offline game."
        )

    def _cancel_offline_recovery_for_lichess(self) -> None:
        if not (
            self._reset_to_start_requested
            or self._activate_offline_after_reset
            or self._automatic_start_reset_recovery
            or self._offline_restore_requested
            or self._pending_offline_restore_game is not None
        ):
            return
        self._reset_to_start_requested = False
        self._activate_offline_after_reset = False
        self._automatic_start_reset_recovery = False
        self._offline_restore_requested = False
        self._pending_offline_restore_game = None
        self._offline_mode_recovery_error = None
        self._occupancy_detector.reset()
        self._reference_move_tracker.reset_activity()
        self._record_runtime_diagnostic(
            "lichess-offline-recovery-cancelled",
            "Active Lichess synchronization cancelled stale Offline recovery.",
        )

    def _update_offline_position_restore(
        self,
        occupied: frozenset[tuple[int, int]],
    ) -> None:
        target_position = self._legal_game.position
        target_occupied = frozenset(target_position)
        discrepancies = tuple(sorted(occupied ^ target_occupied))
        self._board_widget.set_reconciliation_squares(discrepancies)
        if discrepancies:
            names = ", ".join(square_name(square) for square in discrepancies)
            self._piece_status.setText(
                f"Restore the saved Offline legal position; check {names}."
            )
            self._occupancy_status.setText(
                "Saved offline position does not yet match the physical board."
            )
            return
        if self._latest_frame is None or self._calibration_result is None:
            raise RuntimeError(
                "Offline restoration completed without a calibrated frame."
            )
        self._piece_synchronizer.restore(target_position)
        self._reference_move_tracker.set_reference(
            self._latest_frame,
            self._calibration_result.mapping.camera_grid,
            self._board_quarter_turns,
        )
        self._offline_restore_requested = False
        self._board_widget.set_pieces(target_position)
        self._board_widget.set_occupied_squares(target_position)
        self._board_widget.set_reconciliation_squares(())
        self._persist_piece_position(target_position)
        self._persist_move_reference()
        self._persist_game_state()
        self._piece_status.setText(
            f"Offline legal game restored; "
            f"{self._legal_game.status.turn} to move."
        )
        self._occupancy_status.setText(
            "Saved offline position confirmed; move reference replaced."
        )

    def _toggle_lichess_connection(self) -> None:
        if self._lichess_worker is not None:
            self._lichess_disconnect_requested = True
            self._lichess_connection_button.setEnabled(False)
            self._lichess_connection_button.setText("Disconnecting...")
            self._lichess_status.setText(
                "Disconnecting from Lichess after the current keep-alive."
            )
            self._lichess_worker.request_stop()
            return
        try:
            client = LichessReadOnlyClient.from_environment()
        except LichessConfigurationError as error:
            self._lichess_status.setText(str(error))
            return
        thread = QThread(self)
        worker = LichessConnectionWorker(client)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.status_changed.connect(
            self._lichess_connection_status_changed
        )
        worker.account_event_received.connect(
            self._lichess_account_event_received
        )
        worker.game_event_received.connect(
            self._lichess_game_event_received
        )
        worker.ongoing_games_received.connect(
            self._lichess_ongoing_games_received
        )
        worker.failed.connect(self._lichess_connection_failed)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(self._lichess_thread_finished)
        thread.finished.connect(thread.deleteLater)
        self._lichess_thread = thread
        self._lichess_worker = worker
        self._lichess_client = client
        self._lichess_disconnect_requested = False
        self._lichess_connection_button.setText("Connecting...")
        self._lichess_connection_button.setEnabled(False)
        self._lichess_status.setText("Connecting to Lichess...")
        self._record_runtime_diagnostic(
            "lichess-connect-requested",
            "Connecting to Lichess.",
        )
        thread.start()

    def _start_new_lichess_game(self) -> None:
        if not _can_start_new_lichess_game(self._latest_lichess_state):
            self._lichess_status.setText(
                "Finish the active Lichess game before starting a new one."
            )
            return
        if self._lichess_ai_creation_thread is not None:
            return
        if self._pending_created_lichess_game_id is not None:
            return
        stored_color = str(
            self._settings.value(_LICHESS_AI_COLOR_SETTING, "white")
        )
        color: AiChallengeColor = (
            cast(AiChallengeColor, stored_color)
            if stored_color in ("white", "black", "random")
            else "white"
        )
        dialog = LichessAiGameDialog(
            level=_stored_integer(
                self._settings.value(_LICHESS_AI_LEVEL_SETTING, 4),
                4,
            ),
            color=color,
            clock_key=str(
                self._settings.value(_LICHESS_AI_CLOCK_SETTING, "10+5")
            ),
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        level, color, _clock_key, clock_limit, clock_increment = (
            dialog.configuration
        )
        if self._lichess_worker is None:
            self._toggle_lichess_connection()
            if self._lichess_worker is None:
                return
        client = self._lichess_client
        if client is None:
            raise RuntimeError(
                "Lichess connected without an authenticated client."
            )
        thread = QThread(self)
        worker = LichessAiGameCreationWorker(
            client,
            level=level,
            color=color,
            clock_limit_seconds=clock_limit,
            clock_increment_seconds=clock_increment,
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.created.connect(self._lichess_ai_game_created)
        worker.rejected.connect(self._lichess_ai_game_creation_rejected)
        worker.uncertain.connect(self._lichess_ai_game_creation_uncertain)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(self._lichess_ai_creation_finished)
        thread.finished.connect(thread.deleteLater)
        self._lichess_ai_creation_thread = thread
        self._lichess_ai_creation_worker = worker
        self._start_new_lichess_game_button.setEnabled(False)
        self._lichess_status.setText(
            f"Creating Lichess AI level {level} game..."
        )
        thread.start()

    def _lichess_ai_game_created(self, game_id: str) -> None:
        self._pending_created_lichess_game_id = game_id
        self._remember_lichess_game_selection(game_id)
        if self._lichess_worker is not None:
            self._lichess_worker.select_game(game_id)
        message = (
            f"Lichess AI game {game_id} created; waiting for its snapshot."
        )
        self._lichess_status.setText(message)
        self._record_runtime_diagnostic(
            "lichess-ai-game-created",
            message,
            details={"game_id": game_id},
        )

    def _lichess_ai_game_creation_rejected(self, message: str) -> None:
        self._pending_created_lichess_game_id = None
        self._lichess_status.setText(message)
        self._record_runtime_diagnostic(
            "lichess-ai-game-creation-rejected",
            message,
        )

    def _lichess_ai_game_creation_uncertain(self, message: str) -> None:
        self._pending_created_lichess_game_id = None
        status = f"Lichess AI game creation outcome is uncertain. {message}"
        self._lichess_status.setText(status)
        self._record_runtime_diagnostic(
            "lichess-ai-game-creation-uncertain",
            status,
        )

    def _lichess_ai_creation_finished(self) -> None:
        self._lichess_ai_creation_thread = None
        self._lichess_ai_creation_worker = None
        self._start_new_lichess_game_button.setEnabled(
            self._pending_created_lichess_game_id is None
            and _can_start_new_lichess_game(self._latest_lichess_state)
        )
        if self._close_pending:
            QTimer.singleShot(0, self.close)

    def _offer_lichess_draw(self) -> None:
        self._start_lichess_game_action("offer-draw")

    def _accept_lichess_draw(self) -> None:
        self._start_lichess_game_action("accept-draw")

    def _decline_lichess_draw(self) -> None:
        self._start_lichess_game_action("decline-draw")

    def _confirm_lichess_resignation(self) -> None:
        answer = QMessageBox.question(
            self,
            "Resign Lichess game",
            "Resign the current Lichess game?",
            QMessageBox.StandardButton.Yes
            | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self._start_lichess_game_action("resign")

    def _confirm_lichess_abort(self) -> None:
        answer = QMessageBox.question(
            self,
            "Abort Lichess game",
            "Abort the current Lichess game?",
            QMessageBox.StandardButton.Yes
            | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self._start_lichess_game_action("abort")

    def _start_lichess_game_action(
        self,
        action: LichessGameAction,
    ) -> None:
        if (
            self._lichess_client is None
            or self._lichess_game_id is None
            or _can_start_new_lichess_game(self._latest_lichess_state)
            or self._lichess_game_action_thread is not None
        ):
            self._lichess_status.setText(
                "The Lichess game control is not available."
            )
            return
        thread = QThread(self)
        worker = LichessGameActionWorker(
            self._lichess_client,
            self._lichess_game_id,
            action,
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.succeeded.connect(self._lichess_game_action_succeeded)
        worker.rejected.connect(self._lichess_game_action_rejected)
        worker.uncertain.connect(self._lichess_game_action_uncertain)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(self._lichess_game_action_finished)
        thread.finished.connect(thread.deleteLater)
        self._lichess_game_action_thread = thread
        self._lichess_game_action_worker = worker
        self._pending_lichess_game_action = action
        self._update_lichess_game_control_buttons()
        description = "draw offer" if action == "offer-draw" else "resignation"
        self._lichess_status.setText(
            f"Submitting Lichess {description}..."
        )
        thread.start()

    def _lichess_game_action_succeeded(
        self,
        game_id: str,
        action: str,
    ) -> None:
        if action == "offer-draw":
            self._draw_offer_game_id = game_id
            message = "Lichess draw offer submitted."
        elif action == "accept-draw":
            self._responded_draw_offer_game_id = game_id
            message = "Lichess draw offer accepted; awaiting final game state."
        elif action == "decline-draw":
            self._responded_draw_offer_game_id = game_id
            message = "Lichess draw offer declined."
        elif action == "abort":
            self._abort_requested_game_id = game_id
            message = "Lichess abort submitted; awaiting final game state."
        else:
            self._resign_requested_game_id = game_id
            message = (
                "Lichess resignation submitted; awaiting final game state."
            )
        self._pending_lichess_game_action = None
        self._lichess_status.setText(message)
        self._record_runtime_diagnostic(
            "lichess-game-action-succeeded",
            message,
            details={"game_id": game_id, "action": action},
        )
        self._update_lichess_game_control_buttons()

    def _lichess_game_action_rejected(
        self,
        game_id: str,
        action: str,
        message: str,
    ) -> None:
        self._pending_lichess_game_action = None
        self._lichess_status.setText(message)
        self._record_runtime_diagnostic(
            "lichess-game-action-rejected",
            message,
            details={"game_id": game_id, "action": action},
        )
        self._update_lichess_game_control_buttons()

    def _lichess_game_action_uncertain(
        self,
        game_id: str,
        action: str,
        message: str,
    ) -> None:
        self._pending_lichess_game_action = None
        self._lichess_status.setText(
            f"Lichess {action} outcome is uncertain. {message}"
        )
        self._record_runtime_diagnostic(
            "lichess-game-action-uncertain",
            message,
            details={"game_id": game_id, "action": action},
        )
        self._update_lichess_game_control_buttons()

    def _update_lichess_game_control_buttons(self) -> None:
        active = (
            self._lichess_client is not None
            and self._lichess_game_id is not None
            and self._latest_lichess_state is not None
            and not _can_start_new_lichess_game(self._latest_lichess_state)
        )
        idle = (
            self._lichess_game_action_thread is None
            and self._pending_lichess_game_action is None
        )
        own_draw_offer = False
        opponent_draw_offer = False
        if (
            self._lichess_draw_offer_state_available
            and self._latest_lichess_state is not None
            and self._lichess_player_color is not None
        ):
            own_draw_offer, opponent_draw_offer = (
                _lichess_draw_offer_flags(
                    self._latest_lichess_state,
                    self._lichess_player_color,
                )
            )
        self._lichess_draw_button.setVisible(not opponent_draw_offer)
        self._lichess_accept_draw_button.setVisible(opponent_draw_offer)
        self._lichess_decline_draw_button.setVisible(opponent_draw_offer)
        self._lichess_draw_button.setEnabled(
            active
            and idle
            and not own_draw_offer
            and self._draw_offer_game_id != self._lichess_game_id
        )
        self._lichess_accept_draw_button.setEnabled(
            active
            and idle
            and opponent_draw_offer
            and self._responded_draw_offer_game_id != self._lichess_game_id
        )
        self._lichess_decline_draw_button.setEnabled(
            active
            and idle
            and opponent_draw_offer
            and self._responded_draw_offer_game_id != self._lichess_game_id
        )
        self._lichess_abort_button.setEnabled(
            active
            and idle
            and _can_abort_lichess_game(self._latest_lichess_game)
            and self._abort_requested_game_id != self._lichess_game_id
        )
        self._lichess_resign_button.setEnabled(
            active
            and idle
            and self._resign_requested_game_id != self._lichess_game_id
        )

    def _update_authoritative_draw_offer_state(
        self,
        state: LichessGameState,
    ) -> None:
        if (
            not self._lichess_draw_offer_state_available
            or self._lichess_player_color is None
            or self._lichess_game_id is None
        ):
            return
        own_draw_offer, opponent_draw_offer = _lichess_draw_offer_flags(
            state,
            self._lichess_player_color,
        )
        if own_draw_offer:
            self._draw_offer_game_id = self._lichess_game_id
        elif self._pending_lichess_game_action != "offer-draw":
            self._draw_offer_game_id = None
        if not opponent_draw_offer:
            self._responded_draw_offer_game_id = None

    def _mark_lichess_game_finished(self, game_id: str | None) -> None:
        if (
            game_id is not None
            and self._lichess_game_id is not None
            and game_id != self._lichess_game_id
        ):
            return
        state = self._latest_lichess_state
        if state is None or _can_start_new_lichess_game(state):
            return
        finished_state = replace(state, status="finished")
        self._latest_lichess_state = finished_state
        if (
            self._latest_lichess_game is not None
            and self._latest_lichess_game.game_id == self._lichess_game_id
        ):
            self._latest_lichess_game = replace(
                self._latest_lichess_game,
                state=finished_state,
            )

    def _lichess_ongoing_games_received(self, value: object) -> None:
        if not isinstance(value, tuple) or not all(
            isinstance(game, OngoingGame) for game in value
        ):
            raise TypeError("Ongoing Lichess games must be a tuple.")
        self._show_lichess_connected()
        games = cast(tuple[OngoingGame, ...], value)
        remembered = self._preferred_lichess_game_id
        self._lichess_game_selector.blockSignals(True)
        self._lichess_game_selector.clear()
        if not games:
            self._mark_lichess_game_finished(self._lichess_game_id)
            self._last_lichess_game_prompt_ids = None
            self._lichess_game_selector.addItem(
                "No ongoing games",
                None,
            )
            self._lichess_game_selector.setEnabled(False)
            selected: str | None = None
            if (
                self._latest_lichess_game is None
                or not _can_start_new_lichess_game(
                    self._latest_lichess_game.state
                )
            ):
                self._clear_lichess_game_information()
            self._start_new_lichess_game_button.setEnabled(
                self._lichess_ai_creation_thread is None
                and self._pending_created_lichess_game_id is None
            )
        elif len(games) == 1:
            self._last_lichess_game_prompt_ids = None
            game = games[0]
            if remembered and remembered != game.game_id:
                self._lichess_game_selector.addItem(
                    "Previously selected game is not ongoing",
                    None,
                )
                self._lichess_game_selector.addItem(
                    _ongoing_game_label(game),
                    game.game_id,
                )
                self._lichess_game_selector.setEnabled(True)
                selected = None
            else:
                self._lichess_game_selector.addItem(
                    _ongoing_game_label(game),
                    game.game_id,
                )
                self._lichess_game_selector.setEnabled(False)
                selected = game.game_id
                if not remembered:
                    self._remember_lichess_game_selection(game.game_id)
        else:
            self._lichess_game_selector.addItem(
                "Select an ongoing game...",
                None,
            )
            for game in games:
                self._lichess_game_selector.addItem(
                    _ongoing_game_label(game),
                    game.game_id,
                )
            matching_index = self._lichess_game_selector.findData(
                remembered
            )
            if matching_index > 0:
                self._lichess_game_selector.setCurrentIndex(
                    matching_index
                )
                selected = remembered
            else:
                selected = None
                game_ids = frozenset(game.game_id for game in games)
                if self._last_lichess_game_prompt_ids != game_ids:
                    self._last_lichess_game_prompt_ids = game_ids
                    selected = self._choose_lichess_game(games)
                    if selected is not None:
                        self._lichess_game_selector.setCurrentIndex(
                            self._lichess_game_selector.findData(selected)
                        )
                        self._remember_lichess_game_selection(selected)
            self._lichess_game_selector.setEnabled(True)
        self._lichess_game_selector.blockSignals(False)
        self._rebuild_lichess_games_menu()
        if self._lichess_worker is not None:
            self._lichess_worker.select_game(selected)

    def _choose_lichess_game(
        self,
        games: tuple[OngoingGame, ...],
    ) -> str | None:
        dialog = LichessGameSelectionDialog(games, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        return dialog.selected_game_id

    def _lichess_game_selection_changed(self, index: int) -> None:
        game_id = self._lichess_game_selector.itemData(index)
        if self._lichess_worker is not None:
            self._lichess_worker.select_game(
                game_id if isinstance(game_id, str) else None
            )
        if isinstance(game_id, str):
            self._remember_lichess_game_selection(game_id)
            self._lichess_status.setText(
                f"Selected Lichess game {game_id}; awaiting snapshot."
            )
            if game_id != self._lichess_game_id:
                self._clear_lichess_game_information(
                    "Loading selected Lichess game information."
                )
        self._rebuild_lichess_games_menu()

    def _remember_lichess_game_selection(self, game_id: str) -> None:
        self._preferred_lichess_game_id = game_id
        self._settings.setValue(_LICHESS_GAME_SETTING, game_id)
        self._settings.sync()

    def _lichess_connection_status_changed(self, message: str) -> None:
        if message.startswith("Connected to Lichess"):
            self._show_lichess_connected()
        if (
            message
            == "Connected to Lichess by polling; waiting for a game."
            and self._latest_lichess_game is not None
            and _can_start_new_lichess_game(
                self._latest_lichess_game.state
            )
            and self._latest_lichess_game.state.result != "*"
        ):
            message = _lichess_polling_snapshot_message(
                LichessPollingSnapshot(
                    self._latest_lichess_game,
                    self._lichess_clocks_available,
                )
            )
        self._lichess_status.setText(message)
        self._record_runtime_diagnostic(
            "lichess-connection-status",
            message,
        )

    def _show_lichess_connected(self) -> None:
        if self._lichess_disconnect_requested:
            return
        self._lichess_connection_button.setText("Disconnect")
        self._lichess_connection_button.setEnabled(True)

    def _lichess_connection_failed(self, message: str) -> None:
        self._lichess_status.setText(message)
        self._record_runtime_diagnostic(
            "lichess-connection-failed",
            message,
        )

    def _lichess_account_event_received(self, event: object) -> None:
        if not isinstance(event, AccountGameEvent):
            return
        if event.event_type == "gameStart":
            if event.game_id == self._pending_created_lichess_game_id:
                self._pending_created_lichess_game_id = None
            if event.game_id != self._lichess_game_id:
                self._lichess_game_controller = None
                self._latest_lichess_game = None
                self._last_projected_lichess_uci = None
                self._draw_offer_game_id = None
                self._responded_draw_offer_game_id = None
                self._lichess_draw_offer_state_available = False
                self._abort_requested_game_id = None
                self._resign_requested_game_id = None
                self._clear_lichess_game_information(
                    "Loading Lichess game information."
                )
            self._lichess_game_id = event.game_id
            self._lichess_player_color = event.color
            self._lichess_opponent_name = event.opponent_name
            self._start_new_lichess_game_button.setEnabled(False)
            self._update_lichess_game_control_buttons()
            self._lichess_status.setText(
                f"Lichess game {event.game_id} started as "
                f"{event.color} against {event.opponent_name}; loading state."
            )
        else:
            if event.game_id == self._lichess_game_id:
                self._mark_lichess_game_finished(event.game_id)
                self._start_new_lichess_game_button.setEnabled(True)
            self._update_lichess_game_control_buttons()
            self._lichess_status.setText(
                f"Lichess game {event.game_id} finished."
            )
        self._record_runtime_diagnostic(
            "lichess-account-event",
            f"{event.event_type} for game {event.game_id}.",
        )

    def _lichess_game_event_received(self, event: object) -> None:
        if isinstance(event, LichessPollingSnapshot):
            if (
                event.game.game_id
                == self._pending_created_lichess_game_id
            ):
                self._pending_created_lichess_game_id = None
            self._lichess_draw_offer_state_available = False
            self._update_lichess_board(event.game.state)
            self._update_lichess_synchronization(event.game)
            self._update_lichess_game_information(
                event.game,
                clocks_available=event.clocks_available,
            )
            self._start_new_lichess_game_button.setEnabled(
                _can_start_new_lichess_game(event.game.state)
            )
            self._update_lichess_game_control_buttons()
            self._lichess_status.setText(
                _lichess_polling_snapshot_message(event)
            )
        elif isinstance(event, LichessGameFull):
            if event.game_id == self._pending_created_lichess_game_id:
                self._pending_created_lichess_game_id = None
            self._lichess_draw_offer_state_available = True
            self._update_lichess_board(event.state)
            self._update_lichess_synchronization(event)
            self._update_authoritative_draw_offer_state(event.state)
            self._update_lichess_game_information(
                event,
                clocks_available=True,
            )
            self._start_new_lichess_game_button.setEnabled(
                _can_start_new_lichess_game(event.state)
            )
            self._update_lichess_game_control_buttons()
            self._lichess_status.setText(
                f"Lichess game {event.game_id}: "
                f"{event.white.name} (White) vs "
                f"{event.black.name} (Black). "
                f"{_lichess_game_state_message(event.state)}"
            )
        elif isinstance(event, LichessGameState):
            self._update_lichess_board(event)
            self._update_lichess_synchronization_state(event)
            self._update_authoritative_draw_offer_state(event)
            if self._latest_lichess_game is not None:
                self._update_lichess_game_information(
                    self._latest_lichess_game,
                    clocks_available=self._lichess_clocks_available,
                )
            self._start_new_lichess_game_button.setEnabled(
                _can_start_new_lichess_game(event)
            )
            self._update_lichess_game_control_buttons()
            self._lichess_status.setText(
                _lichess_game_state_message(event)
            )
        else:
            return
        self._record_runtime_diagnostic(
            "lichess-game-state",
            self._lichess_status.text(),
        )

    def _update_lichess_board(self, state: LichessGameState) -> None:
        self._latest_lichess_state = state
        if (
            self._latest_lichess_game is not None
            and self._latest_lichess_game.game_id == self._lichess_game_id
        ):
            self._latest_lichess_game = replace(
                self._latest_lichess_game,
                state=state,
            )
        position, last_move = _lichess_board_state(state)
        self._lichess_board_widget.set_pieces(position)
        self._lichess_board_widget.set_occupied_squares(position)
        self._lichess_board_widget.set_last_move(last_move)
        self._update_turn_indicator()

    def _clear_lichess_game_information(
        self,
        status: str = "No active Lichess game information.",
    ) -> None:
        self._lichess_clocks_available = False
        self._lichess_clock_snapshot_monotonic = None
        self._projector_window.set_edge_clocks(
            None,
            None,
            board_quarter_turns=self._board_quarter_turns,
        )
        self._lichess_black_player.setText("Black: --")
        self._lichess_black_clock.setText("Black clock: --:--")
        self._lichess_white_player.setText("White: --")
        self._lichess_white_clock.setText("White clock: --:--")
        self._lichess_game_information_status.setText(status)
        action = getattr(self, "_current_lichess_game_action", None)
        if isinstance(action, QAction):
            action.setEnabled(self._latest_lichess_game is not None)

    def _update_lichess_game_information(
        self,
        game: LichessGameFull,
        *,
        clocks_available: bool,
    ) -> None:
        self._lichess_clocks_available = clocks_available
        self._lichess_clock_snapshot_monotonic = (
            monotonic()
            if clocks_available
            and not _can_start_new_lichess_game(game.state)
            else None
        )
        (
            black_player,
            black_clock,
            white_player,
            white_clock,
            status,
        ) = _lichess_game_information(
            game,
            clocks_available=clocks_available,
        )
        self._lichess_black_player.setText(black_player)
        self._lichess_black_clock.setText(black_clock)
        self._lichess_white_player.setText(white_player)
        self._lichess_white_clock.setText(white_clock)
        self._lichess_game_information_status.setText(status)
        self._update_projected_lichess_clocks(
            self._displayed_lichess_game()
        )
        action = getattr(self, "_current_lichess_game_action", None)
        if isinstance(action, QAction):
            action.setEnabled(True)

    def _displayed_lichess_game(self) -> LichessGameFull | None:
        game = self._latest_lichess_game
        baseline = self._lichess_clock_snapshot_monotonic
        if game is None or baseline is None:
            return game
        elapsed_ms = int(max(0.0, monotonic() - baseline) * 1000)
        return replace(
            game,
            state=_displayed_lichess_clock_state(
                game.state,
                elapsed_ms=elapsed_ms,
                clocks_available=self._lichess_clocks_available,
            ),
        )

    def _refresh_lichess_clock_display(self) -> None:
        game = self._displayed_lichess_game()
        if game is None or self._lichess_clock_snapshot_monotonic is None:
            return
        information = _lichess_game_information(
            game,
            clocks_available=self._lichess_clocks_available,
        )
        self._lichess_black_clock.setText(information[1])
        self._lichess_white_clock.setText(information[3])
        self._update_projected_lichess_clocks(game)

    def _update_projected_lichess_clocks(
        self,
        game: LichessGameFull | None,
    ) -> None:
        timeout_result = (
            game is not None
            and game.state.status == "outoftime"
            and game.state.winner in ("white", "black")
        )
        agreed_draw = (
            game is not None
            and game.state.status == "draw"
            and game.state.result == "1/2-1/2"
        )
        clocks_visible = (
            game is not None
            and self._lichess_clocks_available
            and (
                self._lichess_clock_snapshot_monotonic is not None
                or timeout_result
                or agreed_draw
            )
            and self._calibration_result is not None
            and not self._board_orientation_needs_detection
            and not self._piece_side_orientation_needs_detection
        )
        if not clocks_visible or game is None:
            self._projector_window.set_edge_clocks(
                None,
                None,
                board_quarter_turns=self._board_quarter_turns,
            )
            return
        white_time_ms, black_time_ms = _lichess_display_clock_times(
            game.state
        )
        result: Literal["1-0", "0-1", "1/2-1/2"] | None = None
        expired_player: PlayerColor | None = None
        if timeout_result:
            result = "1-0" if game.state.winner == "white" else "0-1"
            expired_player = (
                "black" if game.state.winner == "white" else "white"
            )
        elif agreed_draw:
            result = "1/2-1/2"
        self._projector_window.set_edge_clocks(
            _format_projected_lichess_clock(white_time_ms),
            _format_projected_lichess_clock(black_time_ms),
            board_quarter_turns=self._board_quarter_turns,
            result=result,
            expired_player=expired_player,
        )

    def _update_lichess_synchronization(
        self,
        game: LichessGameFull,
    ) -> None:
        self._latest_lichess_game = game
        if (
            self._lichess_game_id != game.game_id
            or self._lichess_player_color is None
        ):
            self._lichess_sync_status.setText(
                "Lichess physical synchronization is waiting for account "
                "game metadata."
            )
            return
        self._cancel_offline_recovery_for_lichess()
        if (
            self._lichess_game_controller is None
            or self._lichess_game_controller.game_id != game.game_id
        ):
            initial_position = (
                LegalGame()
                if game.initial_fen == "startpos"
                else LegalGame(game.initial_fen)
            ).position
            current_game = LegalGame(game.state.fen)
            accepted_position = self._piece_synchronizer.position
            if (
                not self._reference_move_tracker.initialized
                or accepted_position
                not in (initial_position, current_game.position)
            ):
                self._lichess_sync_status.setText(
                    "Physical reconciliation required: compare the live "
                    "board with the current Lichess position."
                )
                self._record_runtime_diagnostic(
                    "lichess-reconciliation-required",
                    self._lichess_sync_status.text(),
                    include_images=True,
                )
                self._start_automatic_lichess_reconciliation()
                return
            self._lichess_game_controller = LichessGameController(
                game.game_id,
                game.initial_fen,
                self._lichess_player_color,
                self._lichess_opponent_name,
                physical_fen=(
                    current_game.fen
                    if accepted_position == current_game.position
                    else None
                ),
            )
        self._update_lichess_synchronization_state(game.state)

    def _update_lichess_synchronization_state(
        self,
        state: LichessGameState,
    ) -> None:
        controller = self._lichess_game_controller
        if controller is None:
            return
        if (
            self._pending_lichess_local_uci is not None
            and state.moves
            == (
                *controller.server_moves,
                self._pending_lichess_local_uci,
            )
        ):
            self._lichess_sync_status.setText(
                "Lichess received the submitted move; awaiting the "
                "submission response."
            )
            return
        update = controller.accept_server_history(state.moves)
        self._lichess_sync_status.setText(update.message)
        self._record_runtime_diagnostic(
            "lichess-synchronization",
            update.message,
        )
        if update.state == "reconciliation-required":
            self._start_automatic_lichess_reconciliation()
            return
        signature = controller.expected_signature
        if signature is None:
            self._last_projected_lichess_uci = None
            return
        if signature.uci == self._last_projected_lichess_uci:
            return
        self._piece_status.setText(update.message)
        projected = self._project_lichess_move_guidance(
            (signature.source, signature.destination),
            ()
            if signature.kind == "castling"
            else tuple(
                change.square
                for change in signature.changes
                if change.square
                not in (signature.source, signature.destination)
            ),
        )
        self._last_projected_lichess_uci = (
            signature.uci if projected else None
        )

    def _record_runtime_diagnostic(
        self,
        event_type: str,
        message: str,
        *,
        details: dict[str, object] | None = None,
        tracker_update: ReferenceTrackerUpdate | None = None,
        include_images: bool = False,
        additional_images: dict[str, ImageArray] | None = None,
    ) -> None:
        controller = self._lichess_game_controller
        controller_state: dict[str, object] | None = None
        if controller is not None:
            controller_state = {
                "game_id": controller.game_id,
                "state": controller.state,
                "server_fen": controller.server_fen,
                "physical_fen": controller.physical_fen,
                "server_moves": list(controller.server_moves),
                "pending_moves": list(controller.pending_moves),
                "server_position": _diagnostic_position(
                    controller.server_position
                ),
                "controller_physical_position": _diagnostic_position(
                    controller.physical_position
                ),
            }
        tracker_state: dict[str, object] = {
            "initialized": self._reference_move_tracker.initialized,
            "last_state": self._last_runtime_tracker_state,
            "board_edge_indicator": {
                "active": self._board_edge_indicator_active,
                "active_seconds": (
                    None
                    if self._board_edge_indicator_active_since is None
                    else round(
                        max(
                            0.0,
                            monotonic()
                            - self._board_edge_indicator_active_since,
                        ),
                        3,
                    )
                ),
                "crossing_detected": (
                    self._board_edge_activity_detector.crossing_detected
                ),
                "largest_component_pixels": (
                    self._board_edge_activity_detector
                    .largest_component_pixels
                ),
                "largest_component_inside_pixels": (
                    self._board_edge_activity_detector
                    .largest_component_inside_pixels
                ),
                "largest_component_outside_pixels": (
                    self._board_edge_activity_detector
                    .largest_component_outside_pixels
                ),
                "largest_component_outer_halo_pixels": (
                    self._board_edge_activity_detector
                    .largest_component_outer_halo_pixels
                ),
                "largest_component_bounds": (
                    None
                    if (
                        self._board_edge_activity_detector
                        .largest_component_bounds
                        is None
                    )
                    else list(
                        self._board_edge_activity_detector
                        .largest_component_bounds
                    )
                ),
            },
            "lichess_guidance": {
                "projection_active": (
                    self._lichess_guidance_projection_active
                ),
                "blackout_pending": self._lichess_guidance_blackout_pending,
                "reprojection_pending": (
                    self._lichess_guidance_reprojection_pending
                ),
                "generation": self._lichess_guidance_generation,
                "physical_squares": [
                    [row, column]
                    for row, column in self._lichess_guidance_squares
                ],
                "detector_manipulation_active": (
                    self._lichess_guidance_edge_entered
                ),
                "lifecycle_state": (
                    "manipulating"
                    if self._board_edge_indicator_active
                    else "arming"
                ),
                "boundary_active": self._board_edge_indicator_active,
                "edge_entry_confirmed": (
                    self._lichess_guidance_edge_entered
                ),
            },
            "illegal_correction": {
                "projection_active": (
                    self._illegal_correction_projection_active
                ),
                "blackout_pending": self._illegal_correction_blackout_pending,
                "generation": self._illegal_correction_generation,
                "physical_squares": [
                    [row, column]
                    for row, column
                    in self._illegal_correction_physical_squares
                ],
                "detector_manipulation_active": (
                    self._illegal_correction_activity_detector
                    .manipulation_active
                ),
            },
        }
        if tracker_update is not None:
            tracker_state.update(
                {
                    "state": tracker_update.state,
                    "move": (
                        None
                        if tracker_update.move is None
                        else [
                            square_name(tracker_update.move[0]),
                            square_name(tracker_update.move[1]),
                        ]
                    ),
                    "message": tracker_update.message,
                    "decision_details": tracker_update.decision_details,
                }
            )
        active_camera: dict[str, object] | None = None
        if (
            self._camera is not None
            and 0 <= self._active_device_index < len(self._devices)
        ):
            active_device = self._devices[self._active_device_index]
            camera_format = self._camera.cameraFormat()
            camera_resolution = camera_format.resolution()
            active_camera = {
                "name": active_device.description(),
                "id": self._device_key(active_device),
                "width": camera_resolution.width(),
                "height": camera_resolution.height(),
                "maximum_fps": camera_format.maxFrameRate(),
            }
        selected_projector: dict[str, object] | None = None
        selected_projector_index = self._projector_selector.currentIndex()
        if 0 <= selected_projector_index < len(self._screens):
            selected_screen = self._screens[selected_projector_index]
            selected_geometry = selected_screen.geometry()
            selected_projector = {
                "name": selected_screen.name(),
                "manufacturer": selected_screen.manufacturer(),
                "model": selected_screen.model(),
                "width": selected_geometry.width(),
                "height": selected_geometry.height(),
                "x": selected_geometry.x(),
                "y": selected_geometry.y(),
            }
        calibration_metrics: dict[str, object] | None = None
        if self._calibration_result is not None:
            result = self._calibration_result
            calibration_metrics = {
                "board_confidence": result.board_detection.confidence,
                "board_method": result.board_detection.method,
                "board_reprojection_error_pixels": (
                    result.board_detection.reprojection_error
                ),
                "mapping_coverage_ratio": result.mapping.coverage_ratio,
                "mapping_decoded_ratio": result.mapping.decoded_ratio,
                "mapping_inlier_ratio": result.mapping.inlier_ratio,
                "mapping_rms_error_projector_pixels": (
                    result.mapping.rms_error
                ),
                "mapping_maximum_error_projector_pixels": (
                    result.mapping.maximum_error
                ),
                "validation_rms_error_camera_pixels": (
                    result.validation.rms_error
                ),
                "validation_maximum_error_camera_pixels": (
                    result.validation.maximum_error
                ),
                "validation_maximum_square_fraction": (
                    result.validation.maximum_square_fraction
                ),
            }
        state: dict[str, object] = {
            "application": {
                "python_version": sys.version,
                "platform": sys.platform,
                "opencv_version": cv2.__version__,
                "diagnostics_error": self._runtime_diagnostics_error,
            },
            "hardware": {
                "active_camera": active_camera,
                "available_cameras": [
                    {
                        "name": device.description(),
                        "id": self._device_key(device),
                    }
                    for device in self._devices
                ],
                "selected_projector": selected_projector,
                "available_projectors": [
                    {
                        "name": screen.name(),
                        "width": screen.geometry().width(),
                        "height": screen.geometry().height(),
                    }
                    for screen in self._screens
                ],
                "latest_frame_size": (
                    None
                    if self._latest_frame is None
                    else [
                        self._latest_frame.shape[1],
                        self._latest_frame.shape[0],
                    ]
                ),
                "camera_status": self._camera_status.text(),
            },
            "lichess": {
                "game_id": self._lichess_game_id,
                "player_color": self._lichess_player_color,
                "opponent": self._lichess_opponent_name,
                "connection_status": self._lichess_status.text(),
                "synchronization_status": self._lichess_sync_status.text(),
                "controller": controller_state,
                "display_position": _diagnostic_position(
                    self._lichess_board_widget.pieces
                ),
            },
            "physical": {
                "game_mode": self._game_mode,
                "accepted_position": _diagnostic_position(
                    self._piece_synchronizer.position
                ),
                "last_rejected_move": (
                    None
                    if self._last_rejected_move is None
                    else [
                        square_name(self._last_rejected_move[0]),
                        square_name(self._last_rejected_move[1]),
                    ]
                ),
                "reference_initialized": (
                    self._reference_move_tracker.initialized
                ),
                "diagnostic_images": {
                    "latest_camera": "diagnostics/runtime-camera.png",
                    "accepted_board": (
                        "diagnostics/runtime-accepted-board.png"
                    ),
                },
            },
            "tracker": tracker_state,
            "calibration": {
                "available": self._calibration_result is not None,
                "verified_available": (
                    self._verified_calibration_result is not None
                ),
                "camera_only": self._camera_only_calibration,
                "camera_geometry_ready": self._camera_geometry_ready,
                "operation_active": self._calibrator.active,
                "quick_verification": self._calibrator.quick_verification,
                "validation_only": self._calibrator.validation_only,
                "coded_pattern_mode": self._calibrator.coded_pattern_mode,
                "projector_output": self._alignment_monitor.output_mode,
                "metrics": calibration_metrics,
                "remembered_square": (
                    None
                    if self._remembered_board_square is None
                    else (
                        f"{chr(ord('a') + self._remembered_board_square[1])}"
                        f"{8 - self._remembered_board_square[0]}"
                    )
                ),
                "selected_square": (
                    None
                    if self._selected_board_square is None
                    else (
                        f"{chr(ord('a') + self._selected_board_square[1])}"
                        f"{8 - self._selected_board_square[0]}"
                    )
                ),
                "pending_square": (
                    None
                    if self._pending_board_square is None
                    else (
                        f"{chr(ord('a') + self._pending_board_square[1])}"
                        f"{8 - self._pending_board_square[0]}"
                    )
                ),
                "alignment_status": self._alignment_status.text(),
            },
            "ui": {
                "turn": self._turn_text.text(),
                "piece_status": self._piece_status.text(),
                "occupancy_status": self._occupancy_status.text(),
                "projector_status": self._projector_status.text(),
                "physical_board_enabled": self._board_widget.isEnabled(),
                "projector_window_visible": (
                    self._projector_window.isVisible()
                ),
                "board_images": {
                    "physical": "diagnostics/runtime-physical-ui.png",
                    "lichess": "diagnostics/runtime-lichess-ui.png",
                },
            },
        }
        images: dict[str, ImageArray] | None = None
        if include_images:
            images = {}
            if self._latest_frame is not None:
                images["camera"] = self._latest_frame
            accepted_reference = self._reference_move_tracker.reference
            if accepted_reference is not None:
                images["accepted-board"] = accepted_reference
        if additional_images:
            if images is None:
                images = {}
            images.update(additional_images)
        if include_images:
            if images is None:
                images = {}
            images.update(self._runtime_board_snapshot_images())
        try:
            queued = self._runtime_diagnostics.record_async(
                event_type,
                message,
                state,
                details=details,
                images=images,
            )
            writer_error = self._runtime_diagnostics.take_async_error()
            if writer_error is not None:
                raise OSError(writer_error)
            if not queued:
                raise OSError(
                    f"Runtime diagnostic event {event_type} was not queued."
                )
            self._runtime_diagnostics_error = None
        except (OSError, ValueError) as error:
            self._runtime_diagnostics_error = str(error)
            self._camera_status.setText(
                f"Runtime diagnostics unavailable: {error}"
            )

    def _runtime_board_snapshot_images(self) -> dict[str, ImageArray]:
        images: dict[str, ImageArray] = {}
        for name, widget in (
            ("physical-ui", self._board_widget),
            ("lichess-ui", self._lichess_board_widget),
        ):
            image = widget.grab().toImage().convertToFormat(
                QImage.Format.Format_RGB888
            )
            width = image.width()
            height = image.height()
            bytes_per_line = image.bytesPerLine()
            buffer = np.frombuffer(
                image.bits(),
                dtype=np.uint8,
                count=image.sizeInBytes(),
            ).reshape(height, bytes_per_line)
            rgb = buffer[:, : width * 3].reshape(height, width, 3)
            images[name] = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return images

    def _lichess_thread_finished(self) -> None:
        self._lichess_thread = None
        self._lichess_worker = None
        self._lichess_client = None
        self._lichess_connection_button.setText("Connect")
        self._lichess_connection_button.setEnabled(True)
        self._start_new_lichess_game_button.setEnabled(
            _can_start_new_lichess_game(self._latest_lichess_state)
        )
        self._update_lichess_game_control_buttons()
        if self._lichess_disconnect_requested:
            self._lichess_status.setText("Lichess disconnected.")
        self._lichess_disconnect_requested = False
        if self._close_pending:
            QTimer.singleShot(0, self.close)

    def _lichess_submission_finished(self) -> None:
        self._lichess_submission_thread = None
        self._lichess_submission_worker = None
        if self._close_pending:
            QTimer.singleShot(0, self.close)

    def _lichess_game_action_finished(self) -> None:
        self._lichess_game_action_thread = None
        self._lichess_game_action_worker = None
        self._update_lichess_game_control_buttons()
        if self._close_pending:
            QTimer.singleShot(0, self.close)

    def _update_turn_indicator(self) -> None:
        state = (
            _lichess_turn_indicator_state(self._latest_lichess_state)
            if self._latest_lichess_state is not None
            else None
            if getattr(self, "_offline_turn_inference_pending", False)
            else _turn_indicator_state(self._game_mode, self._legal_game)
        )
        if state is None:
            self._turn_indicator.hide()
            self._turn_text.setText("Not enforced")
            self._turn_indicator.setToolTip("")
            self._turn_indicator.setAccessibleName("No enforced chess turn")
            self._update_projected_turn_indicator()
            return
        color, description = state
        self._turn_indicator.setStyleSheet(
            f"background-color: {color}; border: 2px solid #404040;"
        )
        self._turn_indicator.setToolTip(description)
        self._turn_indicator.setAccessibleName(description)
        self._turn_indicator.show()
        self._turn_text.setText(description)
        self._update_projected_turn_indicator()

    def _update_projected_turn_indicator(self) -> None:
        corner = (
            None
            if (
                getattr(self, "_offline_turn_inference_pending", False)
                and self._latest_lichess_state is None
            )
            else _projected_turn_corner(
                self._project_turn_indicator,
                self._game_mode,
                self._legal_game,
                self._latest_lichess_state,
                self._board_quarter_turns,
            )
        )
        self._projector_window.set_turn_indicator(corner)

    def _restore_move_reference(self) -> None:
        if (
            self._camera is None
            or not self._piece_synchronizer.initialized
            or not 0 <= self._active_device_index < len(self._devices)
        ):
            return
        resolution = self._camera.cameraFormat().resolution()
        try:
            reference = self._reference_store.load_matching(
                camera_id=self._device_key(
                    self._devices[self._active_device_index]
                ),
                camera_resolution=(
                    resolution.width(),
                    resolution.height(),
                ),
                orientation=self._board_quarter_turns,
                position=self._piece_synchronizer.position,
            )
        except ValueError as error:
            self._piece_status.setText(str(error))
            return
        if reference is None:
            return
        self._reference_move_tracker.set_canonical_reference(reference)
        self._piece_status.setText(
            "Piece position and move reference restored."
        )

    def _persist_move_reference(self) -> None:
        reference = self._reference_move_tracker.reference
        if (
            reference is None
            or self._camera is None
            or not self._piece_synchronizer.initialized
            or not 0 <= self._active_device_index < len(self._devices)
        ):
            return
        resolution = self._camera.cameraFormat().resolution()
        try:
            self._reference_store.save(
                reference,
                camera_id=self._device_key(
                    self._devices[self._active_device_index]
                ),
                camera_resolution=(
                    resolution.width(),
                    resolution.height(),
                ),
                orientation=self._board_quarter_turns,
                position=self._piece_synchronizer.position,
            )
        except OSError as error:
            self._piece_status.setText(str(error))

    def _detect_current_board_orientation(self) -> None:
        if (
            not self._board_orientation_needs_detection
            or self._latest_frame is None
            or self._calibration_result is None
        ):
            return
        try:
            detected_turns = detect_board_orientation(
                bgr_to_gray(self._latest_frame),
                self._calibration_result.mapping.camera_grid,
            )
        except (CalibrationError, ValueError):
            return
        complete_orientation_retained = (
            self._board_quarter_turns % 2
            == detected_turns % 2
        )
        if not complete_orientation_retained:
            self._board_quarter_turns = detected_turns
        self._board_orientation_needs_detection = False
        if (
            complete_orientation_retained
            and self._piece_side_orientation_needs_detection
        ):
            self._piece_side_orientation_needs_detection = False
            self._refresh_pending_lichess_guidance()

    def _restore_remembered_square(self) -> None:
        if (
            self._remembered_board_square is None
            or self._calibration_result is None
            or self._calibrator.active
        ):
            return
        row, column = self._remembered_board_square
        if not self._open_projector():
            return
        self._selected_board_square = None
        self._pending_board_square = (row, column)
        self._board_widget.set_selected_square((row, column))
        self._projector_status.setText(
            "Restoring the remembered square after capturing a black reference."
        )
        QTimer.singleShot(
            250,
            lambda: self._project_board_square_after_black(row, column),
        )

    def _calibration_error(self, source: str, message: str) -> None:
        self._projector_window.show_black()
        invalidation_error = self._invalidate_saved_calibration()
        self._invalidate_calibration()
        camera_state = "error" if source == "camera" else "ok"
        projector_state = "error" if source == "projector" else "ok"
        self._set_calibration_indicator(
            self._camera_calibration_indicator,
            "Camera calibration",
            camera_state,
            message if source == "camera" else "",
        )
        self._set_calibration_indicator(
            self._projector_calibration_indicator,
            "Projector calibration",
            projector_state,
            message if source == "projector" else "",
        )
        self._alignment_status.setText(
            f"{source.capitalize()} calibration error: {message} "
            "The projection mapping was invalidated."
        )
        self._projector_status.setText(
            "Projection unavailable until calibration succeeds."
        )
        if invalidation_error is not None:
            self._projector_status.setText(
                f"{self._projector_status.text()} {invalidation_error}"
            )
        self._record_runtime_diagnostic(
            "alignment-calibration-error",
            message,
            details={
                "source": source,
                "previous_calibration_retained": False,
                "previous_calibration_invalidated": True,
            },
            include_images=True,
        )

    def _invalidate_saved_calibration(self) -> str | None:
        try:
            self._calibration_store.invalidate()
        except OSError as error:
            return str(error)
        return None

    def _projector_selection_changed(self) -> None:
        if self._calibrator.active:
            return
        selected_index = self._projector_selector.currentIndex()
        if 0 <= selected_index < len(self._screens):
            selected_screen = self._screens[selected_index]
            self._settings.setValue(
                _PROJECTOR_SETTING,
                selected_screen.name(),
            )
            self._settings.setValue(
                _PROJECTOR_LABEL_SETTING,
                CameraWindow._projector_screen_label(
                    selected_screen,
                    QGuiApplication.primaryScreen(),
                ),
            )
            self._settings.sync()
        self._record_runtime_diagnostic(
            "projector-selection-changed",
            "The selected projector display changed.",
            details={
                "selected_index": self._projector_selector.currentIndex()
            },
        )
        self._restore_saved_calibration()

    def _restore_saved_calibration(self) -> None:
        if self._camera is None:
            return
        selected_index = self._projector_selector.currentIndex()
        if not 0 <= selected_index < len(self._screens):
            self._invalidate_calibration()
            self._record_runtime_diagnostic(
                "saved-calibration-unavailable",
                "No projector display is selected.",
                details={"reason": "projector-unavailable"},
            )
            return

        active_device = self._devices[self._active_device_index]
        camera_resolution = self._camera.cameraFormat().resolution()
        selected_screen = self._screens[selected_index]
        projector_geometry = selected_screen.geometry()
        try:
            result = self._calibration_store.load_matching(
                camera_id=self._device_key(active_device),
                camera_resolution=(
                    camera_resolution.width(),
                    camera_resolution.height(),
                ),
                projector_name=selected_screen.name(),
                projector_resolution=(
                    projector_geometry.width(),
                    projector_geometry.height(),
                ),
            )
        except (OSError, ValueError) as error:
            self._invalidate_calibration()
            self._record_runtime_diagnostic(
                "saved-calibration-load-failed",
                str(error),
                details={"match": "camera-and-projector"},
            )
            QMessageBox.warning(
                self,
                "Saved calibration",
                str(error),
            )
            return
        if result is None:
            try:
                result = self._calibration_store.load_matching(
                    camera_id=self._device_key(active_device),
                    camera_resolution=(
                        camera_resolution.width(),
                        camera_resolution.height(),
                    ),
                    projector_name=None,
                    projector_resolution=None,
                )
            except (OSError, ValueError) as error:
                self._invalidate_calibration()
                self._record_runtime_diagnostic(
                    "saved-calibration-load-failed",
                    str(error),
                    details={"match": "camera-only"},
                )
                QMessageBox.warning(
                    self,
                    "Saved calibration",
                    str(error),
                )
                return
            if result is None:
                self._invalidate_calibration()
                self._record_runtime_diagnostic(
                    "saved-calibration-unavailable",
                    "No saved calibration matches the active camera.",
                    details={"reason": "camera-mismatch"},
                )
                return
            self._activate_camera_only_calibration(result)
            self._projector_status.setText(
                "Loaded saved camera board geometry. Physical projection "
                "is unavailable until the calibrated display returns."
            )
            self._record_runtime_diagnostic(
                "saved-calibration-loaded",
                self._projector_status.text(),
                details={"match": "camera-only"},
            )
            return

        self._activate_calibration_result(result)
        self._projector_status.setText(
            "Loaded the matching saved projector calibration. "
            "Recalibrate if the board plane or hardware moved."
        )
        self._record_runtime_diagnostic(
            "saved-calibration-loaded",
            self._projector_status.text(),
            details={"match": "camera-and-projector"},
            include_images=True,
        )
        self._restore_remembered_square()

    def _receive_video_frame(self, video_frame: QVideoFrame) -> None:
        if not video_frame.isValid():
            return

        image = video_frame.toImage()
        if image.isNull():
            self._detection_status.setText(
                "The selected camera produced an unreadable video frame."
            )
            return

        rgb_image = image.convertToFormat(QImage.Format.Format_RGB888)
        width = rgb_image.width()
        height = rgb_image.height()
        bytes_per_line = rgb_image.bytesPerLine()
        buffer = np.frombuffer(
            rgb_image.bits(),
            dtype=np.uint8,
            count=rgb_image.sizeInBytes(),
        ).reshape(height, bytes_per_line)
        rgb_frame = buffer[:, : width * 3].reshape(height, width, 3)
        camera_frame: ImageArray = np.asarray(
            cv2.cvtColor(
                rgb_frame,
                cv2.COLOR_RGB2BGR,
            ),
            dtype=np.uint8,
        )
        self._latest_frame = camera_frame.copy()
        self._alignment_monitor.accept_frame(self._latest_frame)
        self._calibrator.accept_frame(camera_frame)
        if not self._snapshot_visible:
            self._display_frame(camera_frame)
            self._detection_status.setText(
                "Live preview. Take a snapshot to detect the board."
            )

    def _take_snapshot(self) -> None:
        if self._latest_frame is None:
            self._detection_status.setText(
                "No camera frame is available for a snapshot."
            )
            return

        snapshot = self._latest_frame.copy()
        detection = _snapshot_board_detection(
            self._detector,
            snapshot,
            self._calibration_result,
        )
        overlay: ImageArray | None = None
        if detection is not None:
            overlay = self._detector.draw_overlay(
                snapshot,
                detection,
            )
            snapshot = overlay
            fit_text = (
                f", fit error {detection.reprojection_error:.2f} px"
                if detection.reprojection_error is not None
                else ""
            )
            self._detection_status.setText(
                f"Board detected - confidence "
                f"{detection.confidence:.0%}, "
                f"method {detection.method}{fit_text}"
            )
        else:
            self._detection_status.setText(
                "No complete empty 8x8 board was detected in this snapshot."
            )
        self._record_runtime_diagnostic(
            "board-snapshot-detected" if detection is not None else (
                "board-snapshot-detection-failed"
            ),
            self._detection_status.text(),
            details={
                "confidence": (
                    None if detection is None else detection.confidence
                ),
                "method": None if detection is None else detection.method,
                "reprojection_error_pixels": (
                    None
                    if detection is None
                    else detection.reprojection_error
                ),
            },
            include_images=True,
        )

        if self._save_diagnostics.isChecked():
            active_device = self._devices[self._active_device_index]
            try:
                capture_directory = self._diagnostic_store.save(
                    original=self._latest_frame,
                    overlay=overlay,
                    detection=detection,
                    camera_name=active_device.description(),
                    camera_id=self._device_key(active_device),
                )
            except OSError as error:
                QMessageBox.warning(self, "Diagnostic save failed", str(error))
                self._detection_status.setText(
                    f"{self._detection_status.text()} Diagnostic save failed."
                )
            else:
                self._detection_status.setText(
                    f"{self._detection_status.text()} "
                    f"Saved to diagnostics/{capture_directory.name}."
                )

        self._snapshot_visible = True
        self._snapshot_button.setEnabled(False)
        self._resume_button.setEnabled(True)
        self._display_frame(snapshot)

    def _resume_live_camera(self) -> None:
        self._snapshot_visible = False
        self._snapshot_button.setEnabled(True)
        self._resume_button.setEnabled(False)
        self._detection_status.setText(
            "Live preview. Take a snapshot to detect the board."
        )
        if self._latest_frame is not None:
            self._display_frame(self._latest_frame)

    def _display_frame(self, frame: ImageArray) -> None:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb_frame.shape
        image = QImage(
            rgb_frame.data,
            width,
            height,
            channels * width,
            QImage.Format.Format_RGB888,
        ).copy()
        self._current_pixmap = QPixmap.fromImage(image)
        self._update_preview()

    def _update_preview(self) -> None:
        if self._current_pixmap is None:
            return
        self._preview.setPixmap(
            self._current_pixmap.scaled(
                self._preview.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def resizeEvent(self, event: QResizeEvent) -> None:
        self._update_preview()
        super().resizeEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() in (Qt.Key.Key_Escape, Qt.Key.Key_Q):
            self.close()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event: QCloseEvent) -> None:
        if (
            self._lichess_ai_creation_thread is not None
            and self._lichess_ai_creation_thread.isRunning()
        ):
            self._close_pending = True
            self._lichess_status.setText(
                "Waiting for Lichess AI game creation before closing."
            )
            event.ignore()
            return
        if (
            self._lichess_game_action_thread is not None
            and self._lichess_game_action_thread.isRunning()
        ):
            self._close_pending = True
            self._lichess_status.setText(
                "Waiting for the pending Lichess game control before closing."
            )
            event.ignore()
            return
        if (
            self._lichess_submission_thread is not None
            and self._lichess_submission_thread.isRunning()
        ):
            self._close_pending = True
            self._lichess_status.setText(
                "Waiting for the pending Lichess move submission before "
                "closing."
            )
            event.ignore()
            return
        if (
            self._lichess_worker is not None
            and self._lichess_thread is not None
            and self._lichess_thread.isRunning()
        ):
            self._close_pending = True
            self._lichess_status.setText(
                "Disconnecting from Lichess before closing."
            )
            self._lichess_worker.request_stop()
            event.ignore()
            return
        _save_window_placement(self, self._settings)
        self._calibrator.cancel()
        self._alignment_monitor.clear()
        if self._camera is not None:
            self._camera.stop()
        self._projector_window.close()
        self._runtime_diagnostics.close()
        event.accept()


def _parse_arguments(arguments: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect an empty physical chessboard using a USB camera."
    )
    parser.add_argument(
        "--camera",
        default=None,
        help="Initial camera index or unique part of its displayed name.",
    )
    return parser.parse_args(arguments)


def run(arguments: list[str] | None = None) -> int:
    parsed = _parse_arguments(sys.argv[1:] if arguments is None else arguments)
    application = QApplication(sys.argv)
    try:
        window = CameraWindow(parsed.camera)
    except RuntimeError as error:
        QMessageBox.critical(None, "Camera error", str(error))
        return 1

    window.show()
    return application.exec()
