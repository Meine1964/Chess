import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import cast

import cv2
import numpy as np
from PySide6.QtCore import QSettings, QThread, QTimer, Qt
from PySide6.QtGui import (
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
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
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
    square_name,
    starting_occupied_squares,
)
from .chessboard_widget import ChessboardWidget, rotate_square
from .diagnostics import DiagnosticStore, RuntimeDiagnosticLog
from .legal_game import GameMode, LegalGame, LegalMoveResult
from .lichess_client import (
    AccountGameEvent,
    LichessConfigurationError,
    LichessGameFull,
    LichessGameState,
    LichessPollingSnapshot,
    LichessReadOnlyClient,
    PlayerColor,
)
from .lichess_game_controller import LichessGameController
from .lichess_worker import LichessConnectionWorker
from .occupancy import (
    SquareOccupancyDetector,
    detect_starting_position_orientation,
)
from .projector import ProjectorWindow, focus_pattern
from .projector_calibration import (
    CalibrationError,
    MappingResult,
    bgr_to_gray,
    detect_board_orientation,
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
_PROJECTOR_SETTING = "projector/screen_name"
_PIECE_POSITION_SETTING = "pieces/accepted_position"
_GAME_MODE_SETTING = "game/mode"
_LEGAL_GAME_FEN_SETTING = "game/offline_fen"
_BOARD_ORIENTATION_SETTING = "board/automatic_clockwise_quarter_turns"
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
    return (
        f"Lichess game {game.game_id}: "
        f"{game.white.name} (White) vs {game.black.name} (Black). "
        f"{state.turn.capitalize()} to move; {clock_text}"
        f"status {state.status}; result {state.result}. "
        "Connected by read-only polling."
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
    default_key: str,
    saved_key: str,
    selector: str | None,
) -> int:
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
    if default_key in device_keys:
        return device_keys.index(default_key)
    return 0


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
        default_key = self._device_key(QMediaDevices.defaultVideoInput())
        saved_key = str(self._settings.value(_CAMERA_SETTING, ""))
        selected_index = (
            _resolve_camera_index(
                device_names,
                device_keys,
                default_key,
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
        self._reference_move_tracker = ReferenceMoveTracker()
        self._piece_synchronizer = PiecePositionSynchronizer()
        self._legal_game = LegalGame()
        self._game_mode: GameMode = "diagnostic"
        self._lichess_thread: QThread | None = None
        self._lichess_worker: LichessConnectionWorker | None = None
        self._lichess_disconnect_requested = False
        self._close_pending = False
        self._lichess_game_id: str | None = None
        self._lichess_player_color: PlayerColor | None = None
        self._lichess_opponent_name = ""
        self._lichess_game_controller: LichessGameController | None = None
        self._last_projected_lichess_uci: str | None = None
        self._latest_lichess_state: LichessGameState | None = None
        self._last_runtime_tracker_state: str | None = None
        self._runtime_diagnostics_error: str | None = None
        self._latest_stable_occupied_ui: (
            frozenset[tuple[int, int]] | None
        ) = None
        self._reset_to_start_requested = False
        self._reconciliation_requested = False
        self._last_rejected_move: (
            tuple[tuple[int, int], tuple[int, int]] | None
        ) = None
        self._camera_only_calibration = False
        self._camera_geometry_ready = False
        self._calibration_result: CalibrationResult | None = None
        self._verified_calibration_result: CalibrationResult | None = None
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
        self._lichess_board_widget.setEnabled(False)
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
        self._camera_selector.setEnabled(bool(self._devices))
        self._game_mode_selector = QComboBox()
        self._game_mode_selector.setMaximumWidth(260)
        self._game_mode_selector.addItem("Diagnostic movement", "diagnostic")
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
        camera_layout.addWidget(QLabel("Camera:"))
        camera_layout.addWidget(self._camera_selector)
        camera_layout.addWidget(QLabel("Rules:"))
        camera_layout.addWidget(self._game_mode_selector)
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

        self._snapshot_button = QPushButton("Take snapshot")
        self._snapshot_button.clicked.connect(self._take_snapshot)
        self._resume_button = QPushButton("Resume live camera")
        self._resume_button.clicked.connect(self._resume_live_camera)
        self._resume_button.setEnabled(False)
        self._save_diagnostics = QCheckBox("Save diagnostic snapshots")
        self._save_diagnostics.setChecked(True)
        self._reset_pieces_button = QPushButton(
            "Reset pieces to starting position"
        )
        self._reset_pieces_button.clicked.connect(
            self._reset_pieces_to_start
        )
        self._reset_pieces_button.setEnabled(False)
        self._reconcile_button = QPushButton("Reconcile physical board")
        self._reconcile_button.clicked.connect(
            self._toggle_physical_reconciliation
        )
        self._reconcile_button.setEnabled(False)
        self._lichess_button = QPushButton("Connect to Lichess")
        self._lichess_button.clicked.connect(
            self._toggle_lichess_connection
        )
        button_layout = QGridLayout()
        button_layout.addWidget(self._snapshot_button, 0, 0)
        button_layout.addWidget(self._resume_button, 0, 1)
        button_layout.addWidget(self._save_diagnostics, 0, 2)
        button_layout.addWidget(self._reset_pieces_button, 1, 0)
        button_layout.addWidget(self._reconcile_button, 1, 1)
        button_layout.addWidget(self._lichess_button, 1, 2)

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
        calibrated_screen_name = self._calibrated_screen_name(
            self._screens
        )
        projector_index = self._default_projector_index(
            self._screens,
            primary_screen,
            calibrated_screen_name or saved_screen_name,
        )
        self._projector_selector.setCurrentIndex(projector_index)

        self._open_projector_button = QPushButton("Open projector (black)")
        self._open_projector_button.clicked.connect(self._open_projector)
        self._focus_pattern_button = QPushButton("Focus pattern")
        self._focus_pattern_button.clicked.connect(
            self._show_focus_pattern
        )
        self._black_projector_button = QPushButton("Black")
        self._black_projector_button.clicked.connect(
            self._black_projector
        )
        self._close_projector_button = QPushButton("Close projector")
        self._close_projector_button.clicked.connect(
            self._close_projector
        )

        projector_display_layout = QHBoxLayout()
        projector_display_layout.addWidget(QLabel("Projector display:"))
        projector_display_layout.addWidget(
            self._projector_selector,
            stretch=1,
        )
        projector_display_layout.addWidget(self._open_projector_button)

        projector_actions_layout = QGridLayout()
        projector_actions_layout.addWidget(
            self._focus_pattern_button,
            0,
            0,
        )
        projector_actions_layout.addWidget(
            self._black_projector_button,
            0,
            1,
        )
        projector_actions_layout.addWidget(
            self._close_projector_button,
            0,
            2,
        )

        self._start_calibration_button = QPushButton(
            "Full calibration (empty board)"
        )
        self._start_calibration_button.setToolTip(
            "Remove all pieces before running the accepted Gray-code "
            "calibration."
        )
        self._start_calibration_button.clicked.connect(
            self._start_projector_calibration
        )
        self._coded_calibration_button = QPushButton(
            "Populated calibration (experimental)"
        )
        self._coded_calibration_button.setToolTip(
            "Experimental hardware path for recalibration without removing "
            "the pieces. A failure retains the previous verified calibration."
        )
        self._coded_calibration_button.clicked.connect(
            self._start_coded_projector_calibration
        )
        self._quick_verify_button = QPushButton(
            "Quick realign / assess"
        )
        self._quick_verify_button.clicked.connect(
            self._start_quick_verification
        )
        self._quick_verify_button.setEnabled(False)
        self._validate_calibration_button = QPushButton(
            "Validate calibration"
        )
        self._validate_calibration_button.clicked.connect(
            self._start_validation_only
        )
        self._validate_calibration_button.setEnabled(False)
        self._cancel_calibration_button = QPushButton("Cancel calibration")
        self._cancel_calibration_button.clicked.connect(
            self._cancel_projector_calibration
        )
        self._cancel_calibration_button.setEnabled(False)
        self._show_grid_button = QPushButton("Project 8x8 grid")
        self._show_grid_button.clicked.connect(self._show_calibrated_grid)
        self._show_grid_button.setEnabled(False)
        self._show_squares_button = QPushButton("Project all 64 squares")
        self._show_squares_button.clicked.connect(
            self._show_calibrated_squares
        )
        self._show_squares_button.setEnabled(False)
        calibration_layout = QGridLayout()
        calibration_layout.addWidget(self._start_calibration_button, 0, 0)
        calibration_layout.addWidget(self._coded_calibration_button, 0, 1)
        calibration_layout.addWidget(
            self._validate_calibration_button,
            0,
            2,
        )
        calibration_layout.addWidget(self._quick_verify_button, 0, 3)
        calibration_layout.addWidget(self._cancel_calibration_button, 1, 0)
        calibration_layout.addWidget(self._show_grid_button, 1, 1)
        calibration_layout.addWidget(
            self._show_squares_button,
            1,
            2,
            1,
            2,
        )
        for button in (
            self._snapshot_button,
            self._resume_button,
            self._reset_pieces_button,
            self._reconcile_button,
            self._lichess_button,
            self._open_projector_button,
            self._focus_pattern_button,
            self._black_projector_button,
            self._close_projector_button,
            self._start_calibration_button,
            self._coded_calibration_button,
            self._validate_calibration_button,
            self._quick_verify_button,
            self._cancel_calibration_button,
            self._show_grid_button,
            self._show_squares_button,
        ):
            button.setMaximumWidth(220)

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
            "Lichess disconnected. Configure LICHESS_TOKEN to connect."
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

        lichess_panel = QGroupBox("Lichess position")
        lichess_panel_layout = QVBoxLayout()
        lichess_panel_layout.addWidget(self._lichess_board_widget)
        lichess_panel.setLayout(lichess_panel_layout)

        interaction_content = QWidget()
        interaction_layout = QVBoxLayout()
        interaction_layout.addLayout(camera_layout)
        interaction_layout.addLayout(button_layout)
        interaction_layout.addLayout(projector_display_layout)
        interaction_layout.addLayout(projector_actions_layout)
        interaction_layout.addLayout(calibration_layout)
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
        if selected_index is None:
            self._wait_for_external_camera()
        else:
            self._start_camera(selected_index)
        self._record_runtime_diagnostic(
            "application-started",
            "Chessboard camera application started.",
        )

    @staticmethod
    def _default_projector_index(
        screens: Sequence[QScreen],
        primary_screen: QScreen | None,
        saved_screen_name: str,
    ) -> int:
        for index, screen in enumerate(screens):
            if screen.name() == saved_screen_name:
                return index
        for index, screen in enumerate(screens):
            if screen is not primary_screen:
                return index
        return 0

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

    def _calibrated_screen_name(
        self,
        screens: Sequence[QScreen],
    ) -> str:
        try:
            saved_display = self._calibration_store.saved_projector_display()
        except ValueError:
            return ""
        if saved_display is None:
            return ""
        saved_name, saved_resolution = saved_display
        for screen in screens:
            geometry = screen.geometry()
            if (
                screen.name() == saved_name
                and (geometry.width(), geometry.height())
                == saved_resolution
            ):
                return saved_name
        return ""

    def _available_screens_changed(self, _screen: QScreen) -> None:
        current_index = self._projector_selector.currentIndex()
        current_name = (
            self._screens[current_index].name()
            if 0 <= current_index < len(self._screens)
            else ""
        )
        screens = list(QGuiApplication.screens())
        primary_screen = QGuiApplication.primaryScreen()
        calibrated_screen_name = self._calibrated_screen_name(screens)
        saved_screen_name = str(
            self._settings.value(_PROJECTOR_SETTING, "")
        )
        preferred_name = (
            calibrated_screen_name
            or current_name
            or saved_screen_name
        )
        selected_index = self._default_projector_index(
            screens,
            primary_screen,
            preferred_name,
        )
        self._projector_selector.blockSignals(True)
        self._projector_selector.clear()
        for screen in screens:
            self._projector_selector.addItem(
                self._projector_screen_label(screen, primary_screen)
            )
        self._screens = screens
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
        default_key = self._device_key(QMediaDevices.defaultVideoInput())
        try:
            selected_index = _resolve_camera_index(
                [device.description() for device in devices],
                device_keys,
                default_key,
                saved_key,
                self._requested_camera_selector,
            )
        except RuntimeError as error:
            self._camera_selector.blockSignals(False)
            self._wait_for_external_camera(str(error))
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
        if self._calibrator.active:
            self._calibrator.cancel()
        self._projector_window.show_black()
        self._suspend_selected_square()
        self._alignment_monitor.set_black()
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
            operation = "Coded-pattern calibration"
        else:
            operation = "Full calibration"
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

    def _activate_calibration_result(
        self,
        result: CalibrationResult,
    ) -> None:
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

    def _alignment_mapping_verified(self, _mapping: MappingResult) -> None:
        if self._calibration_result is None:
            return
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
        self._board_widget.set_selected_square(
            self._remembered_board_square
        )

    def _reset_occupancy(self, status: str) -> None:
        self._occupancy_detector.reset()
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
        if (
            self._reference_move_tracker.initialized
            and not self._reset_to_start_requested
            and not self._reconciliation_requested
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

    def _update_reference_move_tracking(self) -> None:
        if self._latest_frame is None or self._calibration_result is None:
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
        try:
            update = self._reference_move_tracker.observe(
                self._latest_frame,
                self._calibration_result.mapping.camera_grid,
                self._board_quarter_turns,
                tracking_position,
                (
                    (lichess_signature,)
                    if lichess_signature is not None
                    else self._legal_game.physical_move_signatures
                    if self._game_mode == "offline-game"
                    else ()
                ),
            )
        except (ValueError, cv2.error) as error:
            self._occupancy_status.setText(
                f"Reference tracking unavailable: {error}"
            )
            return
        status_message = _reference_tracking_status_message(update)
        if status_message is not None:
            self._occupancy_status.setText(status_message)
        if (
            update.state != self._last_runtime_tracker_state
            or update.state
            in {
                "accepted",
                "ambiguous",
                "illegal-removal",
                "pending-capture",
            }
        ):
            self._last_runtime_tracker_state = update.state
            self._record_runtime_diagnostic(
                "reference-tracker",
                update.message,
                tracker_update=update,
                include_images=True,
            )
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
            self._last_rejected_move = None
            self._occupancy_status.setText(
                "Board matches the accepted reference."
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
            newly_marked = self._last_rejected_move != update.move
            self._last_rejected_move = update.move
            self._board_widget.set_illegal_source(square)
            self._piece_status.setText(
                _illegal_correction_message(update.move, self._legal_game)
            )
            if newly_marked:
                self._project_illegal_move_correction(update.move)
            return
        if update.state != "accepted" or update.move is None:
            return
        source, destination = update.move
        legal_move: LegalMoveResult | None = None
        lichess_pending = (
            lichess_controller is not None
            and lichess_signature is not None
        )
        if lichess_controller is not None and not lichess_pending:
            self._reject_unconfirmed_lichess_move(update.move)
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
            try:
                legal_move = self._legal_game.apply_move(
                    source,
                    destination,
                )
            except ValueError as error:
                self._reference_move_tracker.reset_activity()
                newly_marked = self._last_rejected_move != update.move
                self._last_rejected_move = update.move
                self._board_widget.set_illegal_source(source)
                self._piece_status.setText(
                    _illegal_correction_message(update.move, self._legal_game)
                )
                if newly_marked:
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
            if legal_move is not None and not lichess_pending:
                self._legal_game.rollback()
            self._reference_move_tracker.reset_activity()
            self._piece_status.setText(
                f"Reference move rejected: {error}"
            )
            return
        self._board_widget.set_pieces(position_update.position)
        self._board_widget.set_occupied_squares(position_update.position)
        self._board_widget.set_illegal_source(None)
        self._last_rejected_move = None
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
            self._project_accepted_move(update.move)

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
            return
        self._piece_synchronizer.reset()
        self._legal_game.reset()
        self._update_turn_indicator()
        update = self._piece_synchronizer.observe(
            occupied,
        )
        self._board_widget.set_pieces(update.position)
        self._persist_piece_position(update.position)
        self._persist_game_state()
        self._board_widget.set_illegal_source(None)
        self._last_rejected_move = None
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

    def _update_lichess_synchronization_projection(self) -> None:
        controller = self._lichess_game_controller
        if controller is None:
            return
        signature = controller.expected_signature
        if signature is None:
            self._last_projected_lichess_uci = None
            return
        self._last_projected_lichess_uci = signature.uci
        self._project_lichess_move_guidance(
            (signature.source, signature.destination)
        )

    def _project_lichess_move_guidance(
        self,
        move: tuple[tuple[int, int], tuple[int, int]],
    ) -> None:
        source, destination = move
        move_name = f"{square_name(source)}-{square_name(destination)}"
        if self._camera_only_calibration:
            self._projector_status.setText(
                f"Lichess move {move_name} is pending; physical projection "
                "is unavailable."
            )
            return
        if self._calibration_result is None or not self._open_projector():
            self._projector_status.setText(
                f"Lichess move {move_name} is pending, but no calibrated "
                "projector is available."
            )
            return
        try:
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
            return
        self._alignment_monitor.set_paused("Lichess move guidance")
        self._projector_status.setText(
            f"Projecting pending Lichess move {move_name} for 2 seconds."
        )
        QTimer.singleShot(2000, self._finish_move_projection)

    def _reject_unconfirmed_lichess_move(
        self,
        move: tuple[tuple[int, int], tuple[int, int]],
    ) -> None:
        controller = self._lichess_game_controller
        self._reference_move_tracker.reset_activity()
        newly_marked = self._last_rejected_move != move
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
        if newly_marked:
            self._project_illegal_move_correction(move)

    def _project_illegal_move_correction(
        self,
        move: tuple[tuple[int, int], tuple[int, int]],
    ) -> None:
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
            else:
                self._projector_window.show_correction(
                    physical_source,
                    physical_destination,
                )
        except (RuntimeError, ValueError) as error:
            self._projector_status.setText(
                f"Illegal move marked on the UI, but projection failed: "
                f"{error}"
            )
            return
        self._alignment_monitor.set_paused("illegal move indication")
        self._projector_status.setText(
            f"Projecting the illegally removed square in red for 2 seconds."
            if source == destination
            else "Projecting the illegal move source and destination in red "
            "for 2 seconds."
        )
        QTimer.singleShot(2000, self._finish_illegal_move_projection)

    def _finish_move_projection(self) -> None:
        if self._projector_window.output_mode != "move":
            return
        self._projector_window.show_black()
        self._alignment_monitor.set_black()
        self._projector_status.setText(
            "Move indication finished; projector returned to black."
        )

    def _finish_illegal_move_projection(self) -> None:
        if self._projector_window.output_mode not in (
            "illegal",
            "correction",
        ):
            return
        self._projector_window.show_black()
        self._alignment_monitor.set_black()
        self._projector_status.setText(
            "Illegal-move indication finished; projector returned to black."
        )

    def _toggle_physical_reconciliation(self) -> None:
        if self._reconciliation_requested:
            self._reconciliation_requested = False
            self._reconcile_button.setText("Reconcile physical board")
            self._black_projector()
            self._occupancy_detector.reset()
            self._reference_move_tracker.reset_activity()
            self._occupancy_status.setText(
                "Physical-board reconciliation cancelled."
            )
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
        self._reconciliation_requested = True
        self._reconcile_button.setText("Cancel reconciliation")
        self._board_widget.set_illegal_source(None)
        self._last_rejected_move = None
        self._black_projector()
        self._occupancy_detector.reset()
        self._reference_move_tracker.reset_activity()
        self._occupancy_status.setText(
            "Reconciliation started; checking stable physical occupancy."
        )
        self._piece_status.setText(
            "Keep the board clear while mismatching squares are checked."
        )

    def _update_physical_reconciliation(
        self,
        physically_occupied: frozenset[tuple[int, int]],
    ) -> None:
        result = compare_physical_occupancy(
            self._piece_synchronizer.position,
            physically_occupied,
        )
        self._board_widget.set_occupied_squares(physically_occupied)
        self._piece_status.setText(result.message)
        if result.matches:
            if self._latest_frame is None or self._calibration_result is None:
                raise RuntimeError(
                    "Reconciliation matched without a calibrated frame."
                )
            self._reference_move_tracker.set_reference(
                self._latest_frame,
                self._calibration_result.mapping.camera_grid,
                self._board_quarter_turns,
            )
            self._persist_move_reference()
            self._reconciliation_requested = False
            self._reconcile_button.setText("Reconcile physical board")
            self._occupancy_status.setText(
                "Reconciliation complete; accepted reference replaced."
            )
            self._piece_status.setText(
                _accepted_position_message(self._legal_game)
                if self._game_mode == "offline-game"
                else "Physical and UI piece positions are synchronized."
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

    def _finish_reconciliation_projection(self) -> None:
        if self._projector_window.output_mode != "reconciliation":
            return
        self._projector_window.show_black()
        self._alignment_monitor.set_black()
        self._occupancy_detector.reset()
        self._projector_status.setText(
            "Reconciliation projection finished; checking occupancy again."
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
        except ValueError:
            self._settings.remove(_LEGAL_GAME_FEN_SETTING)
            self._settings.setValue(_GAME_MODE_SETTING, "diagnostic")
            self._settings.sync()
            return
        if (
            not self._piece_synchronizer.initialized
            or restored.position != self._piece_synchronizer.position
        ):
            self._settings.setValue(_GAME_MODE_SETTING, "diagnostic")
            self._settings.sync()
            return
        self._legal_game = restored
        self._game_mode = "offline-game"

    def _persist_game_state(self) -> None:
        self._settings.setValue(_GAME_MODE_SETTING, self._game_mode)
        self._settings.setValue(_LEGAL_GAME_FEN_SETTING, self._legal_game.fen)
        self._settings.sync()

    def _game_mode_changed(self, index: int) -> None:
        selected = str(self._game_mode_selector.itemData(index))
        if selected == "diagnostic":
            self._game_mode = "diagnostic"
            self._board_widget.set_illegal_source(None)
            self._last_rejected_move = None
            self._persist_game_state()
            self._update_turn_indicator()
            self._piece_status.setText(
                "Diagnostic movement mode accepts unambiguous relocations."
            )
            return
        if selected != "offline-game":
            raise ValueError(f"Unknown game mode: {selected}")
        position = self._piece_synchronizer.position
        if position == self._legal_game.position:
            self._game_mode = "offline-game"
            self._persist_game_state()
            self._update_turn_indicator()
            self._piece_status.setText(
                f"Offline legal game enabled; "
                f"{self._legal_game.status.turn} to move."
            )
            return
        self._game_mode_selector.blockSignals(True)
        self._game_mode_selector.setCurrentIndex(0)
        self._game_mode_selector.blockSignals(False)
        self._update_turn_indicator()
        self._piece_status.setText(
            "Cannot enable legal play: reset the physical board to the "
            "starting position or restore its saved legal game."
        )

    def _toggle_lichess_connection(self) -> None:
        if self._lichess_worker is not None:
            self._lichess_disconnect_requested = True
            self._lichess_button.setEnabled(False)
            self._lichess_button.setText("Disconnecting...")
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
        worker.failed.connect(self._lichess_connection_failed)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(self._lichess_thread_finished)
        thread.finished.connect(thread.deleteLater)
        self._lichess_thread = thread
        self._lichess_worker = worker
        self._lichess_disconnect_requested = False
        self._lichess_button.setText("Disconnect Lichess")
        self._lichess_status.setText("Connecting to Lichess...")
        self._record_runtime_diagnostic(
            "lichess-connect-requested",
            "Connecting to Lichess.",
        )
        thread.start()

    def _lichess_connection_status_changed(self, message: str) -> None:
        self._lichess_status.setText(message)
        self._record_runtime_diagnostic(
            "lichess-connection-status",
            message,
        )

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
            if event.game_id != self._lichess_game_id:
                self._lichess_game_controller = None
                self._last_projected_lichess_uci = None
            self._lichess_game_id = event.game_id
            self._lichess_player_color = event.color
            self._lichess_opponent_name = event.opponent_name
            self._lichess_status.setText(
                f"Lichess game {event.game_id} started as "
                f"{event.color} against {event.opponent_name}; loading state."
            )
        else:
            self._lichess_status.setText(
                f"Lichess game {event.game_id} finished."
            )
        self._record_runtime_diagnostic(
            "lichess-account-event",
            f"{event.event_type} for game {event.game_id}.",
        )

    def _lichess_game_event_received(self, event: object) -> None:
        if isinstance(event, LichessPollingSnapshot):
            self._update_lichess_board(event.game.state)
            self._update_lichess_synchronization(event.game)
            self._lichess_status.setText(
                _lichess_polling_snapshot_message(event)
            )
        elif isinstance(event, LichessGameFull):
            self._update_lichess_board(event.state)
            self._update_lichess_synchronization(event)
            self._lichess_status.setText(
                f"Lichess game {event.game_id}: "
                f"{event.white.name} (White) vs "
                f"{event.black.name} (Black). "
                f"{_lichess_game_state_message(event.state)}"
            )
        elif isinstance(event, LichessGameState):
            self._update_lichess_board(event)
            self._update_lichess_synchronization_state(event)
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
        position, last_move = _lichess_board_state(state)
        self._lichess_board_widget.set_pieces(position)
        self._lichess_board_widget.set_occupied_squares(position)
        self._lichess_board_widget.set_last_move(last_move)
        self._update_turn_indicator()

    def _update_lichess_synchronization(
        self,
        game: LichessGameFull,
    ) -> None:
        if (
            self._lichess_game_id != game.game_id
            or self._lichess_player_color is None
        ):
            self._lichess_sync_status.setText(
                "Lichess physical synchronization is waiting for account "
                "game metadata."
            )
            return
        if (
            self._lichess_game_controller is None
            or self._lichess_game_controller.game_id != game.game_id
        ):
            initial_position = LegalGame(game.initial_fen).position
            if (
                not self._reference_move_tracker.initialized
                or self._piece_synchronizer.position != initial_position
            ):
                self._lichess_sync_status.setText(
                    "Physical reconciliation required: Phase 7 currently "
                    "attaches only when the accepted physical board matches "
                    "the Lichess initial position."
                )
                self._record_runtime_diagnostic(
                    "lichess-reconciliation-required",
                    self._lichess_sync_status.text(),
                    include_images=True,
                )
                return
            self._lichess_game_controller = LichessGameController(
                game.game_id,
                game.initial_fen,
                self._lichess_player_color,
                self._lichess_opponent_name,
            )
        self._update_lichess_synchronization_state(game.state)

    def _update_lichess_synchronization_state(
        self,
        state: LichessGameState,
    ) -> None:
        controller = self._lichess_game_controller
        if controller is None:
            return
        update = controller.accept_server_history(state.moves)
        self._lichess_sync_status.setText(update.message)
        self._record_runtime_diagnostic(
            "lichess-synchronization",
            update.message,
        )
        signature = controller.expected_signature
        if signature is None:
            self._last_projected_lichess_uci = None
            return
        if signature.uci == self._last_projected_lichess_uci:
            return
        self._last_projected_lichess_uci = signature.uci
        self._piece_status.setText(update.message)
        self._project_lichess_move_guidance(
            (signature.source, signature.destination)
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
        try:
            self._runtime_diagnostics.record(
                event_type,
                message,
                state,
                details=details,
                images=images,
            )
            self._save_runtime_board_snapshots()
            self._runtime_diagnostics_error = None
        except (OSError, ValueError) as error:
            self._runtime_diagnostics_error = str(error)
            self._camera_status.setText(
                f"Runtime diagnostics unavailable: {error}"
            )

    def _save_runtime_board_snapshots(self) -> None:
        diagnostic_root = _PROJECT_ROOT / "diagnostics"
        diagnostic_root.mkdir(parents=True, exist_ok=True)
        for name, widget in (
            ("physical-ui", self._board_widget),
            ("lichess-ui", self._lichess_board_widget),
        ):
            temporary_path = diagnostic_root / f"runtime-{name}.tmp.png"
            final_path = diagnostic_root / f"runtime-{name}.png"
            if not widget.grab().save(str(temporary_path), "PNG"):
                raise OSError(
                    f"Could not save runtime {name} board snapshot."
                )
            temporary_path.replace(final_path)

    def _lichess_thread_finished(self) -> None:
        self._lichess_thread = None
        self._lichess_worker = None
        self._lichess_button.setText("Connect to Lichess")
        self._lichess_button.setEnabled(True)
        if self._lichess_disconnect_requested:
            self._lichess_status.setText("Lichess disconnected.")
        self._lichess_disconnect_requested = False
        if self._close_pending:
            QTimer.singleShot(0, self.close)

    def _update_turn_indicator(self) -> None:
        state = (
            _lichess_turn_indicator_state(self._latest_lichess_state)
            if self._latest_lichess_state is not None
            else _turn_indicator_state(self._game_mode, self._legal_game)
        )
        if state is None:
            self._turn_indicator.hide()
            self._turn_text.setText("Not enforced")
            self._turn_indicator.setToolTip("")
            self._turn_indicator.setAccessibleName("No enforced chess turn")
            return
        color, description = state
        self._turn_indicator.setStyleSheet(
            f"background-color: {color}; border: 2px solid #404040;"
        )
        self._turn_indicator.setToolTip(description)
        self._turn_indicator.setAccessibleName(description)
        self._turn_indicator.show()
        self._turn_text.setText(description)

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
            turns = detect_board_orientation(
                bgr_to_gray(self._latest_frame),
                self._calibration_result.mapping.camera_grid,
            )
        except (CalibrationError, ValueError):
            return
        self._board_quarter_turns = turns
        self._board_orientation_needs_detection = False

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
        self._calibrator.cancel()
        self._alignment_monitor.clear()
        if self._camera is not None:
            self._camera.stop()
        self._projector_window.close()
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
