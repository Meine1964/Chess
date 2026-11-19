import argparse
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QSettings, QTimer, Qt
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
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
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
from .chessboard_widget import ChessboardWidget, rotate_square
from .diagnostics import DiagnosticStore
from .projector import ProjectorWindow, focus_pattern
from .projector_calibration import (
    CalibrationError,
    MappingResult,
    bgr_to_gray,
    detect_board_orientation,
)

_SETTINGS_ORGANIZATION = "ChessProject"
_SETTINGS_APPLICATION = "ChessboardCamera"
_CAMERA_SETTING = "camera/device_id"
_PROJECTOR_SETTING = "projector/screen_name"
_PROJECTOR_X_SETTING = "projector/target_x_percent"
_PROJECTOR_Y_SETTING = "projector/target_y_percent"
_PROJECTOR_SIZE_SETTING = "projector/target_size_percent"
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
        self._calibration_store = CalibrationStore(
            _PROJECT_ROOT / "calibrations"
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
        self._preview.setMinimumSize(960, 540)
        self._preview.setStyleSheet("background: #202020;")
        self._board_widget = ChessboardWidget()
        self._board_widget.setEnabled(False)
        self._board_widget.square_clicked.connect(self._board_square_clicked)
        self._board_quarter_turns = int(
            self._settings.value(_BOARD_ORIENTATION_SETTING, 0)
        ) % 2
        self._board_orientation_needs_detection = True

        self._camera_selector = QComboBox()
        self._camera_selector.addItems(device_names)
        if selected_index is not None:
            self._camera_selector.setCurrentIndex(selected_index)
        self._camera_selector.setEnabled(bool(self._devices))

        camera_layout = QHBoxLayout()
        camera_layout.addWidget(QLabel("Camera:"))
        camera_layout.addWidget(self._camera_selector, stretch=1)

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
            "Place the entire empty board on a plain table. Press Esc or Q to exit."
        )

        self._snapshot_button = QPushButton("Take snapshot")
        self._snapshot_button.clicked.connect(self._take_snapshot)
        self._resume_button = QPushButton("Resume live camera")
        self._resume_button.clicked.connect(self._resume_live_camera)
        self._resume_button.setEnabled(False)
        self._save_diagnostics = QCheckBox("Save diagnostic snapshots")
        self._save_diagnostics.setChecked(True)
        button_layout = QHBoxLayout()
        button_layout.addWidget(self._snapshot_button)
        button_layout.addWidget(self._resume_button)
        button_layout.addWidget(self._save_diagnostics)

        self._screens = list(QGuiApplication.screens())
        self._projector_selector = QComboBox()
        primary_screen = QGuiApplication.primaryScreen()
        for screen in self._screens:
            geometry = screen.geometry()
            primary_suffix = " (primary)" if screen is primary_screen else ""
            hardware_parts = [
                value
                for value in (screen.manufacturer(), screen.model())
                if value
            ]
            hardware = (
                f" [{' '.join(hardware_parts)}]"
                if hardware_parts
                else ""
            )
            self._projector_selector.addItem(
                f"{screen.name()}{hardware} - "
                f"{geometry.width()}x{geometry.height()} at "
                f"({geometry.x()}, {geometry.y()}){primary_suffix}"
            )
        saved_screen_name = str(
            self._settings.value(_PROJECTOR_SETTING, "")
        )
        projector_index = self._default_projector_index(
            self._screens,
            primary_screen,
            saved_screen_name,
        )
        self._projector_selector.setCurrentIndex(projector_index)

        self._target_x = self._percentage_spin_box(
            int(self._settings.value(_PROJECTOR_X_SETTING, 50))
        )
        self._target_y = self._percentage_spin_box(
            int(self._settings.value(_PROJECTOR_Y_SETTING, 50))
        )
        self._target_size = self._percentage_spin_box(
            int(self._settings.value(_PROJECTOR_SIZE_SETTING, 10)),
            minimum=1,
            maximum=50,
        )
        self._target_x.valueChanged.connect(self._update_projector_target)
        self._target_y.valueChanged.connect(self._update_projector_target)
        self._target_size.valueChanged.connect(self._update_projector_target)

        self._open_projector_button = QPushButton("Open projector (black)")
        self._open_projector_button.clicked.connect(self._open_projector)
        self._show_target_button = QPushButton("Show square")
        self._show_target_button.clicked.connect(
            self._show_manual_target
        )
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

        projector_target_layout = QHBoxLayout()
        projector_target_layout.addWidget(QLabel("Square center X:"))
        projector_target_layout.addWidget(self._target_x)
        projector_target_layout.addWidget(QLabel("Y:"))
        projector_target_layout.addWidget(self._target_y)
        projector_target_layout.addWidget(QLabel("Size:"))
        projector_target_layout.addWidget(self._target_size)
        projector_target_layout.addWidget(self._show_target_button)
        projector_target_layout.addWidget(self._focus_pattern_button)
        projector_target_layout.addWidget(self._black_projector_button)
        projector_target_layout.addWidget(self._close_projector_button)

        self._start_calibration_button = QPushButton(
            "Full projector calibration"
        )
        self._start_calibration_button.clicked.connect(
            self._start_projector_calibration
        )
        self._coded_calibration_button = QPushButton(
            "Full calibration - coded pattern"
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
        calibration_layout = QHBoxLayout()
        calibration_layout.addWidget(self._start_calibration_button)
        calibration_layout.addWidget(self._coded_calibration_button)
        calibration_layout.addWidget(self._validate_calibration_button)
        calibration_layout.addWidget(self._quick_verify_button)
        calibration_layout.addWidget(self._cancel_calibration_button)
        calibration_layout.addWidget(self._show_grid_button)
        calibration_layout.addWidget(self._show_squares_button)

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

        content_layout = QHBoxLayout()
        content_layout.addWidget(self._preview, stretch=3)
        content_layout.addWidget(self._board_widget, stretch=2)

        layout = QVBoxLayout()
        layout.addLayout(content_layout, stretch=1)
        layout.addLayout(camera_layout)
        layout.addLayout(button_layout)
        layout.addLayout(projector_display_layout)
        layout.addLayout(projector_target_layout)
        layout.addLayout(calibration_layout)
        layout.addWidget(self._calibration_progress)
        layout.addLayout(calibration_indicator_layout)
        layout.addWidget(self._camera_status)
        layout.addWidget(self._detection_status)
        layout.addWidget(self._projector_status)
        layout.addWidget(self._alignment_status)
        layout.addWidget(self._instructions)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.setWindowTitle("Chessboard Camera Detection")
        self.resize(1450, 900)
        self._projector_selector.currentIndexChanged.connect(
            self._projector_selection_changed
        )
        self._camera_selector.currentIndexChanged.connect(
            self._camera_selection_changed
        )
        self._media_devices.videoInputsChanged.connect(
            self._camera_devices_changed
        )
        self._open_projector()
        if selected_index is None:
            self._wait_for_external_camera()
        else:
            self._start_camera(selected_index)

    @staticmethod
    def _percentage_spin_box(
        value: int,
        minimum: int = 0,
        maximum: int = 100,
    ) -> QSpinBox:
        spin_box = QSpinBox()
        spin_box.setRange(minimum, maximum)
        spin_box.setSuffix("%")
        spin_box.setValue(min(max(value, minimum), maximum))
        return spin_box

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

    def _camera_selection_changed(self, selected_index: int) -> None:
        previous_index = self._active_device_index
        try:
            self._start_camera(selected_index)
        except RuntimeError as error:
            self._camera_selector.blockSignals(True)
            self._camera_selector.setCurrentIndex(previous_index)
            self._camera_selector.blockSignals(False)
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

    def _open_projector(self) -> bool:
        selected_index = self._projector_selector.currentIndex()
        if not 0 <= selected_index < len(self._screens):
            QMessageBox.critical(
                self,
                "Projector error",
                "The selected projector display is no longer available.",
            )
            return False

        screen = self._screens[selected_index]
        self._projector_window.show_black()
        self._update_projector_target()
        try:
            self._projector_window.show_on_screen(screen)
        except RuntimeError as error:
            QMessageBox.critical(self, "Projector error", str(error))
            return False

        self._settings.setValue(_PROJECTOR_SETTING, screen.name())
        self._settings.sync()
        geometry = screen.geometry()
        self._projector_status.setText(
            f"Projector open black on {screen.name()} at "
            f"{geometry.width()}x{geometry.height()}."
        )
        return True

    def _update_projector_target(self) -> None:
        self._projector_window.set_target(
            self._target_x.value(),
            self._target_y.value(),
            self._target_size.value(),
        )
        self._settings.setValue(
            _PROJECTOR_X_SETTING,
            self._target_x.value(),
        )
        self._settings.setValue(
            _PROJECTOR_Y_SETTING,
            self._target_y.value(),
        )
        self._settings.setValue(
            _PROJECTOR_SIZE_SETTING,
            self._target_size.value(),
        )
        self._settings.sync()

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
        self._invalidate_calibration()
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

    def _show_manual_target(self) -> None:
        self._suspend_selected_square()
        self._projector_window.show_target()
        self._alignment_monitor.set_paused("manual projector target")

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
        self._show_target_button.setEnabled(not active)
        self._focus_pattern_button.setEnabled(not active)
        self._close_projector_button.setEnabled(not active)
        self._snapshot_button.setEnabled(not active and not self._snapshot_visible)
        if active:
            self._alignment_monitor.set_paused("projector calibration")

    def _cancel_projector_calibration(self) -> None:
        self._calibrator.cancel()
        self._restore_remembered_square()

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
        self._restore_remembered_square()

    def _projector_calibration_failed(self, message: str) -> None:
        self._projector_window.show_black()
        if (
            (
                self._calibrator.quick_verification
                or self._calibrator.validation_only
                or self._calibrator.coded_pattern_mode
            )
            and self._calibration_result is not None
        ):
            result = self._calibration_result
            self._projector_window.set_projector_grid(
                result.mapping.projector_grid
            )
            selected_screen = self._screens[
                self._projector_selector.currentIndex()
            ]
            geometry = selected_screen.geometry()
            self._alignment_monitor.configure(
                result,
                (geometry.width(), geometry.height()),
            )
            self._alignment_monitor.set_black()
            operation = (
                "Quick realignment"
                if self._calibrator.quick_verification
                else (
                    "Validation"
                    if self._calibrator.validation_only
                    else "Coded-pattern calibration"
                )
            )
            self._projector_status.setText(
                f"{operation} failed: {message} The previous verified "
                "calibration remains active."
            )
            if self._calibrator.validation_only:
                QMessageBox.warning(
                    self,
                    "Projector calibration validation failed",
                    message,
                )
            self._restore_remembered_square()
            return
        self._invalidate_calibration()
        self._set_calibration_indicator(
            self._projector_calibration_indicator,
            "Projector calibration",
            "error",
            message,
        )
        QMessageBox.warning(
            self,
            "Projector calibration failed",
            message,
        )

    def _board_square_clicked(self, row: int, column: int) -> None:
        if self._calibration_result is None:
            self._projector_status.setText(
                "No matching projector calibration is available."
            )
            return
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
            QMessageBox.warning(self, "Square projection", str(error))
            return

        self._pending_board_square = None
        self._selected_board_square = (row, column)
        self._remembered_board_square = (row, column)
        self._board_widget.set_selected_square((row, column))
        self._alignment_monitor.set_selected_square(
            physical_row,
            physical_column,
        )
        square_name = f"{chr(ord('a') + column)}{8 - row}"
        self._projector_status.setText(
            f"Projecting {square_name} (row {row + 1}, column {column + 1})."
        )

    def _activate_calibration_result(
        self,
        result: CalibrationResult,
    ) -> None:
        self._calibration_result = result
        self._verified_calibration_result = result
        self._board_orientation_needs_detection = True
        self._suspend_selected_square()
        self._projector_window.set_projector_grid(
            result.mapping.projector_grid
        )
        self._board_widget.setEnabled(True)
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

    def _invalidate_calibration(self) -> None:
        self._calibration_result = None
        self._verified_calibration_result = None
        self._suspend_selected_square()
        self._projector_window.clear_projector_grid()
        self._board_widget.setEnabled(False)
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
        self._projector_window.set_projector_grid(mapping.projector_grid)
        if self._selected_board_square is not None:
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
        self._projector_status.setText(
            f"Board moved {displacement:.1f} camera pixels; projected "
            "coordinates were updated without full recalibration."
        )

    def _alignment_mapping_verified(self, _mapping: MappingResult) -> None:
        if self._calibration_result is None:
            return
        self._verified_calibration_result = self._calibration_result

    def _suspend_selected_square(self) -> None:
        self._selected_board_square = None
        self._pending_board_square = None
        self._board_widget.set_selected_square(
            self._remembered_board_square
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
        self._settings.setValue(_BOARD_ORIENTATION_SETTING, turns)
        self._settings.sync()

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
        self._suspend_selected_square()
        if self._verified_calibration_result is not None:
            self._calibration_result = self._verified_calibration_result
            self._projector_window.set_projector_grid(
                self._verified_calibration_result.mapping.projector_grid
            )
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
            "The last verified calibration remains active."
        )
        self._projector_status.setText(
            "The last verified calibration remains active; restoring the "
            "remembered square."
        )
        self._restore_remembered_square()

    def _projector_selection_changed(self) -> None:
        if self._calibrator.active:
            return
        self._restore_saved_calibration()

    def _restore_saved_calibration(self) -> None:
        if self._camera is None:
            return
        selected_index = self._projector_selector.currentIndex()
        if not 0 <= selected_index < len(self._screens):
            self._invalidate_calibration()
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
            QMessageBox.warning(
                self,
                "Saved calibration",
                str(error),
            )
            return
        if result is None:
            self._invalidate_calibration()
            return

        self._activate_calibration_result(result)
        self._projector_status.setText(
            "Loaded the matching saved projector calibration. "
            "Recalibrate if the board plane or hardware moved."
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
