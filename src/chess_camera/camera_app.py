import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QSettings, Qt
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

from .board_detector import BoardDetector, ImageArray
from .calibration_controller import (
    CalibrationResult,
    ProjectorCalibrationController,
)
from .calibration_store import CalibrationStore
from .chessboard_widget import ChessboardWidget
from .diagnostics import DiagnosticStore
from .projector import ProjectorWindow

_SETTINGS_ORGANIZATION = "ChessProject"
_SETTINGS_APPLICATION = "ChessboardCamera"
_CAMERA_SETTING = "camera/device_id"
_PROJECTOR_SETTING = "projector/screen_name"
_PROJECTOR_X_SETTING = "projector/target_x_percent"
_PROJECTOR_Y_SETTING = "projector/target_y_percent"
_PROJECTOR_SIZE_SETTING = "projector/target_size_percent"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


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
        self._devices = list(QMediaDevices.videoInputs())
        device_names = [device.description() for device in self._devices]
        device_keys = [self._device_key(device) for device in self._devices]
        default_key = self._device_key(QMediaDevices.defaultVideoInput())
        saved_key = str(self._settings.value(_CAMERA_SETTING, ""))
        selected_index = _resolve_camera_index(
            device_names,
            device_keys,
            default_key,
            saved_key,
            camera_selector,
        )

        self._camera: QCamera | None = None
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
        self._calibration_result: CalibrationResult | None = None
        self._selected_board_square: tuple[int, int] | None = None
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

        self._camera_selector = QComboBox()
        self._camera_selector.addItems(device_names)
        self._camera_selector.setCurrentIndex(selected_index)
        self._use_camera_button = QPushButton("Use selected camera")
        self._use_camera_button.clicked.connect(self._use_selected_camera)

        camera_layout = QHBoxLayout()
        camera_layout.addWidget(QLabel("Camera:"))
        camera_layout.addWidget(self._camera_selector, stretch=1)
        camera_layout.addWidget(self._use_camera_button)

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
            self._projector_window.show_target
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
        projector_target_layout.addWidget(self._black_projector_button)
        projector_target_layout.addWidget(self._close_projector_button)

        self._start_calibration_button = QPushButton(
            "Start projector calibration"
        )
        self._start_calibration_button.clicked.connect(
            self._start_projector_calibration
        )
        self._cancel_calibration_button = QPushButton("Cancel calibration")
        self._cancel_calibration_button.clicked.connect(self._calibrator.cancel)
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
        calibration_layout.addWidget(self._cancel_calibration_button)
        calibration_layout.addWidget(self._show_grid_button)
        calibration_layout.addWidget(self._show_squares_button)

        self._calibration_progress = QProgressBar()
        self._calibration_progress.setRange(0, 1)
        self._calibration_progress.setValue(0)
        self._calibration_progress.setVisible(False)

        self._projector_status = QLabel(
            "Projector closed. Open it black before showing the square."
        )
        self._projector_status.setTextInteractionFlags(
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
        layout.addWidget(self._camera_status)
        layout.addWidget(self._detection_status)
        layout.addWidget(self._projector_status)
        layout.addWidget(self._instructions)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.setWindowTitle("Chessboard Camera Detection")
        self.resize(1450, 900)
        self._projector_selector.currentIndexChanged.connect(
            self._projector_selection_changed
        )
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

    def _use_selected_camera(self) -> None:
        selected_index = self._camera_selector.currentIndex()
        try:
            self._start_camera(selected_index)
        except RuntimeError as error:
            QMessageBox.critical(self, "Camera error", str(error))

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

    def _black_projector(self) -> None:
        if self._calibrator.active:
            self._calibrator.cancel()
        self._projector_window.show_black()
        self._selected_board_square = None
        self._board_widget.set_selected_square(None)
        self._projector_status.setText("Projector is black.")

    def _close_projector(self) -> None:
        if self._calibrator.active:
            self._calibrator.cancel()
        self._projector_window.show_black()
        self._projector_window.close()
        self._selected_board_square = None
        self._board_widget.set_selected_square(None)
        self._projector_status.setText("Projector closed.")

    def _show_calibrated_grid(self) -> None:
        if not self._open_projector():
            return
        try:
            self._projector_window.show_grid()
        except RuntimeError as error:
            QMessageBox.warning(self, "Projector calibration", str(error))
            return
        self._selected_board_square = None
        self._board_widget.set_selected_square(None)
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
        self._selected_board_square = None
        self._board_widget.set_selected_square(None)
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
        self._cancel_calibration_button.setEnabled(active)
        self._camera_selector.setEnabled(not active)
        self._use_camera_button.setEnabled(not active)
        self._projector_selector.setEnabled(not active)
        self._open_projector_button.setEnabled(not active)
        self._show_target_button.setEnabled(not active)
        self._close_projector_button.setEnabled(not active)
        self._snapshot_button.setEnabled(not active and not self._snapshot_visible)

    def _projector_calibration_completed(
        self,
        result: CalibrationResult,
    ) -> None:
        self._activate_calibration_result(result)
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

    def _projector_calibration_failed(self, message: str) -> None:
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
        if self._selected_board_square == (row, column):
            self._black_projector()
            return
        if not self._open_projector():
            return
        try:
            self._projector_window.show_square(row, column)
        except (RuntimeError, ValueError) as error:
            QMessageBox.warning(self, "Square projection", str(error))
            return

        self._selected_board_square = (row, column)
        self._board_widget.set_selected_square((row, column))
        square_name = f"{chr(ord('a') + column)}{8 - row}"
        self._projector_status.setText(
            f"Projecting {square_name} (row {row + 1}, column {column + 1})."
        )

    def _activate_calibration_result(
        self,
        result: CalibrationResult,
    ) -> None:
        self._calibration_result = result
        self._selected_board_square = None
        self._projector_window.set_projector_grid(
            result.mapping.projector_grid
        )
        self._board_widget.set_selected_square(None)
        self._board_widget.setEnabled(True)
        self._show_grid_button.setEnabled(True)
        self._show_squares_button.setEnabled(True)

    def _invalidate_calibration(self) -> None:
        self._calibration_result = None
        self._selected_board_square = None
        self._projector_window.clear_projector_grid()
        self._board_widget.set_selected_square(None)
        self._board_widget.setEnabled(False)
        self._show_grid_button.setEnabled(False)
        self._show_squares_button.setEnabled(False)

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
        detection = self._detector.detect(snapshot)
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
