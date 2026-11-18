import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QCloseEvent, QImage, QKeyEvent, QPixmap, QResizeEvent
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
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .board_detector import BoardDetector, ImageArray
from .diagnostics import DiagnosticStore

_SETTINGS_ORGANIZATION = "ChessProject"
_SETTINGS_APPLICATION = "ChessboardCamera"
_CAMERA_SETTING = "camera/device_id"
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

        layout = QVBoxLayout()
        layout.addWidget(self._preview, stretch=1)
        layout.addLayout(camera_layout)
        layout.addLayout(button_layout)
        layout.addWidget(self._camera_status)
        layout.addWidget(self._detection_status)
        layout.addWidget(self._instructions)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.setWindowTitle("Chessboard Camera Detection")
        self.resize(1200, 800)
        self._start_camera(selected_index)

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
            self._detection_status.setText(
                f"Board detected - confidence "
                f"{detection.confidence:.0%}, "
                f"fit error {detection.reprojection_error:.2f} px"
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
        if self._camera is not None:
            self._camera.stop()
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
