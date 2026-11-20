from typing import Literal

import cv2
import numpy as np
from PySide6.QtCore import QObject, QRunnable, QThreadPool, QTimer, Signal, Slot

from .board_detector import (
    BoardDetection,
    BoardDetector,
    ImageArray,
    StableBoardTracker,
)
from .calibration_controller import CalibrationResult
from .projector_calibration import (
    CalibrationError,
    MappingResult,
    bgr_to_gray,
    interpolate_grid_point,
    locate_validation_target,
    remap_projector_mapping,
)

OutputMode = Literal["black", "selected", "paused"]


class _DetectionSignals(QObject):
    completed = Signal(int, object, object)
    failed = Signal(int, str)


class _DetectionTask(QRunnable):
    def __init__(
        self,
        generation: int,
        frame: ImageArray,
        detector: BoardDetector | StableBoardTracker,
    ) -> None:
        super().__init__()
        self._generation = generation
        self._frame = frame
        self._detector = detector
        self.signals = _DetectionSignals()

    @Slot()
    def run(self) -> None:
        try:
            detection = self._detector.detect(self._frame)
        except (ValueError, cv2.error) as error:
            self.signals.failed.emit(self._generation, str(error))
            return
        self.signals.completed.emit(
            self._generation,
            detection,
            self._frame,
        )


class AlignmentMonitor(QObject):
    status_changed = Signal(str)
    alignment_lost = Signal(str)
    calibration_error = Signal(str, str)
    mapping_updated = Signal(object, float)
    mapping_verified = Signal(object)
    camera_grid_checked = Signal()

    def __init__(
        self,
        detector: BoardDetector,
        *,
        interval_milliseconds: int = 200,
        board_remap_threshold: float = 1.5,
        projection_error_fraction: float = 0.1,
        stable_checks_after_remap: int = 5,
    ) -> None:
        super().__init__()
        if interval_milliseconds < 100:
            raise ValueError("Alignment interval must be at least 100 ms.")
        if board_remap_threshold <= 0:
            raise ValueError("Board remap threshold must be positive.")
        if projection_error_fraction <= 0:
            raise ValueError("Projection error fraction must be positive.")
        if stable_checks_after_remap < 1:
            raise ValueError(
                "Stable checks after remapping must be at least one."
            )

        self._detector = detector
        self._tracker = StableBoardTracker(detector)
        self._thread_pool = QThreadPool.globalInstance()
        self._timer = QTimer(self)
        self._timer.setInterval(interval_milliseconds)
        self._timer.timeout.connect(self._start_check)
        self._interval_milliseconds = interval_milliseconds
        self._board_remap_threshold = board_remap_threshold
        self._projection_error_fraction = projection_error_fraction
        self._stable_checks_after_remap = stable_checks_after_remap
        self._result: CalibrationResult | None = None
        self._projector_size = (0, 0)
        self._camera_only = False
        self._latest_frame: ImageArray | None = None
        self._black_reference: ImageArray | None = None
        self._selected_square: tuple[int, int] | None = None
        self._verify_selected_square = True
        self._output_mode: OutputMode = "paused"
        self._check_running = False
        self._generation = 0
        self._geometry_failure_count = 0
        self._geometry_failure_source: str | None = None
        self._stable_board_checks = stable_checks_after_remap
        self._remapped_since_verification = False

    def configure(
        self,
        result: CalibrationResult,
        projector_size: tuple[int, int] | None,
    ) -> None:
        if (
            projector_size is not None
            and (projector_size[0] <= 1 or projector_size[1] <= 1)
        ):
            raise ValueError("Projector dimensions must be greater than one.")
        self._result = result
        self._tracker.reset(result.board_detection)
        self._projector_size = projector_size or (0, 0)
        self._camera_only = projector_size is None
        self._selected_square = None
        self._verify_selected_square = True
        self._output_mode = "black"
        self._black_reference = None
        self._geometry_failure_count = 0
        self._geometry_failure_source = None
        self._stable_board_checks = self._stable_checks_after_remap
        self._remapped_since_verification = False
        self._generation += 1
        self._check_running = False
        self._timer.start()
        self.status_changed.emit(
            f"Alignment monitor active every "
            f"{self._interval_milliseconds / 1000:g} second."
        )

    def clear(self) -> None:
        self._result = None
        self._camera_only = False
        self._tracker.reset()
        self._selected_square = None
        self._verify_selected_square = True
        self._output_mode = "paused"
        self._black_reference = None
        self._generation += 1
        self._check_running = False
        self._timer.stop()
        self.status_changed.emit("Alignment monitor inactive.")

    def accept_frame(self, frame: ImageArray) -> None:
        self._latest_frame = frame

    @property
    def output_mode(self) -> OutputMode:
        return self._output_mode

    def capture_black_reference(self, frame: ImageArray) -> None:
        if self._result is not None:
            self._black_reference = bgr_to_gray(frame)

    def set_black(self) -> None:
        if self._result is None:
            return
        self._generation += 1
        self._check_running = False
        self._output_mode = "black"
        self._selected_square = None
        self._verify_selected_square = True
        self._geometry_failure_count = 0
        self._geometry_failure_source = None
        self.status_changed.emit(
            "Alignment monitor active; acquiring black reference."
        )

    def set_selected_square(
        self,
        row: int,
        column: int,
        *,
        verify_projection: bool = True,
    ) -> None:
        if self._result is None:
            return
        if not 0 <= row < 8 or not 0 <= column < 8:
            raise ValueError("Selected square must be inside the 8x8 board.")
        self._generation += 1
        self._check_running = False
        self._output_mode = "selected"
        self._selected_square = (row, column)
        self._verify_selected_square = verify_projection
        self._geometry_failure_count = 0
        self.status_changed.emit(
            "Alignment monitor checking the active projected square."
        )

    def set_paused(self, reason: str) -> None:
        if self._result is None:
            return
        self._generation += 1
        self._check_running = False
        self._output_mode = "paused"
        self._selected_square = None
        self.status_changed.emit(f"Alignment monitor paused: {reason}")

    def _start_check(self) -> None:
        if (
            self._result is None
            or self._output_mode == "paused"
            or self._check_running
            or self._latest_frame is None
        ):
            return
        frame = self._latest_frame.copy()
        generation = self._generation
        task = _DetectionTask(generation, frame, self._tracker)
        task.signals.completed.connect(self._check_completed)
        task.signals.failed.connect(self._check_failed)
        self._check_running = True
        self._thread_pool.start(task)

    @Slot(int, object, object)
    def _check_completed(
        self,
        generation: int,
        detection_object: object,
        frame_object: object,
    ) -> None:
        if generation != self._generation or self._result is None:
            return
        self._check_running = False
        detection = (
            detection_object
            if isinstance(detection_object, BoardDetection)
            else None
        )
        frame = (
            frame_object
            if isinstance(frame_object, np.ndarray)
            else None
        )
        if frame is None:
            self.status_changed.emit(
                "Alignment check failed because its camera frame was invalid."
            )
            return
        if self._tracker.state != "stable_board":
            return
        if detection is None:
            return

        displacements = np.linalg.norm(
            detection.grid_points - self._result.mapping.camera_grid,
            axis=2,
        )
        displacement = float(np.max(displacements))
        if displacement > self._board_remap_threshold:
            old_camera_grid = self._result.mapping.camera_grid
            if self._camera_only:
                previous = self._result.mapping
                updated_mapping = MappingResult(
                    camera_grid=np.asarray(
                        detection.grid_points,
                        dtype=np.float32,
                    ).copy(),
                    projector_grid=previous.projector_grid.copy(),
                    camera_to_projector_homography=(
                        previous.camera_to_projector_homography.copy()
                    ),
                    coverage_ratio=previous.coverage_ratio,
                    decoded_ratio=previous.decoded_ratio,
                    inlier_ratio=previous.inlier_ratio,
                    rms_error=previous.rms_error,
                    maximum_error=previous.maximum_error,
                )
            else:
                try:
                    updated_mapping = remap_projector_mapping(
                        self._result.mapping,
                        detection.grid_points,
                        self._projector_size,
                    )
                except (CalibrationError, ValueError) as error:
                    self._geometry_failed(str(error), "camera")
                    return
            self._result = CalibrationResult(
                board_detection=detection,
                mapping=updated_mapping,
                validation=self._result.validation,
            )
            self._warp_black_reference(
                old_camera_grid,
                detection.grid_points,
            )
            self._geometry_failure_count = 0
            self._geometry_failure_source = None
            self._stable_board_checks = 0
            self._remapped_since_verification = True
            self.mapping_updated.emit(updated_mapping, displacement)
            if self._output_mode == "selected":
                self.status_changed.emit(
                    f"Board moved {displacement:.1f} px; projector coordinates "
                    "were remapped and will be checked on the next frame."
                )
                return
        else:
            self._stable_board_checks = min(
                self._stable_board_checks + 1,
                self._stable_checks_after_remap,
            )

        if self._output_mode == "black":
            self._black_reference = bgr_to_gray(frame)
            self._geometry_failure_count = 0
            self._geometry_failure_source = None
            self.camera_grid_checked.emit()
            self.status_changed.emit(
                f"Alignment OK; board drift {displacement:.1f} px. "
                "Projector drift will be checked on the next selected square."
            )
            return

        if self._selected_square is None or self._black_reference is None:
            self.status_changed.emit(
                "Board alignment OK; waiting for a black reference before "
                "projector verification."
            )
            return
        if not self._verify_selected_square:
            self._geometry_failure_count = 0
            self._geometry_failure_source = None
            self.status_changed.emit(
                f"Alignment OK; board drift {displacement:.1f} px. "
                "Projected-square verification skipped because the selected "
                "square is occupied."
            )
            return
        if self._stable_board_checks < self._stable_checks_after_remap:
            self.status_changed.emit(
                "Board movement detected; projected coordinates are following "
                "the board while alignment verification waits for it to stop."
            )
            return

        row, column = self._selected_square
        expected = interpolate_grid_point(
            detection.grid_points,
            column + 0.5,
            row + 0.5,
        )
        horizontal_steps = np.linalg.norm(
            np.diff(detection.grid_points, axis=1),
            axis=2,
        )
        vertical_steps = np.linalg.norm(
            np.diff(detection.grid_points, axis=0),
            axis=2,
        )
        square_width = float(
            np.median(
                np.concatenate(
                    (horizontal_steps.reshape(-1), vertical_steps.reshape(-1))
                )
            )
        )
        try:
            measured = locate_validation_target(
                self._black_reference,
                bgr_to_gray(frame),
                expected,
                max(8, round(square_width * 0.75)),
            )
        except CalibrationError as error:
            self._geometry_failed(
                str(error),
                self._optical_error_source(),
            )
            return

        error = float(np.linalg.norm(measured - expected))
        error_fraction = error / square_width
        if error_fraction > self._projection_error_fraction:
            self._geometry_failed(
                f"The projected square is misaligned by "
                f"{error_fraction:.1%} of a square.",
                self._optical_error_source(),
            )
            return

        was_remapped = self._remapped_since_verification
        self._geometry_failure_count = 0
        self._geometry_failure_source = None
        self._remapped_since_verification = False
        if was_remapped:
            self.mapping_verified.emit(self._result.mapping)
        self.status_changed.emit(
            f"Alignment OK; board drift {displacement:.1f} px, projected "
            f"square error {error:.1f} px ({error_fraction:.1%})."
        )

    @Slot(int, str)
    def _check_failed(self, generation: int, message: str) -> None:
        if generation != self._generation:
            return
        self._check_running = False
        self.status_changed.emit(f"Alignment check failed: {message}")

    def _optical_error_source(self) -> str:
        return (
            "camera"
            if self._remapped_since_verification
            else "projector"
        )

    def _geometry_failed(self, message: str, source: str) -> None:
        if source not in ("camera", "projector"):
            raise ValueError(f"Unknown calibration error source: {source}")
        if source != self._geometry_failure_source:
            self._geometry_failure_count = 0
            self._geometry_failure_source = source
        self._geometry_failure_count += 1
        if self._geometry_failure_count >= 2:
            self._lose_alignment(
                f"{message} The failure occurred in two consecutive checks.",
                source,
            )
        else:
            self.status_changed.emit(
                f"Possible alignment loss (1/2): {message}"
            )

    def _warp_black_reference(
        self,
        old_camera_grid: np.ndarray,
        current_camera_grid: np.ndarray,
    ) -> None:
        if self._black_reference is None:
            return
        transform, _mask = cv2.findHomography(
            old_camera_grid.reshape(-1, 2),
            current_camera_grid.reshape(-1, 2),
            method=0,
        )
        if transform is None:
            self._black_reference = None
            return
        height, width = self._black_reference.shape
        self._black_reference = cv2.warpPerspective(
            self._black_reference,
            transform,
            (width, height),
        )

    def _lose_alignment(self, message: str, source: str) -> None:
        self._timer.stop()
        self._result = None
        self._output_mode = "paused"
        self._generation += 1
        self._check_running = False
        self.calibration_error.emit(source, message)
        self.alignment_lost.emit(message)
