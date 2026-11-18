from collections.abc import Callable
from dataclasses import dataclass, replace

import cv2
import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QObject, QTimer, Signal

from .board_detector import BoardDetection, BoardDetector, FloatArray, ImageArray
from .projector import ProjectorWindow
from .projector_calibration import (
    CalibrationError,
    GrayImage,
    MappingResult,
    ValidationResult,
    bgr_to_gray,
    decode_gray_code,
    fit_projector_mapping,
    gray_code_bit_count,
    gray_code_pattern,
    interpolate_grid_point,
    locate_validation_target,
    measure_coverage,
    refine_projector_mesh,
    validation_result,
)


@dataclass(frozen=True)
class CalibrationResult:
    board_detection: BoardDetection
    mapping: MappingResult
    validation: ValidationResult


@dataclass(frozen=True)
class _CaptureStep:
    key: str
    pattern: GrayImage
    description: str


class ProjectorCalibrationController(QObject):
    status_changed = Signal(str)
    progress_changed = Signal(int, int)
    completed = Signal(object)
    failed = Signal(str)
    active_changed = Signal(bool)
    _OMITTED_LEAST_SIGNIFICANT_BITS = 1

    def __init__(
        self,
        projector: ProjectorWindow,
        detector: BoardDetector,
        *,
        settle_milliseconds: int = 250,
        frames_per_state: int = 3,
        intensity: int = 112,
        maximum_refinement_iterations: int = 2,
        capture_timeout_milliseconds: int = 3000,
    ) -> None:
        super().__init__()
        if settle_milliseconds < 0:
            raise ValueError("Settle time must not be negative.")
        if frames_per_state < 1:
            raise ValueError("At least one frame per state is required.")
        if not 1 <= intensity <= 255:
            raise ValueError("Intensity must be between 1 and 255.")
        if maximum_refinement_iterations < 0:
            raise ValueError("Maximum refinement iterations must not be negative.")
        if capture_timeout_milliseconds < 1:
            raise ValueError("Capture timeout must be positive.")

        self._projector = projector
        self._detector = detector
        self._settle_milliseconds = settle_milliseconds
        self._frames_per_state = frames_per_state
        self._intensity = intensity
        self._maximum_refinement_iterations = maximum_refinement_iterations
        self._capture_timeout_milliseconds = capture_timeout_milliseconds
        self._active = False
        self._collecting = False
        self._projector_size = (0, 0)
        self._steps: list[_CaptureStep] = []
        self._step_index = 0
        self._captures: dict[str, GrayImage] = {}
        self._frame_accumulator: NDArray[np.float32] | None = None
        self._frame_count = 0
        self._detection: BoardDetection | None = None
        self._mapping: MappingResult | None = None
        self._validation_coordinates: list[tuple[float, float]] = []
        self._validation_expected: list[FloatArray] = []
        self._validation_measured: list[FloatArray] = []
        self._validation_widths: list[float] = []
        self._validation_index = -1
        self._refinement_iteration = 0
        self._progress_total = 0
        self._capture_token = 0
        self._run_token = 0

    @property
    def active(self) -> bool:
        return self._active

    def start(self, projector_width: int, projector_height: int) -> None:
        if self._active:
            raise RuntimeError("Projector calibration is already active.")
        if projector_width <= 1 or projector_height <= 1:
            raise ValueError("Projector dimensions must be greater than one.")

        self._active = True
        self._run_token += 1
        self._collecting = False
        self._projector_size = (projector_width, projector_height)
        self._captures.clear()
        self._detection = None
        self._mapping = None
        self._validation_expected.clear()
        self._validation_measured.clear()
        self._validation_widths.clear()
        self._validation_index = -1
        self._refinement_iteration = 0
        self._steps = self._build_capture_steps(
            projector_width,
            projector_height,
        )
        self._step_index = 0
        self._progress_total = len(self._steps) + 10
        self.active_changed.emit(True)
        self.status_changed.emit(
            "Starting projector calibration with the projector black."
        )
        self.progress_changed.emit(0, self._progress_total)
        self._display_current_step()

    def cancel(self) -> None:
        if not self._active:
            return
        self._active = False
        self._run_token += 1
        self._capture_token += 1
        self._collecting = False
        self._projector.show_black()
        self.active_changed.emit(False)
        self.status_changed.emit("Projector calibration cancelled.")

    def accept_frame(self, frame: ImageArray) -> None:
        if not self._active or not self._collecting:
            return
        gray = bgr_to_gray(frame)
        if self._frame_accumulator is None:
            self._frame_accumulator = np.zeros(gray.shape, dtype=np.float32)
        if gray.shape != self._frame_accumulator.shape:
            self._fail("Camera resolution changed during calibration.")
            return

        self._frame_accumulator += gray
        self._frame_count += 1
        if self._frame_count < self._frames_per_state:
            return

        averaged = np.rint(
            self._frame_accumulator / self._frame_count
        ).astype(np.uint8)
        self._collecting = False
        if self._validation_index >= 0:
            try:
                self._accept_validation_capture(averaged)
            except (CalibrationError, ValueError, cv2.error) as error:
                self._fail(str(error))
        else:
            step = self._steps[self._step_index]
            self._captures[step.key] = averaged
            self._step_index += 1
            self.progress_changed.emit(
                self._step_index,
                self._progress_total,
            )
            self._schedule(0, self._advance_capture)

    def _build_capture_steps(
        self,
        width: int,
        height: int,
    ) -> list[_CaptureStep]:
        steps = [
            _CaptureStep(
                "black",
                np.zeros((height, width), dtype=np.uint8),
                "Capturing projector-black board reference",
            ),
            _CaptureStep(
                "white",
                np.full(
                    (height, width),
                    self._intensity,
                    dtype=np.uint8,
                ),
                "Measuring projector coverage",
            ),
        ]
        for axis, length in (("x", width), ("y", height)):
            captured_bit_count = (
                gray_code_bit_count(length)
                - self._OMITTED_LEAST_SIGNIFICANT_BITS
            )
            for bit in range(captured_bit_count):
                for inverse in (False, True):
                    suffix = "inverse" if inverse else "normal"
                    steps.append(
                        _CaptureStep(
                            f"{axis}_{bit}_{suffix}",
                            gray_code_pattern(
                                width,
                                height,
                                axis,
                                bit,
                                inverse=inverse,
                                intensity=self._intensity,
                            ),
                            f"Capturing {axis.upper()} Gray-code bit "
                            f"{bit + 1}/{captured_bit_count} "
                            f"({suffix})",
                        )
                    )
        return steps

    def _display_current_step(self) -> None:
        if not self._active:
            return
        step = self._steps[self._step_index]
        self.status_changed.emit(step.description)
        self._projector.show_pattern(step.pattern)
        self._schedule(
            self._settle_milliseconds,
            self._begin_frame_collection,
        )

    def _begin_frame_collection(self) -> None:
        if not self._active:
            return
        self._frame_accumulator = None
        self._frame_count = 0
        self._collecting = True
        self._capture_token += 1
        token = self._capture_token
        run_token = self._run_token
        QTimer.singleShot(
            self._capture_timeout_milliseconds,
            lambda: self._capture_timed_out(token, run_token),
        )

    def _capture_timed_out(self, token: int, run_token: int) -> None:
        if (
            self._active
            and self._collecting
            and token == self._capture_token
            and run_token == self._run_token
        ):
            self._fail(
                "No complete set of fresh camera frames arrived within "
                f"{self._capture_timeout_milliseconds / 1000:.1f} seconds."
            )

    def _advance_capture(self) -> None:
        if not self._active:
            return
        if self._step_index < len(self._steps):
            self._display_current_step()
            return
        try:
            self._calculate_mapping()
            self._start_validation()
        except (CalibrationError, ValueError, cv2.error) as error:
            self._fail(str(error))

    def _calculate_mapping(self) -> None:
        black = self._captures["black"]
        illuminated = self._captures["white"]
        detection = self._detector.detect(black)
        if detection is None:
            raise CalibrationError(
                "The empty board was not detected with the projector black."
            )
        coverage = measure_coverage(black, illuminated, detection)

        width, height = self._projector_size
        horizontal_pairs = [
            (
                self._captures[f"x_{bit}_normal"],
                self._captures[f"x_{bit}_inverse"],
            )
            for bit in range(
                gray_code_bit_count(width)
                - self._OMITTED_LEAST_SIGNIFICANT_BITS
            )
        ]
        vertical_pairs = [
            (
                self._captures[f"y_{bit}_normal"],
                self._captures[f"y_{bit}_inverse"],
            )
            for bit in range(
                gray_code_bit_count(height)
                - self._OMITTED_LEAST_SIGNIFICANT_BITS
            )
        ]
        projector_x, projector_y, valid = decode_gray_code(
            horizontal_pairs,
            vertical_pairs,
            coverage.response_mask,
            horizontal_omitted_bits=self._OMITTED_LEAST_SIGNIFICANT_BITS,
            vertical_omitted_bits=self._OMITTED_LEAST_SIGNIFICANT_BITS,
        )
        mapping = fit_projector_mapping(
            detection,
            projector_x,
            projector_y,
            valid,
            self._projector_size,
            coverage.coverage_ratio,
        )
        self._detection = detection
        self._mapping = mapping
        self._projector.set_projector_grid(mapping.projector_grid)

    def _start_validation(self) -> None:
        if self._detection is None or self._mapping is None:
            raise RuntimeError("Calibration mapping is not available.")
        self.status_changed.emit("Mapping complete; refreshing black reference.")
        self._validation_coordinates = [
            (0.5, 0.5),
            (3.5, 0.5),
            (7.5, 0.5),
            (0.5, 3.5),
            (3.5, 3.5),
            (7.5, 3.5),
            (0.5, 7.5),
            (3.5, 7.5),
            (7.5, 7.5),
        ]
        self._validation_index = 0
        self._show_validation_black()

    def _show_validation_black(self) -> None:
        if not self._active:
            return
        width, height = self._projector_size
        self._projector.show_pattern(
            np.zeros((height, width), dtype=np.uint8)
        )
        self._schedule(
            self._settle_milliseconds,
            self._begin_frame_collection,
        )

    def _show_validation_target(self) -> None:
        if (
            not self._active
            or self._mapping is None
            or self._detection is None
        ):
            return
        board_x, board_y = self._validation_coordinates[
            self._validation_index
        ]
        projector_point = interpolate_grid_point(
            self._mapping.projector_grid,
            board_x,
            board_y,
        )
        expected = interpolate_grid_point(
            self._mapping.camera_grid,
            board_x,
            board_y,
        )
        square_width = self._local_square_width(board_x, board_y)
        self._validation_expected.append(expected)
        self._validation_widths.append(square_width)

        projector_square_width = self._local_projector_square_width(
            board_x,
            board_y,
        )
        radius = max(4, min(24, round(projector_square_width * 0.18)))
        width, height = self._projector_size
        pattern = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(
            pattern,
            (
                round(float(projector_point[0])),
                round(float(projector_point[1])),
            ),
            radius,
            self._intensity,
            -1,
            cv2.LINE_AA,
        )
        self.status_changed.emit(
            f"Validating target {self._validation_index + 1}/9"
        )
        self._projector.show_pattern(pattern)
        self._schedule(
            self._settle_milliseconds,
            self._begin_frame_collection,
        )

    def _show_validation_target_safely(self) -> None:
        try:
            self._show_validation_target()
        except (CalibrationError, ValueError, cv2.error) as error:
            self._fail(str(error))

    def _accept_validation_capture(self, averaged: GrayImage) -> None:
        if self._validation_index == 0 and "validation_black" not in self._captures:
            self._captures["validation_black"] = averaged
            self._show_validation_target_safely()
            return

        expected = self._validation_expected[-1]
        square_width = self._validation_widths[-1]
        measured = locate_validation_target(
            self._captures["validation_black"],
            averaged,
            expected,
            max(8, round(square_width * 0.75)),
        )
        self._validation_measured.append(measured)
        self._validation_index += 1
        completed_steps = (
            self._progress_total - 10 + self._validation_index + 1
        )
        self.progress_changed.emit(completed_steps, self._progress_total)
        if self._validation_index < len(self._validation_coordinates):
            self._schedule(0, self._show_validation_target_safely)
            return

        try:
            self._finish_validation()
        except (CalibrationError, ValueError) as error:
            self._fail(str(error))

    def _finish_validation(self) -> None:
        if self._detection is None or self._mapping is None:
            raise RuntimeError("Calibration mapping is not available.")
        measured = np.asarray(self._validation_measured, dtype=np.float32)
        expected = np.asarray(self._validation_expected, dtype=np.float32)
        widths = np.asarray(self._validation_widths, dtype=np.float32)
        validation = validation_result(measured, expected, widths)
        if validation.maximum_square_fraction >= 0.1:
            if (
                self._refinement_iteration
                < self._maximum_refinement_iterations
            ):
                self._refine_and_repeat_validation(measured)
                return
            raise CalibrationError(
                f"Validation error is "
                f"{validation.maximum_square_fraction:.1%} of a square; "
                "the maximum allowed is 10%."
            )

        result = CalibrationResult(
            board_detection=self._detection,
            mapping=self._mapping,
            validation=validation,
        )
        self._active = False
        self._collecting = False
        self._projector.show_grid()
        self.active_changed.emit(False)
        self.progress_changed.emit(
            self._progress_total,
            self._progress_total,
        )
        self.status_changed.emit(
            f"Calibration succeeded: {validation.maximum_error:.2f} px "
            f"maximum validation error."
        )
        self.completed.emit(result)

    def _refine_and_repeat_validation(
        self,
        measured_points: FloatArray,
    ) -> None:
        if self._mapping is None:
            raise RuntimeError("Calibration mapping is not available.")
        board_points = np.asarray(
            self._validation_coordinates,
            dtype=np.float32,
        )
        expected_points = np.asarray(
            self._validation_expected,
            dtype=np.float32,
        )
        refined_grid = refine_projector_mesh(
            self._mapping.projector_grid,
            self._mapping.camera_to_projector_homography,
            board_points,
            expected_points,
            measured_points,
            self._projector_size,
        )
        self._mapping = replace(
            self._mapping,
            projector_grid=refined_grid,
        )
        self._projector.set_projector_grid(refined_grid)
        self._refinement_iteration += 1
        self._progress_total += 10
        self.progress_changed.emit(
            self._progress_total - 10,
            self._progress_total,
        )
        self.status_changed.emit(
            f"Refining alignment from validation measurements "
            f"({self._refinement_iteration}/"
            f"{self._maximum_refinement_iterations})."
        )
        self._captures.pop("validation_black", None)
        self._validation_expected.clear()
        self._validation_measured.clear()
        self._validation_widths.clear()
        self._validation_index = 0
        self._schedule(0, self._show_validation_black)

    def _local_square_width(self, x: float, y: float) -> float:
        if self._mapping is None:
            raise RuntimeError("Calibration mapping is not available.")
        center = interpolate_grid_point(self._mapping.camera_grid, x, y)
        left = interpolate_grid_point(
            self._mapping.camera_grid,
            max(0.0, x - 0.5),
            y,
        )
        right = interpolate_grid_point(
            self._mapping.camera_grid,
            min(8.0, x + 0.5),
            y,
        )
        top = interpolate_grid_point(
            self._mapping.camera_grid,
            x,
            max(0.0, y - 0.5),
        )
        bottom = interpolate_grid_point(
            self._mapping.camera_grid,
            x,
            min(8.0, y + 0.5),
        )
        horizontal = float(np.linalg.norm(right - left))
        vertical = float(np.linalg.norm(bottom - top))
        if horizontal <= 0 or vertical <= 0 or not np.all(np.isfinite(center)):
            raise CalibrationError("Could not measure local board square size.")
        return (horizontal + vertical) / 2.0

    def _local_projector_square_width(self, x: float, y: float) -> float:
        if self._mapping is None:
            raise RuntimeError("Calibration mapping is not available.")
        left = interpolate_grid_point(
            self._mapping.projector_grid,
            max(0.0, x - 0.5),
            y,
        )
        right = interpolate_grid_point(
            self._mapping.projector_grid,
            min(8.0, x + 0.5),
            y,
        )
        top = interpolate_grid_point(
            self._mapping.projector_grid,
            x,
            max(0.0, y - 0.5),
        )
        bottom = interpolate_grid_point(
            self._mapping.projector_grid,
            x,
            min(8.0, y + 0.5),
        )
        return (
            float(np.linalg.norm(right - left))
            + float(np.linalg.norm(bottom - top))
        ) / 2.0

    def _fail(self, message: str) -> None:
        self._active = False
        self._collecting = False
        self._run_token += 1
        self._capture_token += 1
        self._projector.show_black()
        self.active_changed.emit(False)
        self.status_changed.emit(f"Calibration failed: {message}")
        self.failed.emit(message)

    def _schedule(
        self,
        delay_milliseconds: int,
        callback: Callable[[], None],
    ) -> None:
        run_token = self._run_token

        def guarded_callback() -> None:
            if self._active and run_token == self._run_token:
                callback()

        QTimer.singleShot(delay_milliseconds, guarded_callback)
