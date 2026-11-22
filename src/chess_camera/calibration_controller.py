from collections.abc import Callable
from dataclasses import dataclass, replace
from time import monotonic

import cv2
import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QObject, QTimer, Signal

from .board_detector import BoardDetection, BoardDetector, FloatArray, ImageArray
from .occupancy import empty_square_candidates
from .projector import ProjectorWindow
from .projector_calibration import (
    BoardLatticeHiddenError,
    CalibrationError,
    CodedPattern,
    GrayImage,
    MappingResult,
    PlaneSupportSelection,
    ValidationResult,
    board_mask,
    board_circle_polygon,
    bgr_to_gray,
    camera_clearance_image,
    clearance_squares,
    coded_marker_pattern,
    decode_gray_code,
    expanded_projector_board_mask,
    fit_coded_pattern_mapping,
    fit_projector_mapping,
    gray_code_bit_count,
    gray_code_pair_quality,
    gray_code_pattern,
    interpolate_grid_point,
    locate_validation_target,
    measure_coverage,
    plane_support_overlay,
    quick_realign_mapping,
    refine_projector_mesh,
    remap_projector_mapping,
    require_board_plane_support,
    select_board_plane_support,
    select_validation_squares,
    support_map_pattern,
    square_interior_labels,
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


_STARTING_POSITION_CORNER_TARGETS = [
    [(1.0, 2.0), (1.25, 2.0)],
    [(7.0, 2.0), (6.75, 2.0)],
    [(1.0, 6.0), (1.25, 6.0)],
    [(7.0, 6.0), (6.75, 6.0)],
]
_STARTING_POSITION_EDGE_TARGETS = [
    [(4.0, 2.0), (4.25, 2.0)],
    [(7.0, 4.0), (7.0, 4.25)],
    [(4.0, 6.0), (3.75, 6.0)],
    [(1.0, 4.0), (1.0, 3.75)],
]
_STARTING_POSITION_CENTER_TARGET = [(4.0, 4.0), (4.25, 4.0)]


class ProjectorCalibrationController(QObject):
    status_changed = Signal(str)
    progress_changed = Signal(int, int)
    completed = Signal(object)
    failed = Signal(str)
    active_changed = Signal(bool)
    _OMITTED_LEAST_SIGNIFICANT_BITS = 1
    _ADAPTIVE_FORCED_CAPTURE_MILLISECONDS = 600
    _MAXIMUM_PAIR_RECAPTURES = 2
    _MINIMUM_VALIDATION_HITS = 9

    def __init__(
        self,
        projector: ProjectorWindow,
        detector: BoardDetector,
        *,
        settle_milliseconds: int = 1000,
        frames_per_state: int = 3,
        intensity: int = 255,
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
        self._validation_candidates: list[list[tuple[float, float]]] = []
        self._validation_candidate_index = 0
        self._validation_expected: list[FloatArray] = []
        self._validation_measured: list[FloatArray] = []
        self._validation_widths: list[float] = []
        self._validation_index = -1
        self._refinement_iteration = 0
        self._progress_total = 0
        self._validation_progress_base = 0
        self._capture_token = 0
        self._run_token = 0
        self._quick_verification = False
        self._validation_only = False
        self._coded_pattern_mode = False
        self._adaptive_cropped_mode = False
        self._adaptive_phase = ""
        self._adaptive_rough_mapping: MappingResult | None = None
        self._adaptive_discovery_candidate: GrayImage | None = None
        self._adaptive_projector_mask: NDArray[np.bool_] | None = None
        self._adaptive_previous_gray: GrayImage | None = None
        self._adaptive_candidate_gray: GrayImage | None = None
        self._adaptive_expected_change_ratio = 0.1
        self._adaptive_gray_fallback = False
        self._adaptive_black_color: ImageArray | None = None
        self._adaptive_empty_squares: NDArray[np.bool_] | None = None
        self._plane_support: PlaneSupportSelection | None = None
        self._plane_support_image: ImageArray | None = None
        self._adaptive_rough_homography: FloatArray | None = None
        self._adaptive_step_shown_at = 0.0
        self._gray_capture_log: list[dict[str, object]] = []
        self._gray_pair_log: list[dict[str, object]] = []
        self._gray_pair_recaptures: dict[str, int] = {}
        self._gray_weakest_pair: tuple[float, GrayImage, GrayImage] | None = (
            None
        )
        self._clearance_squares: tuple[tuple[int, int], ...] = ()
        self._lattice_hidden: BoardLatticeHiddenError | None = None
        self._validation_log: list[dict[str, object]] = []
        self._capture_progress_base = 0
        self._coded_pattern: CodedPattern | None = None
        self._inverse_coded_pattern: CodedPattern | None = None
        self._quick_phase = ""

    @property
    def active(self) -> bool:
        return self._active

    @property
    def quick_verification(self) -> bool:
        return self._quick_verification

    @property
    def validation_only(self) -> bool:
        return self._validation_only

    @property
    def coded_pattern_mode(self) -> bool:
        return self._coded_pattern_mode

    @property
    def adaptive_cropped_mode(self) -> bool:
        return self._adaptive_cropped_mode

    @property
    def diagnostic_captures(self) -> dict[str, ImageArray]:
        captures = {
            name.replace("_", "-"): image.copy()
            for name, image in self._captures.items()
            if name in {"black", "white", "coded", "coded_inverse"}
        }
        if self._adaptive_cropped_mode and self._plane_support_image is not None:
            captures["plane-support"] = self._plane_support_image.copy()
        if self._adaptive_cropped_mode and self._gray_weakest_pair is not None:
            _score, normal, inverse = self._gray_weakest_pair
            captures["gray-weakest-normal"] = normal.copy()
            captures["gray-weakest-inverse"] = inverse.copy()
        return captures

    @property
    def clearance_squares(self) -> tuple[tuple[int, int], ...]:
        return self._clearance_squares

    @property
    def camera_clearance_view(self) -> ImageArray | None:
        """Black reference with the squares to clear outlined in red."""
        hidden = self._lattice_hidden
        black = self._captures.get("black")
        if not self._adaptive_cropped_mode or hidden is None or black is None:
            return None
        return camera_clearance_image(
            black,
            hidden.rough_detection.grid_points,
            hidden.clear_axis,
        )

    @property
    def support_map(self) -> NDArray[np.uint8] | None:
        """RGB board map: usable squares green, squares to clear red."""
        selection = self._plane_support
        mapping = self._mapping or self._adaptive_rough_mapping
        if not self._adaptive_cropped_mode or selection is None or (
            mapping is None
        ):
            return None
        usable = tuple(
            (int(row), int(column))
            for row, column in np.argwhere(
                selection.square_status == "retained"
            )
        )
        return support_map_pattern(
            self._projector_size,
            mapping.projector_grid,
            usable,
            self._clearance_squares,
            self._intensity,
        )

    @property
    def diagnostic_details(self) -> dict[str, object]:
        if not self._adaptive_cropped_mode:
            return {}
        details: dict[str, object] = {
            "occupancy_empty_squares": (
                None
                if self._adaptive_empty_squares is None
                else int(np.count_nonzero(self._adaptive_empty_squares))
            ),
            "gray_code_captures": [
                dict(entry) for entry in self._gray_capture_log
            ],
            "gray_code_pairs": [dict(entry) for entry in self._gray_pair_log],
        }
        if self._plane_support is not None:
            details["plane_support"] = self._plane_support.details()
        if self._clearance_squares:
            details["clearance_squares"] = [
                [row, column] for row, column in self._clearance_squares
            ]
        if self._validation_log:
            details["validation_targets"] = [
                dict(entry) for entry in self._validation_log
            ]
        return details

    def start(self, projector_width: int, projector_height: int) -> None:
        if self._active:
            raise RuntimeError("Projector calibration is already active.")
        if projector_width <= 1 or projector_height <= 1:
            raise ValueError("Projector dimensions must be greater than one.")

        self._active = True
        self._run_token += 1
        self._quick_verification = False
        self._validation_only = False
        self._coded_pattern_mode = False
        self._adaptive_cropped_mode = False
        self._adaptive_phase = ""
        self._adaptive_rough_mapping = None
        self._capture_progress_base = 0
        self._coded_pattern = None
        self._inverse_coded_pattern = None
        self._quick_phase = ""
        self._collecting = False
        self._projector_size = (projector_width, projector_height)
        self._captures.clear()
        self._detection = None
        self._mapping = None
        self._validation_expected.clear()
        self._validation_measured.clear()
        self._validation_widths.clear()
        self._validation_candidates.clear()
        self._validation_candidate_index = 0
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

    def start_coded_pattern(
        self,
        projector_width: int,
        projector_height: int,
    ) -> None:
        if self._active:
            raise RuntimeError("Projector calibration is already active.")
        if projector_width <= 1 or projector_height <= 1:
            raise ValueError("Projector dimensions must be greater than one.")

        pattern = coded_marker_pattern(
            projector_width,
            projector_height,
            intensity=self._intensity,
        )
        inverse_pattern = coded_marker_pattern(
            projector_width,
            projector_height,
            intensity=self._intensity,
            inverse=True,
            staggered=True,
        )
        self._active = True
        self._run_token += 1
        self._quick_verification = False
        self._validation_only = False
        self._coded_pattern_mode = True
        self._adaptive_cropped_mode = False
        self._adaptive_phase = ""
        self._adaptive_rough_mapping = None
        self._capture_progress_base = 0
        self._coded_pattern = pattern
        self._inverse_coded_pattern = inverse_pattern
        self._quick_phase = ""
        self._collecting = False
        self._projector_size = (projector_width, projector_height)
        self._captures.clear()
        self._detection = None
        self._mapping = None
        self._validation_expected.clear()
        self._validation_measured.clear()
        self._validation_widths.clear()
        self._validation_candidates.clear()
        self._validation_candidate_index = 0
        self._validation_index = -1
        self._refinement_iteration = 0
        self._steps = [
            _CaptureStep(
                "black",
                np.zeros(
                    (projector_height, projector_width),
                    dtype=np.uint8,
                ),
                "Capturing populated-board black reference",
            ),
            _CaptureStep(
                "coded",
                pattern.image,
                "Capturing uniquely coded projector pattern",
            ),
            _CaptureStep(
                "coded_inverse",
                inverse_pattern.image,
                "Capturing complementary coded projector pattern",
            ),
        ]
        self._step_index = 0
        self._progress_total = len(self._steps) + 10
        self.active_changed.emit(True)
        self.status_changed.emit(
            "Starting single-shot coded-pattern calibration."
        )
        self.progress_changed.emit(0, self._progress_total)
        self._display_current_step()

    def start_adaptive_cropped(
        self,
        projector_width: int,
        projector_height: int,
    ) -> None:
        if self._active:
            raise RuntimeError("Projector calibration is already active.")
        if projector_width <= 1 or projector_height <= 1:
            raise ValueError("Projector dimensions must be greater than one.")

        pattern = coded_marker_pattern(
            projector_width,
            projector_height,
            intensity=self._intensity,
        )
        self._active = True
        self._run_token += 1
        self._quick_verification = False
        self._validation_only = False
        self._coded_pattern_mode = False
        self._adaptive_cropped_mode = True
        self._adaptive_phase = "discovery"
        self._adaptive_rough_mapping = None
        self._adaptive_discovery_candidate = None
        self._adaptive_projector_mask = None
        self._adaptive_previous_gray = None
        self._adaptive_candidate_gray = None
        self._adaptive_expected_change_ratio = 0.1
        self._adaptive_gray_fallback = False
        self._adaptive_black_color = None
        self._adaptive_empty_squares = None
        self._plane_support = None
        self._plane_support_image = None
        self._adaptive_rough_homography = None
        self._adaptive_step_shown_at = 0.0
        self._gray_capture_log = []
        self._gray_pair_log = []
        self._gray_pair_recaptures = {}
        self._gray_weakest_pair = None
        self._clearance_squares = ()
        self._lattice_hidden = None
        self._validation_log = []
        self._coded_pattern = pattern
        self._inverse_coded_pattern = None
        self._quick_phase = ""
        self._collecting = False
        self._capture_progress_base = 0
        self._projector_size = (projector_width, projector_height)
        self._captures.clear()
        self._detection = None
        self._mapping = None
        self._validation_expected.clear()
        self._validation_measured.clear()
        self._validation_widths.clear()
        self._validation_candidates.clear()
        self._validation_candidate_index = 0
        self._validation_index = -1
        self._refinement_iteration = 0
        self._steps = [
            _CaptureStep(
                "black",
                np.zeros(
                    (projector_height, projector_width),
                    dtype=np.uint8,
                ),
                "Capturing populated-board black reference",
            ),
            _CaptureStep(
                "coded",
                pattern.image,
                "Capturing brief coded discovery pattern",
            ),
        ]
        self._step_index = 0
        gray_steps = sum(
            2
            * (
                gray_code_bit_count(length)
                - self._OMITTED_LEAST_SIGNIFICANT_BITS
            )
            for length in (projector_width, projector_height)
        )
        self._progress_total = len(self._steps) + gray_steps + 10
        self.active_changed.emit(True)
        self.status_changed.emit(
            "Starting board-cropped calibration with the projector black."
        )
        self.progress_changed.emit(0, self._progress_total)
        self._display_current_step()

    def start_verification(
        self,
        result: CalibrationResult,
        projector_width: int,
        projector_height: int,
    ) -> None:
        if self._active:
            raise RuntimeError("Projector calibration is already active.")
        if projector_width <= 1 or projector_height <= 1:
            raise ValueError("Projector dimensions must be greater than one.")

        self._active = True
        self._run_token += 1
        self._quick_verification = True
        self._validation_only = False
        self._coded_pattern_mode = False
        self._adaptive_cropped_mode = False
        self._adaptive_phase = ""
        self._adaptive_rough_mapping = None
        self._capture_progress_base = 0
        self._coded_pattern = None
        self._inverse_coded_pattern = None
        self._quick_phase = "fit"
        self._collecting = False
        self._projector_size = (projector_width, projector_height)
        self._captures.clear()
        self._detection = result.board_detection
        self._mapping = result.mapping
        self._validation_expected.clear()
        self._validation_measured.clear()
        self._validation_widths.clear()
        self._validation_candidates.clear()
        self._validation_candidate_index = 0
        self._validation_index = -1
        self._refinement_iteration = 0
        self._steps = [
            _CaptureStep(
                "black",
                np.zeros(
                    (projector_height, projector_width),
                    dtype=np.uint8,
                ),
                "Capturing projector-black quick-verification reference",
            )
        ]
        self._step_index = 0
        self._progress_total = 12
        self.active_changed.emit(True)
        self.status_changed.emit("Starting measured quick realignment.")
        self.progress_changed.emit(0, self._progress_total)
        self._display_current_step()

    def start_validation_only(
        self,
        result: CalibrationResult,
        projector_width: int,
        projector_height: int,
    ) -> None:
        if self._active:
            raise RuntimeError("Projector calibration is already active.")
        if projector_width <= 1 or projector_height <= 1:
            raise ValueError("Projector dimensions must be greater than one.")

        self._active = True
        self._run_token += 1
        self._quick_verification = False
        self._validation_only = True
        self._coded_pattern_mode = False
        self._adaptive_cropped_mode = False
        self._adaptive_phase = ""
        self._adaptive_rough_mapping = None
        self._capture_progress_base = 0
        self._coded_pattern = None
        self._inverse_coded_pattern = None
        self._quick_phase = ""
        self._collecting = False
        self._projector_size = (projector_width, projector_height)
        self._captures.clear()
        self._detection = result.board_detection
        self._mapping = result.mapping
        self._validation_expected.clear()
        self._validation_measured.clear()
        self._validation_widths.clear()
        self._validation_candidates.clear()
        self._validation_candidate_index = 0
        self._validation_index = -1
        self._refinement_iteration = 0
        self._steps.clear()
        self._progress_total = 10
        self._projector.set_projector_grid(result.mapping.projector_grid)
        self.active_changed.emit(True)
        self.status_changed.emit(
            "Starting validation of the existing projector calibration."
        )
        self.progress_changed.emit(0, self._progress_total)
        self._start_validation()

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
        if (
            self._adaptive_cropped_mode
            and self._adaptive_phase == "discovery"
            and self._steps[self._step_index].key == "coded"
        ):
            averaged = self._accept_adaptive_discovery_frame(gray)
            if averaged is None:
                return
            rough_detection = self._detection
            if rough_detection is None:
                self._fail("The populated board was not detected in black.")
                return
            self._collecting = False
            self._adaptive_discovery_candidate = averaged
            self._projector.show_black()
            self._schedule(25, self._fit_adaptive_discovery_candidate)
            return
        if (
            self._adaptive_cropped_mode
            and self._adaptive_phase == "gray-code"
        ):
            if self._adaptive_gray_fallback:
                if self._frame_accumulator is None:
                    self._frame_accumulator = np.zeros(
                        gray.shape,
                        dtype=np.float32,
                    )
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
                self._log_gray_capture("forced", averaged)
                self._adaptive_previous_gray = averaged
                self._complete_frame_capture(averaged)
                return
            averaged = self._accept_adaptive_gray_frame(gray)
            if averaged is None:
                return
            self._complete_frame_capture(averaged)
            return
        if self._frame_accumulator is None:
            self._frame_accumulator = np.zeros(gray.shape, dtype=np.float32)
        if gray.shape != self._frame_accumulator.shape:
            self._fail("Camera resolution changed during calibration.")
            return

        if (
            self._adaptive_cropped_mode
            and self._adaptive_phase == "discovery"
            and self._steps[self._step_index].key == "black"
            and frame.ndim == 3
        ):
            self._adaptive_black_color = frame
        self._frame_accumulator += gray
        self._frame_count += 1
        required_frames = (
            1
            if (
                self._adaptive_cropped_mode
                and self._adaptive_phase == "discovery"
                and self._steps[self._step_index].key != "black"
            )
            else self._frames_per_state
        )
        if self._frame_count < required_frames:
            return

        averaged = np.rint(
            self._frame_accumulator / self._frame_count
        ).astype(np.uint8)
        self._complete_frame_capture(averaged)

    def _accept_adaptive_discovery_frame(
        self,
        gray: GrayImage,
    ) -> GrayImage | None:
        black = self._captures["black"]
        detection = self._detection
        if gray.shape != black.shape:
            self._fail("Camera resolution changed during calibration.")
            return None
        if detection is None:
            self._fail("Adaptive discovery synchronization is unavailable.")
            return None
        usable = board_mask(gray.shape, detection) != 0
        if not np.any(usable):
            self._fail("The detected board has no usable camera pixels.")
            return None

        changed_ratio = float(
            np.mean(cv2.absdiff(black, gray)[usable] > 15)
        )
        if changed_ratio < 0.005:
            self._adaptive_candidate_gray = None
            self._frame_accumulator = None
            self._frame_count = 0
            return None

        previous = self._adaptive_candidate_gray
        if previous is not None:
            stable_ratio = float(
                np.mean(cv2.absdiff(previous, gray)[usable] > 15)
            )
            if stable_ratio > 0.02:
                self._frame_accumulator = None
                self._frame_count = 0

        if self._frame_accumulator is None:
            self._frame_accumulator = np.zeros(
                gray.shape,
                dtype=np.float32,
            )
        self._frame_accumulator += gray
        self._frame_count += 1
        self._adaptive_candidate_gray = gray.copy()
        if self._frame_count < self._frames_per_state:
            return None

        return np.rint(
            self._frame_accumulator / self._frame_count
        ).astype(np.uint8)

    def _accept_adaptive_gray_frame(
        self,
        gray: GrayImage,
    ) -> GrayImage | None:
        previous = self._adaptive_previous_gray
        detection = self._detection
        if previous is None or detection is None:
            self._fail("Adaptive Gray-code synchronization is unavailable.")
            return None
        if gray.shape != previous.shape:
            self._fail("Camera resolution changed during calibration.")
            return None
        usable = board_mask(gray.shape, detection) != 0
        if not np.any(usable):
            self._fail("The detected board has no usable camera pixels.")
            return None

        difference = cv2.absdiff(previous, gray)
        changed_ratio = float(np.mean(difference[usable] > 15))
        required_ratio = max(
            0.005,
            min(0.04, self._adaptive_expected_change_ratio * 0.25),
        )
        if changed_ratio < required_ratio:
            return None

        candidate = self._adaptive_candidate_gray
        if candidate is None:
            self._adaptive_candidate_gray = gray.copy()
            return None

        candidate_difference = cv2.absdiff(candidate, gray)
        stable_ratio = float(np.mean(candidate_difference[usable] > 15))
        if stable_ratio > 0.02:
            self._adaptive_candidate_gray = gray.copy()
            return None

        averaged = np.rint(
            (
                candidate.astype(np.float32)
                + gray.astype(np.float32)
            )
            / 2.0
        ).astype(np.uint8)
        self._log_gray_capture("transition", averaged)
        self._adaptive_previous_gray = averaged
        self._adaptive_candidate_gray = None
        return averaged

    def _log_gray_capture(self, mode: str, averaged: GrayImage) -> None:
        # Validation captures follow the Gray-code steps in the same phase.
        if self._validation_index >= 0 or self._step_index >= len(
            self._steps
        ):
            return
        previous = self._adaptive_previous_gray
        detection = self._detection
        changed_ratio: float | None = None
        if (
            previous is not None
            and detection is not None
            and previous.shape == averaged.shape
        ):
            usable = board_mask(averaged.shape, detection) != 0
            if np.any(usable):
                changed_ratio = round(
                    float(
                        np.mean(cv2.absdiff(previous, averaged)[usable] > 15)
                    ),
                    4,
                )
        self._gray_capture_log.append(
            {
                "key": self._steps[self._step_index].key,
                "mode": mode,
                "changed_ratio": changed_ratio,
                "elapsed_milliseconds": round(
                    (monotonic() - self._adaptive_step_shown_at) * 1000
                ),
            }
        )

    def _complete_frame_capture(self, averaged: GrayImage) -> None:
        self._collecting = False
        if self._validation_index >= 0:
            try:
                self._accept_validation_capture(averaged)
            except (CalibrationError, ValueError, cv2.error) as error:
                self._fail(str(error))
        else:
            step = self._steps[self._step_index]
            self._captures[step.key] = averaged
            if (
                self._adaptive_cropped_mode
                and self._adaptive_phase == "discovery"
                and step.key == "black"
            ):
                detection = self._detector.detect(averaged)
                if detection is None:
                    self._fail(
                        "The populated board was not detected with the "
                        "projector black."
                    )
                    return
                self._detection = detection
                self._adaptive_empty_squares = self._classify_empty_squares(
                    detection
                )
            if (
                self._adaptive_cropped_mode
                and self._adaptive_phase == "gray-code"
                and step.key.endswith("_inverse")
                and not self._verify_gray_code_pair(step.key)
            ):
                return
            self._step_index += 1
            self.progress_changed.emit(
                self._capture_progress_base + self._step_index,
                self._progress_total,
            )
            self._schedule(0, self._advance_capture)

    def _verify_gray_code_pair(self, inverse_key: str) -> bool:
        """Check a captured pair; recapture or fail when it is unreliable."""
        pair_key = inverse_key.removesuffix("_inverse")
        normal_key = f"{pair_key}_normal"
        detection = self._detection
        homography = self._adaptive_rough_homography
        if detection is None or homography is None:
            self._fail("Adaptive Gray-code verification is unavailable.")
            return False
        normal = self._captures[normal_key]
        inverse = self._captures[inverse_key]
        quality = gray_code_pair_quality(
            normal,
            inverse,
            board_mask(normal.shape, detection) != 0,
            self._steps[self._step_index - 1].pattern,
            homography,
        )
        recaptures = self._gray_pair_recaptures.get(pair_key, 0)
        self._gray_pair_log.append(
            {
                "pair": pair_key,
                "contrast": round(quality.contrast, 4),
                "agreement": (
                    None
                    if quality.agreement is None
                    else round(quality.agreement, 4)
                ),
                "checked_ratio": round(quality.checked_ratio, 4),
                "recaptures": recaptures,
                "accepted": quality.acceptable,
            }
        )
        score = min(
            quality.contrast,
            1.0 if quality.agreement is None else quality.agreement,
        )
        if self._gray_weakest_pair is None or score < self._gray_weakest_pair[0]:
            self._gray_weakest_pair = (score, normal.copy(), inverse.copy())
        if quality.acceptable:
            return True

        axis, bit = pair_key.split("_")
        if recaptures < self._MAXIMUM_PAIR_RECAPTURES:
            self._gray_pair_recaptures[pair_key] = recaptures + 1
            self._step_index -= 1
            # The projector still shows the inverse pattern, so the next
            # accepted frame must differ from the inverse capture.
            self._adaptive_previous_gray = inverse.copy()
            self.status_changed.emit(
                f"Recapturing {axis.upper()} Gray-code bit {int(bit) + 1}; "
                "the camera did not see the projected pattern reliably."
            )
            self._schedule(0, self._advance_capture)
            return False

        agreement = (
            "not checked"
            if quality.agreement is None
            else f"{quality.agreement:.0%}"
        )
        self._fail(
            f"The camera did not see projected {axis.upper()} Gray-code bit "
            f"{int(bit) + 1} reliably after {recaptures} recaptures "
            f"(contrast {quality.contrast:.0%}, pattern agreement "
            f"{agreement}). Keep hands and shadows away from the board and "
            "check that the projector image is in focus."
        )
        return False

    def _classify_empty_squares(
        self,
        detection: BoardDetection,
    ) -> NDArray[np.bool_] | None:
        color = self._adaptive_black_color
        if color is None:
            return None
        try:
            return empty_square_candidates(color, detection.grid_points)
        except (ValueError, cv2.error):
            return None

    def _allowed_marker_mask(
        self,
        image_shape: tuple[int, int],
        detection: BoardDetection,
    ) -> NDArray[np.bool_] | None:
        empty_squares = self._adaptive_empty_squares
        if empty_squares is None or not np.any(empty_squares):
            return None
        labels = square_interior_labels(image_shape, detection.grid_points)
        return np.isin(labels, np.flatnonzero(empty_squares.reshape(-1)))

    def _fit_adaptive_discovery_candidate(self) -> None:
        candidate = self._adaptive_discovery_candidate
        rough_detection = self._detection
        pattern = self._coded_pattern
        if candidate is None or rough_detection is None or pattern is None:
            self._fail("Coded discovery candidate is not available.")
            return
        try:
            detection, mapping = fit_coded_pattern_mapping(
                self._captures["black"],
                candidate,
                rough_detection,
                pattern,
                self._projector_size,
                require_quadrant_coverage=False,
                allowed_marker_mask=self._allowed_marker_mask(
                    candidate.shape,
                    rough_detection,
                ),
            )
        except (CalibrationError, ValueError, cv2.error) as error:
            self._adaptive_discovery_candidate = None
            if isinstance(error, BoardLatticeHiddenError):
                self._lattice_hidden = error
            self._fail(f"Coded discovery failed: {error}")
            return

        self._adaptive_discovery_candidate = None
        self._detection = detection
        self._adaptive_rough_mapping = mapping
        self._captures["coded"] = candidate
        self._step_index += 1
        self.progress_changed.emit(
            self._capture_progress_base + self._step_index,
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

    def _build_cropped_gray_code_steps(
        self,
        projector_mask: NDArray[np.bool_],
    ) -> list[_CaptureStep]:
        width, height = self._projector_size
        if projector_mask.shape != (height, width):
            raise ValueError(
                "Projector board mask must match projector dimensions."
            )
        steps: list[_CaptureStep] = []
        for axis, length in (("x", width), ("y", height)):
            captured_bit_count = (
                gray_code_bit_count(length)
                - self._OMITTED_LEAST_SIGNIFICANT_BITS
            )
            for bit in range(captured_bit_count):
                for inverse in (False, True):
                    suffix = "inverse" if inverse else "normal"
                    pattern = gray_code_pattern(
                        width,
                        height,
                        axis,
                        bit,
                        inverse=inverse,
                        intensity=self._intensity,
                    )
                    steps.append(
                        _CaptureStep(
                            f"{axis}_{bit}_{suffix}",
                            np.where(projector_mask, pattern, 0).astype(
                                np.uint8
                            ),
                            f"Capturing cropped {axis.upper()} Gray-code bit "
                            f"{bit + 1}/{captured_bit_count} ({suffix})",
                        )
                    )
        return steps

    def _display_current_step(self) -> None:
        if not self._active:
            return
        step = self._steps[self._step_index]
        if (
            self._adaptive_cropped_mode
            and self._adaptive_phase == "gray-code"
        ):
            self._adaptive_candidate_gray = None
            self._adaptive_gray_fallback = False
            projector_mask = self._adaptive_projector_mask
            if projector_mask is None:
                self._fail("Adaptive projector board mask is unavailable.")
                return
            if self._step_index > 0:
                previous_step = self._steps[self._step_index - 1]
                projector_changed = (
                    step.pattern[projector_mask]
                    != previous_step.pattern[projector_mask]
                )
                previous_capture = self._captures[previous_step.key]
            else:
                projector_changed = step.pattern[projector_mask] != 0
                previous_capture = self._captures["black"]
            self._adaptive_expected_change_ratio = float(
                np.mean(projector_changed)
            )
            if self._adaptive_expected_change_ratio == 0.0:
                self._captures[step.key] = previous_capture.copy()
                self._gray_capture_log.append(
                    {
                        "key": step.key,
                        "mode": "reused",
                        "changed_ratio": 0.0,
                        "elapsed_milliseconds": 0,
                    }
                )
                self._step_index += 1
                self.progress_changed.emit(
                    self._capture_progress_base + self._step_index,
                    self._progress_total,
                )
                self._schedule(0, self._advance_capture)
                return
        self.status_changed.emit(step.description)
        self._projector.show_pattern(step.pattern)
        self._adaptive_step_shown_at = monotonic()
        settle_milliseconds = self._settle_milliseconds
        if (
            self._adaptive_cropped_mode
            and self._adaptive_phase == "discovery"
            and step.key == "coded"
        ):
            settle_milliseconds = 250
        elif (
            self._adaptive_cropped_mode
            and self._adaptive_phase == "gray-code"
        ):
            settle_milliseconds = 0
        self._schedule(
            settle_milliseconds,
            self._begin_frame_collection,
        )

    def _begin_frame_collection(self) -> None:
        if not self._active:
            return
        self._frame_accumulator = None
        self._frame_count = 0
        self._adaptive_candidate_gray = None
        self._collecting = True
        self._capture_token += 1
        token = self._capture_token
        run_token = self._run_token
        QTimer.singleShot(
            self._capture_timeout_milliseconds,
            lambda: self._capture_timed_out(token, run_token),
        )
        if (
            self._adaptive_cropped_mode
            and self._adaptive_phase == "gray-code"
        ):
            QTimer.singleShot(
                self._ADAPTIVE_FORCED_CAPTURE_MILLISECONDS,
                lambda: self._enable_adaptive_gray_fallback(
                    token,
                    run_token,
                ),
            )

    def _enable_adaptive_gray_fallback(
        self,
        token: int,
        run_token: int,
    ) -> None:
        if (
            self._active
            and self._collecting
            and token == self._capture_token
            and run_token == self._run_token
        ):
            self._adaptive_gray_fallback = True
            self._frame_accumulator = None
            self._frame_count = 0

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
            if self._quick_verification:
                self._prepare_quick_verification()
            elif (
                self._adaptive_cropped_mode
                and self._adaptive_phase == "discovery"
            ):
                self._prepare_adaptive_gray_code()
                return
            elif self._adaptive_cropped_mode:
                self._calculate_adaptive_mapping()
            elif self._coded_pattern_mode:
                self._calculate_coded_mapping()
            else:
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

    def _prepare_adaptive_gray_code(self) -> None:
        detection = self._detection
        rough_mapping = self._adaptive_rough_mapping
        if detection is None or rough_mapping is None:
            raise RuntimeError(
                "Validated coded discovery mapping is not available."
            )
        projector_mask = expanded_projector_board_mask(
            self._projector_size,
            rough_mapping.projector_grid,
        )
        homography, _mask = cv2.findHomography(
            detection.grid_points.reshape(-1, 2).astype(np.float64),
            rough_mapping.projector_grid.reshape(-1, 2).astype(np.float64),
        )
        if homography is None:
            raise CalibrationError(
                "The rough discovery mapping has no valid homography."
            )
        self._adaptive_rough_homography = homography
        self._detection = detection
        self._mapping = None
        self._steps = self._build_cropped_gray_code_steps(projector_mask)
        self._adaptive_projector_mask = projector_mask
        self._adaptive_previous_gray = self._captures["black"].copy()
        self._adaptive_candidate_gray = None
        self._step_index = 0
        self._capture_progress_base = 2
        self._adaptive_phase = "gray-code"
        self.status_changed.emit(
            "Coded discovery complete; starting board-cropped Gray code."
        )
        self._display_current_step()

    def _calculate_adaptive_mapping(self) -> None:
        detection = self._detection
        if detection is None:
            raise RuntimeError("Adaptive board detection is not available.")
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
        usable_mask = board_mask(
            self._captures["black"].shape,
            detection,
        )
        projector_x, projector_y, valid = decode_gray_code(
            horizontal_pairs,
            vertical_pairs,
            usable_mask,
            horizontal_omitted_bits=self._OMITTED_LEAST_SIGNIFICANT_BITS,
            vertical_omitted_bits=self._OMITTED_LEAST_SIGNIFICANT_BITS,
        )
        board_pixels = int(np.count_nonzero(usable_mask))
        coverage_ratio = (
            0.0
            if board_pixels == 0
            else float(np.count_nonzero(valid) / board_pixels)
        )
        selection = select_board_plane_support(
            detection,
            projector_x,
            projector_y,
            valid,
            self._projector_size,
            empty_squares=self._adaptive_empty_squares,
        )
        self._plane_support = selection
        self._plane_support_image = plane_support_overlay(
            self._captures["black"],
            detection.grid_points,
            selection,
        )
        if not selection.sufficiently_spread:
            self._clearance_squares = clearance_squares(selection)
        require_board_plane_support(selection)
        mapping = fit_projector_mapping(
            detection,
            projector_x,
            projector_y,
            selection.mask,
            self._projector_size,
            coverage_ratio,
        )
        self._mapping = mapping
        self._projector.set_projector_grid(mapping.projector_grid)

    def _calculate_coded_mapping(self) -> None:
        pattern = self._coded_pattern
        if pattern is None or self._inverse_coded_pattern is None:
            raise RuntimeError("Coded projector pattern is not available.")
        black = self._captures["black"]
        rough_detection = self._detector.detect(black)
        if rough_detection is None:
            raise CalibrationError(
                "The populated board was not located with the projector black."
            )
        detection, mapping = fit_coded_pattern_mapping(
            black,
            self._captures["coded"],
            rough_detection,
            pattern,
            self._projector_size,
            inverse_illuminated=self._captures["coded_inverse"],
            inverse_pattern=self._inverse_coded_pattern,
        )
        self._detection = detection
        self._mapping = mapping
        self._projector.set_projector_grid(mapping.projector_grid)

    def _prepare_quick_verification(self) -> None:
        if self._mapping is None:
            raise RuntimeError("Saved calibration mapping is not available.")
        current_detection = self._detector.detect(self._captures["black"])
        if current_detection is None:
            raise CalibrationError(
                "The empty board was not detected for quick verification."
            )
        self._mapping = remap_projector_mapping(
            self._mapping,
            current_detection.grid_points,
            self._projector_size,
        )
        self._detection = current_detection
        self._projector.set_projector_grid(self._mapping.projector_grid)

    def _start_validation(self) -> None:
        if self._detection is None or self._mapping is None:
            raise RuntimeError("Calibration mapping is not available.")
        self.status_changed.emit("Mapping complete; refreshing black reference.")
        if self._quick_verification:
            if self._quick_phase == "fit":
                self._validation_candidates = (
                    _STARTING_POSITION_CORNER_TARGETS
                    + [_STARTING_POSITION_CENTER_TARGET]
                )
                self._validation_progress_base = 1
            else:
                self._validation_candidates = _STARTING_POSITION_EDGE_TARGETS
                self._validation_progress_base = 7
        else:
            self._validation_candidates = (
                _STARTING_POSITION_CORNER_TARGETS
                + _STARTING_POSITION_EDGE_TARGETS
                + [_STARTING_POSITION_CENTER_TARGET]
            )
            if self._adaptive_cropped_mode:
                self._validation_candidates = (
                    self._adaptive_validation_candidates()
                )
            self._validation_progress_base = (
                self._progress_total
                - len(self._validation_candidates)
                - 1
            )
        self._validation_candidates = [
            candidates.copy()
            for candidates in self._validation_candidates
        ]
        self._validation_coordinates = [
            candidates[0] for candidates in self._validation_candidates
        ]
        self._validation_candidate_index = 0
        self._validation_index = 0
        self._show_validation_black()

    def _adaptive_validation_candidates(
        self,
    ) -> list[list[tuple[float, float]]]:
        selection = self._plane_support
        if selection is None:
            raise RuntimeError("Plane support is not available.")
        squares = select_validation_squares(selection.square_status)
        if len(squares) < self._MINIMUM_VALIDATION_HITS:
            raise CalibrationError(
                f"Only {len(squares)} usable squares are available for "
                f"validation targets; {self._MINIMUM_VALIDATION_HITS} are "
                "required."
            )
        self._progress_total += len(squares) - 9
        return [
            [(column + 0.5, row + 0.5)] for row, column in squares
        ]

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
        expected = interpolate_grid_point(
            self._mapping.camera_grid,
            board_x,
            board_y,
        )
        square_width = self._local_square_width(board_x, board_y)
        self._validation_expected.append(expected)
        self._validation_widths.append(square_width)

        target_polygon = board_circle_polygon(
            self._mapping.projector_grid,
            board_x,
            board_y,
            0.14,
        )
        width, height = self._projector_size
        pattern = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(
            pattern,
            [np.rint(target_polygon).astype(np.int32)],
            self._intensity,
            lineType=cv2.LINE_AA,
        )
        self.status_changed.emit(
            f"Validating target {self._validation_index + 1}/"
            f"{len(self._validation_coordinates)}"
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
            self.progress_changed.emit(
                self._validation_progress_base + 1,
                self._progress_total,
            )
            self._show_validation_target_safely()
            return

        expected = self._validation_expected[-1]
        square_width = self._validation_widths[-1]
        try:
            measured = locate_validation_target(
                self._captures["validation_black"],
                averaged,
                expected,
                max(8, round(square_width * 0.75)),
            )
        except CalibrationError:
            if self._adaptive_cropped_mode:
                # A piece can hide a surplus target; it counts as a miss.
                measured = np.full(2, np.nan, dtype=np.float32)
            else:
                candidates = self._validation_candidates[
                    self._validation_index
                ]
                current_coordinate = self._validation_coordinates[
                    self._validation_index
                ]
                current_candidate = candidates.index(current_coordinate)
                next_candidate = current_candidate + 1
                if next_candidate >= len(candidates):
                    raise
                self._validation_expected.pop()
                self._validation_widths.pop()
                self._validation_candidate_index = next_candidate
                self._validation_coordinates[self._validation_index] = (
                    candidates[next_candidate]
                )
                self.status_changed.emit(
                    f"Validation target {self._validation_index + 1} was "
                    "not detected; trying an alternative location."
                )
                self._schedule(0, self._show_validation_target_safely)
                return
        self._validation_measured.append(measured)
        self._validation_index += 1
        self._validation_candidate_index = 0
        completed_steps = (
            self._validation_progress_base
            + self._validation_index
            + 1
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
        if self._quick_verification and self._quick_phase == "fit":
            board_coordinates = np.asarray(
                self._validation_coordinates,
                dtype=np.float32,
            )
            self._mapping = quick_realign_mapping(
                self._mapping,
                board_coordinates,
                measured,
                self._projector_size,
            )
            self._projector.set_projector_grid(
                self._mapping.projector_grid
            )
            self._quick_phase = "validate"
            self._validation_expected.clear()
            self._validation_measured.clear()
            self._validation_widths.clear()
            self._validation_index = -1
            self._captures.pop("validation_black", None)
            self.status_changed.emit(
                "Candidate quick correction fitted; measuring independent "
                "validation targets."
            )
            self._start_validation()
            return

        validation = (
            self._adaptive_validation_result(measured, expected, widths)
            if self._adaptive_cropped_mode
            else validation_result(measured, expected, widths)
        )
        if (
            not self._adaptive_cropped_mode
            and validation.maximum_square_fraction >= 0.1
        ):
            if (
                not self._quick_verification
                and not self._validation_only
                and self._refinement_iteration
                < self._maximum_refinement_iterations
            ):
                self._refine_and_repeat_validation(measured)
                return
            if self._retry_worst_validation_target(validation, widths):
                return
            fractions = validation.errors / widths
            worst_index = int(np.argmax(fractions))
            worst_coordinate = self._validation_coordinates[worst_index]
            raise CalibrationError(
                f"Validation target {worst_index + 1} at "
                f"({worst_coordinate[0]:.2f}, "
                f"{worst_coordinate[1]:.2f}) has "
                f"{fractions[worst_index]:.1%} square error; the maximum "
                "allowed is 10%."
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
        operation = (
            "Quick realignment"
            if self._quick_verification
            else "Validation"
            if self._validation_only
            else "Calibration"
        )
        self.status_changed.emit(
            f"{operation} succeeded: {validation.maximum_error:.2f} px "
            "maximum validation error."
        )
        self.completed.emit(result)

    def _adaptive_validation_result(
        self,
        measured: FloatArray,
        expected: FloatArray,
        widths: FloatArray,
    ) -> ValidationResult:
        """Accept surplus targets when enough spread targets hit."""
        detected = np.all(np.isfinite(measured), axis=1)
        fractions = np.full(len(measured), np.nan, dtype=np.float32)
        fractions[detected] = (
            np.linalg.norm(measured[detected] - expected[detected], axis=1)
            / widths[detected]
        )
        hits = detected & (np.nan_to_num(fractions, nan=1.0) < 0.1)
        self._validation_log = [
            {
                "square": [int(y - 0.5), int(x - 0.5)],
                "detected": bool(detected[index]),
                "square_error": (
                    None
                    if not detected[index]
                    else round(float(fractions[index]), 4)
                ),
                "hit": bool(hits[index]),
            }
            for index, (x, y) in enumerate(self._validation_coordinates)
        ]
        quadrants = {
            (y < 4, x < 4)
            for (x, y), hit in zip(
                self._validation_coordinates, hits, strict=True
            )
            if hit
        }
        hit_count = int(np.count_nonzero(hits))
        if hit_count < self._MINIMUM_VALIDATION_HITS:
            raise CalibrationError(
                f"Only {hit_count} of {len(measured)} validation targets "
                "were within 10% of a square; "
                f"{self._MINIMUM_VALIDATION_HITS} are required."
            )
        if len(quadrants) < 4:
            raise CalibrationError(
                "Validation targets within 10% of a square do not cover "
                "every board quadrant."
            )
        return validation_result(measured[hits], expected[hits], widths[hits])

    def _retry_worst_validation_target(
        self,
        validation: ValidationResult,
        widths: FloatArray,
    ) -> bool:
        if len(self._validation_candidates) != len(
            self._validation_coordinates
        ):
            return False
        fractions = validation.errors / widths
        worst_index = int(np.argmax(fractions))
        candidates = self._validation_candidates[worst_index]
        current_coordinate = self._validation_coordinates[worst_index]
        current_candidate = candidates.index(current_coordinate)
        next_candidate = current_candidate + 1
        if next_candidate >= len(candidates):
            return False

        self._validation_coordinates[worst_index] = candidates[next_candidate]
        self._captures.pop("validation_black", None)
        self._validation_expected.clear()
        self._validation_measured.clear()
        self._validation_widths.clear()
        self._validation_candidate_index = 0
        self._validation_index = 0
        self._progress_total += len(self._validation_coordinates) + 1
        self._validation_progress_base = (
            self._progress_total
            - len(self._validation_coordinates)
            - 1
        )
        self.progress_changed.emit(
            self._validation_progress_base,
            self._progress_total,
        )
        self.status_changed.emit(
            f"Validation target {worst_index + 1} had excessive error; "
            "retrying at a nearby location."
        )
        self._schedule(0, self._show_validation_black)
        return True

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
        self._validation_progress_base = (
            self._progress_total
            - len(self._validation_coordinates)
            - 1
        )
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
        self._validation_coordinates = [
            candidates[0] for candidates in self._validation_candidates
        ]
        self._validation_candidate_index = 0
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
