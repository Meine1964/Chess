from collections.abc import Callable
from dataclasses import dataclass, field, replace
from time import monotonic
from typing import Literal

import cv2
import numpy as np
from PySide6.QtCore import QObject, QRunnable, QThreadPool, QTimer, Signal, Slot

from .board_detector import (
    BoardDetection,
    BoardDetector,
    FloatArray,
    ImageArray,
    StableBoardTracker,
)
from .calibration_controller import CalibrationResult
from .projector_calibration import (
    CalibrationError,
    GrayImage,
    MappingResult,
    bgr_to_gray,
    interpolate_grid_point,
    locate_validation_target,
    remap_projector_mapping,
)

OutputMode = Literal["black", "selected", "recovery", "paused"]
RecoveryHypothesisName = Literal["camera_moved", "board_moved"]

# Lit squares of the startup verification pattern, indexed by physical grid
# row and column. Under every other board symmetry and every one-square shift
# it correlates with itself by at most 0.1, unlike a plain checkerboard.
RECOVERY_PATTERN = np.asarray(
    [
        [character == "#" for character in row]
        for row in (
            "..#.#...",
            "#.######",
            "...###..",
            "####.#..",
            "#.###..#",
            "#..#..#.",
            "..#.#.#.",
            "...###.#",
        )
    ],
    dtype=bool,
)
_RECOVERY_MAXIMUM_ATTEMPTS = 3
_RECOVERY_SETTLE_SECONDS = 0.5
_RECOVERY_DISPLAY_TIMEOUT_SECONDS = 3.0
_RECOVERY_RESPONSE_OFFSET = 20.0
_RECOVERY_MINIMUM_SCORE = 0.4
_RECOVERY_MINIMUM_MARGIN = 0.2
_RECOVERY_RIVAL_SEPARATION = 0.25
_RECOVERY_MINIMUM_SCORED_SQUARES = 16
_RECOVERY_SAMPLE_OFFSETS = np.linspace(0.3, 0.7, 5, dtype=np.float32)
_ALL_SQUARES = tuple((row, column) for row in range(8) for column in range(8))


def _point_diagnostic(point: FloatArray) -> list[float]:
    return [round(float(value), 2) for value in point]


@dataclass(frozen=True)
class RecoveryHypothesis:
    name: RecoveryHypothesisName
    symmetry: int
    camera_grid: FloatArray
    square_width: float
    score: float
    squares_scored: int

    def diagnostic(self) -> dict[str, object]:
        return {
            "hypothesis": self.name,
            "symmetry": self.symmetry,
            "score": round(self.score, 3),
            "squares_scored": self.squares_scored,
            "grid_center_camera_point": _point_diagnostic(
                np.mean(self.camera_grid.reshape(-1, 2), axis=0)
            ),
            "square_width_pixels": round(self.square_width, 2),
        }


@dataclass
class _RecoveryState:
    detection: BoardDetection
    old_camera_grid: FloatArray
    black_reference: GrayImage
    displacement: float
    reason: str
    requested_at: float
    inverted: bool = False
    shown_at: float | None = None
    display_delays: list[float] = field(default_factory=list)
    attempts: int = 0
    pattern_frame: GrayImage | None = None
    inverse_frame: GrayImage | None = None
    last_response: FloatArray | None = None
    last_hypotheses: tuple[RecoveryHypothesis, ...] = ()
    last_error: str | None = None


def grid_symmetries(grid: FloatArray) -> tuple[FloatArray, ...]:
    """Return the eight index orderings of one physical camera grid."""
    flipped = np.flip(grid, axis=1)
    return tuple(
        np.ascontiguousarray(np.rot90(source, turns, axes=(0, 1)))
        for source in (grid, flipped)
        for turns in range(4)
    )


def _median_square_width(grid: FloatArray) -> float:
    horizontal = np.linalg.norm(np.diff(grid, axis=1), axis=2)
    vertical = np.linalg.norm(np.diff(grid, axis=0), axis=2)
    return float(
        np.median(np.concatenate((horizontal.reshape(-1), vertical.reshape(-1))))
    )


def recovery_pattern_image(
    projector_grid: FloatArray,
    projector_size: tuple[int, int],
    *,
    inverted: bool,
) -> GrayImage:
    """Render the verification pattern, or its inverse, on the board grid."""
    width, height = projector_size
    image = np.zeros((height, width), dtype=np.uint8)
    lit = RECOVERY_PATTERN != inverted
    for row, column in zip(*np.nonzero(lit), strict=True):
        quad = np.asarray(
            (
                projector_grid[row, column],
                projector_grid[row, column + 1],
                projector_grid[row + 1, column + 1],
                projector_grid[row + 1, column],
            ),
            dtype=np.float64,
        )
        cv2.fillConvexPoly(
            image,
            np.rint(quad * 16).astype(np.int32),
            255,
            lineType=cv2.LINE_AA,
            shift=4,
        )
    return image


def normalized_recovery_response(
    black: GrayImage,
    pattern: GrayImage,
    inverse: GrayImage,
) -> FloatArray:
    """Return the pattern response independent of surface reflectance.

    Values approach +1 where the pattern lit the surface, -1 where its
    inverse did, and 0 where no projected light falls.
    """
    if not black.shape == pattern.shape == inverse.shape or black.ndim != 2:
        raise ValueError("Recovery images must be equally sized grayscale images.")
    black_blurred, pattern_blurred, inverse_blurred = (
        cv2.GaussianBlur(image.astype(np.float32), (0, 0), 1.0)
        for image in (black, pattern, inverse)
    )
    projected = np.maximum(
        pattern_blurred + inverse_blurred - 2 * black_blurred,
        0.0,
    )
    return np.asarray(
        (pattern_blurred - inverse_blurred)
        / (projected + _RECOVERY_RESPONSE_OFFSET),
        dtype=np.float32,
    )


def _score_grid(
    response: FloatArray,
    grid: FloatArray,
    squares: tuple[tuple[int, int], ...],
) -> float:
    indices = np.asarray(squares, dtype=np.intp)
    rows, columns = indices[:, 0], indices[:, 1]
    top_left = grid[rows, columns][:, None, :]
    top_right = grid[rows, columns + 1][:, None, :]
    bottom_left = grid[rows + 1, columns][:, None, :]
    bottom_right = grid[rows + 1, columns + 1][:, None, :]
    u_values, v_values = np.meshgrid(
        _RECOVERY_SAMPLE_OFFSETS,
        _RECOVERY_SAMPLE_OFFSETS,
    )
    u = u_values.reshape(1, -1, 1)
    v = v_values.reshape(1, -1, 1)
    points = (
        (1 - u) * (1 - v) * top_left
        + u * (1 - v) * top_right
        + (1 - u) * v * bottom_left
        + u * v * bottom_right
    )
    x = np.rint(points[..., 0]).astype(np.intp)
    y = np.rint(points[..., 1]).astype(np.intp)
    height, width = response.shape
    inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    samples = np.where(
        inside,
        response[np.clip(y, 0, height - 1), np.clip(x, 0, width - 1)],
        0.0,
    )
    signs = np.where(RECOVERY_PATTERN[rows, columns], 1.0, -1.0)
    return float(np.mean(signs * np.median(samples, axis=1)))


def score_recovery_hypotheses(
    response: FloatArray,
    live_camera_grid: FloatArray,
    saved_camera_grid: FloatArray | None,
    squares: tuple[tuple[int, int], ...] = _ALL_SQUARES,
) -> tuple[RecoveryHypothesis, ...]:
    """Score how well each explanation of the grid change predicts the pattern.

    The camera-moved hypothesis predicts the pattern on the live grid under
    any of its eight index orderings; the board-moved hypothesis predicts it
    on the saved grid.
    """
    if not squares:
        raise ValueError("At least one board square must be scored.")
    candidates: list[tuple[RecoveryHypothesisName, int, FloatArray]] = [
        ("camera_moved", symmetry, grid)
        for symmetry, grid in enumerate(grid_symmetries(live_camera_grid))
    ]
    if saved_camera_grid is not None:
        candidates.append(("board_moved", 0, saved_camera_grid))
    return tuple(
        RecoveryHypothesis(
            name=name,
            symmetry=symmetry,
            camera_grid=grid,
            square_width=_median_square_width(grid),
            score=_score_grid(response, grid, squares),
            squares_scored=len(squares),
        )
        for name, symmetry, grid in candidates
    )


def select_recovery_hypothesis(
    hypotheses: tuple[RecoveryHypothesis, ...],
) -> RecoveryHypothesis | None:
    """Return the clearly best hypothesis, or None when none or several fit.

    Rivals predicting nearly the same grid do not make the result ambiguous.
    """
    if not hypotheses:
        return None
    best = max(hypotheses, key=lambda hypothesis: hypothesis.score)
    if best.score < _RECOVERY_MINIMUM_SCORE:
        return None
    for rival in hypotheses:
        if rival is best:
            continue
        separation = float(
            np.max(np.linalg.norm(rival.camera_grid - best.camera_grid, axis=2))
        ) / best.square_width
        if (
            separation > _RECOVERY_RIVAL_SEPARATION
            and rival.score > best.score - _RECOVERY_MINIMUM_MARGIN
        ):
            return None
    return best


def camera_moved_mapping(
    mapping: MappingResult,
    live_camera_grid: FloatArray,
) -> MappingResult:
    """Keep the projector grid and refit the camera side to a new camera grid."""
    homography, _mask = cv2.findHomography(
        live_camera_grid.reshape(-1, 2),
        mapping.projector_grid.reshape(-1, 2),
        method=0,
    )
    if homography is None:
        raise CalibrationError(
            "The camera-to-projector homography could not be refitted."
        )
    return replace(
        mapping,
        camera_grid=np.asarray(live_camera_grid, dtype=np.float32).copy(),
        camera_to_projector_homography=np.asarray(
            homography,
            dtype=mapping.camera_to_projector_homography.dtype,
        ),
    )


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
    recovery_projection_requested = Signal(object, str, object)
    recovery_completed = Signal(object, str, object, object)
    recovery_failed = Signal(str, object, object)

    def __init__(
        self,
        detector: BoardDetector,
        *,
        interval_milliseconds: int = 200,
        board_remap_threshold: float = 1.5,
        projection_error_fraction: float = 0.1,
        stable_checks_after_remap: int = 5,
        clock: Callable[[], float] = monotonic,
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
        self._startup_verification_pending = False
        self._camera_changed = False
        self._recovery_occupied_squares: frozenset[tuple[int, int]] = frozenset()
        self._recovery: _RecoveryState | None = None
        self._clock = clock
        self._check_started_at = 0.0

    def configure(
        self,
        result: CalibrationResult,
        projector_size: tuple[int, int],
        *,
        startup_verification: bool = False,
        camera_changed: bool = False,
        recovery_occupied_squares: frozenset[tuple[int, int]] = frozenset(),
    ) -> None:
        if projector_size[0] <= 1 or projector_size[1] <= 1:
            raise ValueError("Projector dimensions must be greater than one.")
        if not all(
            0 <= value < 8
            for square in recovery_occupied_squares
            for value in square
        ):
            raise ValueError("Occupied squares must be inside the 8x8 board.")
        self._result = result
        self._tracker.reset(None if camera_changed else result.board_detection)
        self._projector_size = projector_size
        self._startup_verification_pending = (
            startup_verification or camera_changed
        )
        self._camera_changed = camera_changed
        self._recovery_occupied_squares = recovery_occupied_squares
        self._recovery = None
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
        self._tracker.reset()
        self._startup_verification_pending = False
        self._camera_changed = False
        self._recovery = None
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

    @property
    def startup_verification_pending(self) -> bool:
        return self._startup_verification_pending

    def recovery_projection_shown(self) -> None:
        """Record that the requested verification pattern is now projected."""
        if self._recovery is not None and self._recovery.shown_at is None:
            self._recovery.shown_at = self._clock()
            self._recovery.display_delays.append(
                self._recovery.shown_at - self._recovery.requested_at
            )

    def capture_black_reference(self, frame: ImageArray) -> None:
        if self._result is not None:
            self._black_reference = bgr_to_gray(frame)

    def set_black(self) -> None:
        if self._result is None:
            return
        self._generation += 1
        self._check_running = False
        self._recovery = None
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
        self._recovery = None
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
        self._recovery = None
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
        self._check_started_at = self._clock()
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
        if self._recovery is not None:
            self._check_recovery(frame)
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
        if self._startup_verification_pending:
            if (
                displacement <= self._board_remap_threshold
                and not self._camera_changed
            ):
                self._startup_verification_pending = False
            else:
                if self._output_mode == "black":
                    self._begin_recovery(detection, frame, displacement)
                return
        if displacement > self._board_remap_threshold:
            old_camera_grid = self._result.mapping.camera_grid
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

    def _begin_recovery(
        self,
        detection: BoardDetection,
        frame: ImageArray,
        displacement: float,
    ) -> None:
        if self._result is None:
            return
        reason = (
            "camera-hardware-changed"
            if self._camera_changed
            else "camera-grid-displaced"
        )
        self._recovery = _RecoveryState(
            detection=detection,
            old_camera_grid=self._result.mapping.camera_grid.copy(),
            black_reference=bgr_to_gray(frame),
            displacement=displacement,
            reason=reason,
            requested_at=self._clock(),
        )
        self._generation += 1
        self._check_running = False
        self._output_mode = "recovery"
        self._request_recovery_pattern(inverted=False)

    def _recovery_scored_squares(self) -> tuple[tuple[int, int], ...]:
        empty = tuple(
            square
            for square in _ALL_SQUARES
            if square not in self._recovery_occupied_squares
        )
        if len(empty) < _RECOVERY_MINIMUM_SCORED_SQUARES:
            return _ALL_SQUARES
        return empty

    def _request_recovery_pattern(self, *, inverted: bool) -> None:
        state = self._recovery
        if state is None or self._result is None:
            return
        state.inverted = inverted
        state.shown_at = None
        state.requested_at = self._clock()
        pattern = recovery_pattern_image(
            self._result.mapping.projector_grid,
            self._projector_size,
            inverted=inverted,
        )
        message = (
            "The saved calibration belongs to a different camera; projecting "
            "a verification pattern to recover the camera grid."
            if self._camera_changed
            else (
                f"The live camera grid differs by {state.displacement:.1f} px "
                "from the saved calibration; projecting a verification "
                "pattern to determine whether the camera or the board moved."
            )
        )
        if not inverted and state.attempts == 0:
            self.status_changed.emit(message)
        self.recovery_projection_requested.emit(
            pattern,
            message,
            {
                "reason": state.reason,
                "displacement_pixels": round(state.displacement, 2),
                "pattern": "inverse" if inverted else "pattern",
                "attempt": state.attempts + 1,
                "scored_squares": len(self._recovery_scored_squares()),
                "board_moved_tested": not self._camera_changed,
            },
        )

    def _check_recovery(self, frame: ImageArray) -> None:
        state = self._recovery
        if state is None or self._result is None:
            return
        captured_at = self._check_started_at
        if state.shown_at is None:
            if (
                captured_at - state.requested_at
                >= _RECOVERY_DISPLAY_TIMEOUT_SECONDS
            ):
                self._fail_recovery(
                    "The verification pattern was not displayed by the "
                    "projector."
                )
            return
        if captured_at < state.shown_at + _RECOVERY_SETTLE_SECONDS:
            return
        captured = bgr_to_gray(frame)
        if captured.shape != state.black_reference.shape:
            self._fail_recovery(
                "The camera image size changed during alignment recovery."
            )
            return
        if not state.inverted:
            state.pattern_frame = captured
            self._request_recovery_pattern(inverted=True)
            return
        state.inverse_frame = captured
        state.attempts += 1
        if state.pattern_frame is None:
            self._request_recovery_pattern(inverted=False)
            return
        response = normalized_recovery_response(
            state.black_reference,
            state.pattern_frame,
            state.inverse_frame,
        )
        hypotheses = score_recovery_hypotheses(
            response,
            state.detection.grid_points,
            None if self._camera_changed else state.old_camera_grid,
            self._recovery_scored_squares(),
        )
        state.last_response = response
        state.last_hypotheses = hypotheses
        selected = select_recovery_hypothesis(hypotheses)
        if selected is None:
            best_score = max(hypothesis.score for hypothesis in hypotheses)
            state.last_error = (
                "The verification pattern matched neither a moved camera nor "
                "a moved board unambiguously (best score "
                f"{best_score:.2f})."
            )
            if state.attempts >= _RECOVERY_MAXIMUM_ATTEMPTS:
                self._fail_recovery(state.last_error)
            else:
                self._request_recovery_pattern(inverted=False)
            return

        try:
            if selected.name == "camera_moved":
                mapping = camera_moved_mapping(
                    self._result.mapping,
                    selected.camera_grid,
                )
                detection = replace(
                    state.detection,
                    grid_points=selected.camera_grid.copy(),
                )
            else:
                mapping = remap_projector_mapping(
                    self._result.mapping,
                    state.detection.grid_points,
                    self._projector_size,
                )
                detection = state.detection
        except (CalibrationError, ValueError) as error:
            self._fail_recovery(f"The recovered mapping is invalid: {error}")
            return

        result = CalibrationResult(
            board_detection=detection,
            mapping=mapping,
            validation=self._result.validation,
        )
        details = self._recovery_details(state, selected)
        images = self._recovery_images(state)
        self._result = result
        self._tracker.reset(detection)
        self._recovery = None
        self._startup_verification_pending = False
        self._camera_changed = False
        self._output_mode = "black"
        self._black_reference = None
        self._geometry_failure_count = 0
        self._geometry_failure_source = None
        self._stable_board_checks = self._stable_checks_after_remap
        self._remapped_since_verification = False
        self._generation += 1
        self._check_running = False
        description = (
            "the camera moved or was replaced"
            if selected.name == "camera_moved"
            else "the board moved"
        )
        message = (
            f"Alignment recovered: {description}; verification pattern score "
            f"{selected.score:.2f}."
        )
        self.status_changed.emit(message)
        self.recovery_completed.emit(result, message, details, images)

    def _fail_recovery(self, message: str) -> None:
        state = self._recovery
        details: dict[str, object] = {}
        images: dict[str, ImageArray] = {}
        if state is not None:
            details = self._recovery_details(state, None)
            images = self._recovery_images(state)
        self._timer.stop()
        self._result = None
        self._recovery = None
        self._startup_verification_pending = False
        self._camera_changed = False
        self._output_mode = "paused"
        self._generation += 1
        self._check_running = False
        self.status_changed.emit(f"Alignment recovery failed: {message}")
        self.recovery_failed.emit(message, details, images)

    def _recovery_details(
        self,
        state: _RecoveryState,
        selected: RecoveryHypothesis | None,
    ) -> dict[str, object]:
        return {
            "reason": state.reason,
            "displacement_pixels": round(state.displacement, 2),
            "attempts": state.attempts,
            "scored_squares": len(self._recovery_scored_squares()),
            "minimum_score": _RECOVERY_MINIMUM_SCORE,
            "minimum_margin": _RECOVERY_MINIMUM_MARGIN,
            "display_delays_seconds": [
                round(delay, 3) for delay in state.display_delays
            ],
            "last_error": state.last_error,
            "hypotheses": [
                hypothesis.diagnostic() for hypothesis in state.last_hypotheses
            ],
            "selected": None if selected is None else selected.diagnostic(),
        }

    @staticmethod
    def _recovery_images(state: _RecoveryState) -> dict[str, ImageArray]:
        images: dict[str, ImageArray] = {
            "recovery-black": state.black_reference,
        }
        if state.pattern_frame is not None:
            images["recovery-pattern"] = state.pattern_frame
        if state.inverse_frame is not None:
            images["recovery-inverse"] = state.inverse_frame
        if state.last_response is not None:
            images["recovery-response"] = np.clip(
                state.last_response * 127 + 128,
                0,
                255,
            ).astype(np.uint8)
        return images

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
