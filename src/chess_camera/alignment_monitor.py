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
BoolArray = np.ndarray[tuple[int, ...], np.dtype[np.bool_]]

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
_RECOVERY_ACCEPTED_RESIDUAL = 0.05
_RECOVERY_RESPONSE_SATURATION = 2.5
_RECOVERY_VERIFIED_RESIDUAL = 0.1
_RECOVERY_FRINGE_STEPS = 4
_RECOVERY_FRINGE_OFFSETS = np.linspace(0.2, 0.8, 7, dtype=np.float32)
_RECOVERY_MINIMUM_MODULATION = 4.0
_RECOVERY_MINIMUM_MODULATION_FRACTION = 0.15
_RECOVERY_MINIMUM_VALID_SAMPLE_FRACTION = 0.5
_RECOVERY_MINIMUM_REFINED_SQUARES = 8
_RECOVERY_MINIMUM_REFINED_SPAN = 3
_RECOVERY_MAXIMUM_FIT_ERROR = 0.08
_RECOVERY_LOCAL_RADIUS = 1.5
_RECOVERY_MINIMUM_LOCAL_SAMPLES = 10
_ALL_SQUARES = tuple((row, column) for row in range(8) for column in range(8))
# Offset measurement uses 7 x 7 cells shifted by half a square, so that every
# pattern edge crosses the middle of a uniformly coloured board square.
# Edges coinciding with board edges would be biased toward the darker square
# by optical blur.
_OFFSET_PATTERN = RECOVERY_PATTERN[:7, :7]
_MINIMUM_OFFSET_CELLS = 8


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
    stage: str = "coarse"
    queue: list[tuple[str, GrayImage]] = field(default_factory=list)
    current: str | None = None
    announce_pending: bool = False
    shown_at: float | None = None
    display_delays: list[float] = field(default_factory=list)
    attempts: int = 0
    frames: dict[str, GrayImage] = field(default_factory=dict)
    last_response: FloatArray | None = None
    last_hypotheses: tuple[RecoveryHypothesis, ...] = ()
    last_error: str | None = None
    selected: RecoveryHypothesis | None = None
    board_camera_grid: FloatArray | None = None
    candidate: CalibrationResult | None = None
    measure_response: FloatArray | None = None
    measured_residual: float | None = None
    refinement: dict[str, object] | None = None
    refinement_error: str | None = None
    verify_response: FloatArray | None = None
    verification: dict[str, object] | None = None


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


_RENDER_SUPERSAMPLING = 4
_UNIT_SQUARE = np.asarray(((0, 0), (1, 0), (1, 1), (0, 1)), dtype=np.float32)


def _render_squares(
    shape: tuple[int, int],
    grid: FloatArray,
    values: FloatArray,
) -> FloatArray:
    """Render per-square values with exact, unbiased pixel coverage.

    Each cell is mapped through the homography of its four grid corners and
    supersampled, so neighbouring squares partition every pixel exactly.
    """
    height, width = shape
    image = np.zeros(shape, dtype=np.float32)
    steps = (np.arange(_RENDER_SUPERSAMPLING) + 0.5) / _RENDER_SUPERSAMPLING - 0.5
    rows, columns = values.shape
    for row in range(rows):
        for column in range(columns):
            value = float(values[row, column])
            if value == 0:
                continue
            quad = np.asarray(
                (
                    grid[row, column],
                    grid[row, column + 1],
                    grid[row + 1, column + 1],
                    grid[row + 1, column],
                ),
                dtype=np.float32,
            )
            left = max(0, int(np.floor(np.min(quad[:, 0]))) - 1)
            top = max(0, int(np.floor(np.min(quad[:, 1]))) - 1)
            right = min(width, int(np.ceil(np.max(quad[:, 0]))) + 2)
            bottom = min(height, int(np.ceil(np.max(quad[:, 1]))) + 2)
            if right <= left or bottom <= top:
                continue
            x_values = (
                np.arange(left, right, dtype=np.float32)[:, None] + steps
            ).reshape(-1)
            y_values = (
                np.arange(top, bottom, dtype=np.float32)[:, None] + steps
            ).reshape(-1)
            x_grid, y_grid = np.meshgrid(x_values, y_values)
            local = _apply_homography(
                cv2.getPerspectiveTransform(quad, _UNIT_SQUARE),
                np.stack((x_grid, y_grid), axis=2),
            )
            inside = (
                (local[..., 0] >= 0)
                & (local[..., 0] < 1)
                & (local[..., 1] >= 0)
                & (local[..., 1] < 1)
            )
            coverage = inside.reshape(
                bottom - top,
                _RENDER_SUPERSAMPLING,
                right - left,
                _RENDER_SUPERSAMPLING,
            ).mean(axis=(1, 3))
            image[top:bottom, left:right] += value * coverage
    return image


def recovery_pattern_image(
    projector_grid: FloatArray,
    projector_size: tuple[int, int],
    *,
    inverted: bool,
    offset: bool = False,
) -> GrayImage:
    """Render the verification pattern, or its inverse, on the board grid.

    The offset variant is shifted by half a square for offset measurement.
    """
    width, height = projector_size
    pattern = _OFFSET_PATTERN if offset else RECOVERY_PATTERN
    corners = _offset_corners(projector_grid) if offset else projector_grid
    lit = np.where(pattern != inverted, 255.0, 0.0)
    return np.asarray(
        np.rint(np.clip(_render_squares((height, width), corners, lit), 0, 255)),
        dtype=np.uint8,
    )


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


def _bilinear(grid: FloatArray, board_points: FloatArray) -> FloatArray:
    """Map cell coordinates (column, row) through a grid of cell corners."""
    points = np.asarray(board_points, dtype=np.float64).reshape(-1, 2)
    column = np.clip(np.floor(points[:, 0]), 0, grid.shape[1] - 2).astype(np.intp)
    row = np.clip(np.floor(points[:, 1]), 0, grid.shape[0] - 2).astype(np.intp)
    u = (points[:, 0] - column)[:, None]
    v = (points[:, 1] - row)[:, None]
    mapped = (
        (1 - u) * (1 - v) * grid[row, column]
        + u * (1 - v) * grid[row, column + 1]
        + (1 - u) * v * grid[row + 1, column]
        + u * v * grid[row + 1, column + 1]
    )
    return np.asarray(mapped, dtype=np.float32).reshape(
        np.shape(board_points)
    )


def _square_samples(
    squares: tuple[tuple[int, int], ...],
    offsets: FloatArray,
) -> FloatArray:
    """Return board coordinates sampled inside each square, shape (n, m, 2)."""
    indices = np.asarray(squares, dtype=np.float32)
    u_values, v_values = np.meshgrid(offsets, offsets)
    return np.stack(
        (
            indices[:, 1, None] + u_values.reshape(1, -1),
            indices[:, 0, None] + v_values.reshape(1, -1),
        ),
        axis=2,
    ).astype(np.float32)


def _sample_image(image: FloatArray, points: FloatArray) -> FloatArray:
    flat = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    sampled = cv2.remap(
        np.asarray(image, dtype=np.float32),
        flat[..., 0],
        flat[..., 1],
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    return np.asarray(sampled, dtype=np.float32).reshape(points.shape[:-1])


def _score_grid(
    response: FloatArray,
    grid: FloatArray,
    squares: tuple[tuple[int, int], ...],
    pattern: BoolArray = RECOVERY_PATTERN,
) -> float:
    board = _square_samples(squares, _RECOVERY_SAMPLE_OFFSETS)
    samples = _sample_image(response, _bilinear(grid, board))
    indices = np.asarray(squares, dtype=np.intp)
    signs = np.where(pattern[indices[:, 0], indices[:, 1]], 1.0, -1.0)
    return float(np.mean(signs * np.median(samples, axis=1)))


def _offset_corners(grid: FloatArray) -> FloatArray:
    """Return the 8 x 8 corners of the half-square-offset cells."""
    rows, columns = np.mgrid[0:8, 0:8]
    board = np.stack((columns + 0.5, rows + 0.5), axis=2).astype(np.float32)
    return _bilinear(grid, board)


def _offset_cells(
    squares: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int], ...]:
    """Return offset cells lying entirely on the given squares."""
    scored = set(squares)
    cells = tuple(
        (row, column)
        for row in range(7)
        for column in range(7)
        if {
            (row, column),
            (row, column + 1),
            (row + 1, column),
            (row + 1, column + 1),
        }
        <= scored
    )
    if len(cells) < _MINIMUM_OFFSET_CELLS:
        return tuple((row, column) for row in range(7) for column in range(7))
    return cells


def offset_pattern_score(
    response: FloatArray,
    grid: FloatArray,
    squares: tuple[tuple[int, int], ...] = _ALL_SQUARES,
) -> float:
    """Score the offset measurement pattern predicted on a board grid."""
    return _score_grid(
        response,
        _offset_corners(grid),
        _offset_cells(squares),
        _OFFSET_PATTERN,
    )


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


def _apply_homography(homography: FloatArray, points: FloatArray) -> FloatArray:
    flat = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(flat, np.asarray(homography, np.float64))
    return np.asarray(mapped, dtype=np.float32).reshape(np.shape(points))


def _pattern_template(
    shape: tuple[int, int],
    grid: FloatArray,
    squares: tuple[tuple[int, int], ...],
) -> tuple[FloatArray, ImageArray]:
    corners = _offset_corners(grid)
    template = _render_squares(
        shape,
        corners,
        np.where(_OFFSET_PATTERN, 1.0, -1.0),
    )
    scored = np.zeros(_OFFSET_PATTERN.shape, dtype=np.float32)
    for row, column in _offset_cells(squares):
        scored[row, column] = 1.0
    mask = np.where(_render_squares(shape, corners, scored) >= 0.5, 255, 0)
    return template, mask.astype(np.uint8)

def measure_pattern_offset(
    response: FloatArray,
    predicted_grid: FloatArray,
    squares: tuple[tuple[int, int], ...] = _ALL_SQUARES,
) -> FloatArray:
    """Return the homography moving predicted pattern points to measured ones.

    The predicted offset-pattern template is registered onto the normalized
    response by enhanced-correlation alignment, restricted to cells lying on
    the scored squares.
    """
    width = _median_square_width(predicted_grid)
    points = predicted_grid.reshape(-1, 2)
    height_limit, width_limit = response.shape
    left = int(max(0, np.floor(np.min(points[:, 0]) - width)))
    top = int(max(0, np.floor(np.min(points[:, 1]) - width)))
    right = int(min(width_limit, np.ceil(np.max(points[:, 0]) + width)))
    bottom = int(min(height_limit, np.ceil(np.max(points[:, 1]) + width)))
    if right - left < 8 or bottom - top < 8:
        raise CalibrationError("The predicted pattern lies outside the image.")
    origin = np.asarray([left, top], dtype=np.float32)
    template, mask = _pattern_template(
        (bottom - top, right - left),
        predicted_grid - origin,
        squares,
    )
    # The normalized response is brightness independent, so faint light just
    # outside the lit region reads as full strength and biases the outer
    # boundary of the scored region; only edges inside it are registered.
    sigma = max(1.0, 0.05 * width)
    margin = max(2, round(2 * sigma))
    mask = cv2.erode(mask, np.ones((2 * margin + 1, 2 * margin + 1), np.uint8))
    template = cv2.GaussianBlur(template, (0, 0), sigma)
    # Saturation equalizes the reflectance-dependent response magnitude of
    # light and dark board squares without moving the pattern edges.
    measured = cv2.GaussianBlur(
        np.clip(
            np.asarray(response[top:bottom, left:right], dtype=np.float32)
            * _RECOVERY_RESPONSE_SATURATION,
            -1.0,
            1.0,
        ),
        (0, 0),
        sigma,
    )
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        _correlation, warp = cv2.findTransformECC(
            template,
            measured,
            warp,
            cv2.MOTION_AFFINE,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6),
            mask,
            1,
        )
    except cv2.error as error:
        raise CalibrationError(
            f"The verification pattern could not be registered: {error}"
        ) from error
    shift = np.asarray(
        [[1, 0, left], [0, 1, top], [0, 0, 1]],
        dtype=np.float64,
    )
    return np.asarray(
        shift
        @ np.vstack((np.asarray(warp, np.float64), (0.0, 0.0, 1.0)))
        @ np.linalg.inv(shift),
        dtype=np.float64,
    )


def pattern_offset_fraction(
    warp: FloatArray,
    predicted_grid: FloatArray,
    squares: tuple[tuple[int, int], ...] = _ALL_SQUARES,
) -> float:
    """Return the largest displacement of a scored square corner in squares."""
    corners = {
        (row + down, column + right)
        for row, column in squares
        for down in (0, 1)
        for right in (0, 1)
    }
    indices = np.asarray(sorted(corners), dtype=np.intp)
    points = predicted_grid[indices[:, 0], indices[:, 1]]
    moved = _apply_homography(warp, points)
    return float(
        np.max(np.linalg.norm(moved - points, axis=1))
    ) / _median_square_width(predicted_grid)


def corrected_projector_grid(
    predicted_grid: FloatArray,
    rendered_projector_grid: FloatArray,
    board_camera_grid: FloatArray,
    hypothesis_projector_grid: FloatArray,
    warp: FloatArray,
) -> FloatArray:
    """Correct a hypothesis mapping by the measured pattern offset.

    The hypothesis predicted rendered projector points at predicted camera
    points; the warp says where they really appeared. The board's projector
    point is where the hypothesis would place the unwarped camera point.
    """
    hypothesis_homography, _mask = cv2.findHomography(
        predicted_grid.reshape(-1, 2),
        rendered_projector_grid.reshape(-1, 2),
        method=0,
    )
    if hypothesis_homography is None:
        raise CalibrationError("The hypothesis mapping could not be fitted.")
    unwarped = _apply_homography(np.linalg.inv(warp), board_camera_grid)
    return np.asarray(
        hypothesis_projector_grid
        + _apply_homography(hypothesis_homography, unwarped)
        - _apply_homography(hypothesis_homography, board_camera_grid),
        dtype=np.float32,
    )


def _projector_to_board_homography(projector_grid: FloatArray) -> FloatArray:
    rows, columns = np.mgrid[0:9, 0:9]
    board = np.stack((columns, rows), axis=2).astype(np.float32)
    homography, _mask = cv2.findHomography(
        projector_grid.reshape(-1, 2),
        board.reshape(-1, 2),
        method=0,
    )
    if homography is None:
        raise CalibrationError("The projector board grid is degenerate.")
    return np.asarray(homography, dtype=np.float64)


def recovery_fringe_image(
    projector_grid: FloatArray,
    projector_size: tuple[int, int],
    axis: int,
    step: int,
) -> GrayImage:
    """Render one phase-shifted fringe with a period of one board square.

    Axis 0 varies along board columns and axis 1 along board rows.
    """
    width, height = projector_size
    homography = _projector_to_board_homography(projector_grid)
    x_values, y_values = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    board = _apply_homography(
        homography,
        np.stack((x_values, y_values), axis=2),
    )
    phase = 2 * np.pi * board[..., axis] + step * np.pi / 2
    mask = np.zeros((height, width), dtype=np.uint8)
    hull = cv2.convexHull(
        np.rint(projector_grid.reshape(-1, 2)).astype(np.int32)
    )
    cv2.fillConvexPoly(mask, hull, 1)
    return np.asarray(
        np.rint(127.5 + 127.5 * np.cos(phase)) * mask,
        dtype=np.uint8,
    )


def refine_projector_grid(
    black: GrayImage,
    fringes: dict[tuple[int, int], GrayImage],
    board_camera_grid: FloatArray,
    rendered_projector_grid: FloatArray,
    squares: tuple[tuple[int, int], ...] = _ALL_SQUARES,
) -> tuple[FloatArray, dict[str, object]]:
    """Measure the board's projector grid from decoded fringe phases."""
    to_board = _projector_to_board_homography(rendered_projector_grid)
    to_projector = np.linalg.inv(to_board)
    board = _square_samples(squares, _RECOVERY_FRINGE_OFFSETS)
    camera_points = _bilinear(board_camera_grid, board)
    prior = _apply_homography(
        to_board,
        _bilinear(rendered_projector_grid, board),
    )
    black_samples = _sample_image(
        cv2.GaussianBlur(black.astype(np.float32), (0, 0), 1.0),
        camera_points,
    )
    decoded = np.zeros_like(board)
    valid = np.ones(board.shape[:2], dtype=bool)
    for axis in (0, 1):
        samples = [
            _sample_image(
                cv2.GaussianBlur(
                    fringes[(axis, step)].astype(np.float32),
                    (0, 0),
                    1.0,
                ),
                camera_points,
            )
            for step in range(_RECOVERY_FRINGE_STEPS)
        ]
        sine = samples[3] - samples[1]
        cosine = samples[0] - samples[2]
        modulation = 0.5 * np.hypot(sine, cosine)
        brightness = np.mean(samples, axis=0) - black_samples
        valid &= (modulation >= _RECOVERY_MINIMUM_MODULATION) & (
            modulation >= _RECOVERY_MINIMUM_MODULATION_FRACTION * brightness
        )
        fraction = np.mod(np.arctan2(sine, cosine) / (2 * np.pi), 1.0)
        expected = prior[..., axis]
        decoded[..., axis] = (
            expected + np.mod(fraction - expected + 0.5, 1.0) - 0.5
        )
    square_valid = (
        np.mean(valid, axis=1) >= _RECOVERY_MINIMUM_VALID_SAMPLE_FRACTION
    )
    valid &= square_valid[:, None]
    refined_squares = np.asarray(squares)[square_valid]
    details: dict[str, object] = {
        "squares": int(np.count_nonzero(square_valid)),
        "samples": int(np.count_nonzero(valid)),
    }
    if (
        len(refined_squares) < _RECOVERY_MINIMUM_REFINED_SQUARES
        or np.ptp(refined_squares[:, 0]) + 1 < _RECOVERY_MINIMUM_REFINED_SPAN
        or np.ptp(refined_squares[:, 1]) + 1 < _RECOVERY_MINIMUM_REFINED_SPAN
    ):
        raise CalibrationError(
            "Fringe refinement found too few usable empty squares "
            f"({details['squares']})."
        )
    board_points = board[valid]
    projector_points = _apply_homography(to_projector, decoded[valid])
    projector_width = _median_square_width(rendered_projector_grid)
    homography, inliers = cv2.findHomography(
        board_points,
        projector_points,
        cv2.RANSAC,
        0.1 * projector_width,
    )
    if homography is None or inliers is None:
        raise CalibrationError("The fringe correspondences could not be fitted.")
    inlier = inliers.reshape(-1).astype(bool)
    residuals = projector_points - _apply_homography(homography, board_points)
    fit_error = float(
        np.median(np.linalg.norm(residuals[inlier], axis=1))
    ) / projector_width
    details["inlier_fraction"] = round(float(np.mean(inlier)), 3)
    details["fit_error_square_fraction"] = round(fit_error, 4)
    if fit_error > _RECOVERY_MAXIMUM_FIT_ERROR:
        raise CalibrationError(
            f"Fringe refinement fit error is {fit_error:.1%} of a square."
        )
    rows, columns = np.mgrid[0:9, 0:9]
    corners = np.stack((columns, rows), axis=2).astype(np.float32)
    refined = _apply_homography(homography, corners)
    inlier_board = board_points[inlier]
    inlier_residuals = residuals[inlier]
    for row in range(9):
        for column in range(9):
            near = (
                np.linalg.norm(inlier_board - corners[row, column], axis=1)
                <= _RECOVERY_LOCAL_RADIUS
            )
            if np.count_nonzero(near) >= _RECOVERY_MINIMUM_LOCAL_SAMPLES:
                refined[row, column] += np.median(
                    inlier_residuals[near],
                    axis=0,
                )
    details["change_square_fraction"] = round(
        float(
            np.max(np.linalg.norm(refined - rendered_projector_grid, axis=2))
        )
        / projector_width,
        4,
    )
    return np.asarray(refined, dtype=np.float32), details


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
        self._queue_recovery_patterns(
            "coarse",
            self._result.mapping.projector_grid,
        )

    def _recovery_scored_squares(self) -> tuple[tuple[int, int], ...]:
        empty = tuple(
            square
            for square in _ALL_SQUARES
            if square not in self._recovery_occupied_squares
        )
        if len(empty) < _RECOVERY_MINIMUM_SCORED_SQUARES:
            return _ALL_SQUARES
        return empty

    def _recovery_message(self, stage: str) -> str:
        state = self._recovery
        if stage == "fringe":
            return (
                "The verification pattern is offset from the predicted "
                "position; projecting fringes to re-measure the alignment."
            )
        if stage == "verify":
            return "Checking the recovered alignment with the verification pattern."
        if self._camera_changed:
            return (
                "The saved calibration belongs to a different camera; "
                "projecting a verification pattern to recover the camera grid."
            )
        displacement = 0.0 if state is None else state.displacement
        return (
            f"The live camera grid differs by {displacement:.1f} px from the "
            "saved calibration; projecting a verification pattern to "
            "determine whether the camera or the board moved."
        )

    def _queue_recovery_patterns(
        self,
        stage: str,
        projector_grid: FloatArray,
    ) -> None:
        state = self._recovery
        if state is None:
            return
        state.stage = stage
        state.announce_pending = (
            stage == "verify" or (stage == "coarse" and state.attempts == 0)
        )
        prefix = "" if stage == "coarse" else f"{stage}-"
        state.queue = [
            (
                f"{prefix}{'inverse' if inverted else 'pattern'}",
                recovery_pattern_image(
                    projector_grid,
                    self._projector_size,
                    inverted=inverted,
                    offset=stage != "coarse",
                ),
            )
            for inverted in (False, True)
        ]
        self._request_next_recovery_projection()

    def _queue_recovery_fringes(self, projector_grid: FloatArray) -> None:
        state = self._recovery
        if state is None:
            return
        state.stage = "fringe"
        state.announce_pending = True
        state.queue = [
            (
                f"fringe-{axis}-{step}",
                recovery_fringe_image(
                    projector_grid,
                    self._projector_size,
                    axis,
                    step,
                ),
            )
            for axis in (0, 1)
            for step in range(_RECOVERY_FRINGE_STEPS)
        ]
        self._request_next_recovery_projection()

    def _request_next_recovery_projection(self) -> None:
        state = self._recovery
        if state is None or not state.queue:
            return
        announce = state.announce_pending
        state.announce_pending = False
        key, image = state.queue.pop(0)
        state.current = key
        state.shown_at = None
        state.requested_at = self._clock()
        message = self._recovery_message(state.stage)
        if announce:
            self.status_changed.emit(message)
        self.recovery_projection_requested.emit(
            image,
            message,
            {
                "reason": state.reason,
                "displacement_pixels": round(state.displacement, 2),
                "stage": state.stage,
                "pattern": key,
                "attempt": state.attempts + 1,
                "scored_squares": len(self._recovery_scored_squares()),
                "board_moved_tested": not self._camera_changed,
                "announce": announce,
            },
        )

    def _check_recovery(self, frame: ImageArray) -> None:
        state = self._recovery
        if state is None or self._result is None or state.current is None:
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
        state.frames[state.current] = captured
        if state.queue:
            self._request_next_recovery_projection()
            return
        if state.stage == "coarse":
            self._finish_coarse_recovery()
        elif state.stage == "measure":
            self._finish_measure_recovery()
        elif state.stage == "fringe":
            self._finish_fringe_recovery()
        else:
            self._finish_verify_recovery()

    def _finish_coarse_recovery(self) -> None:
        state = self._recovery
        if state is None or self._result is None:
            return
        state.attempts += 1
        squares = self._recovery_scored_squares()
        response = normalized_recovery_response(
            state.black_reference,
            state.frames["pattern"],
            state.frames["inverse"],
        )
        hypotheses = score_recovery_hypotheses(
            response,
            state.detection.grid_points,
            None if self._camera_changed else state.old_camera_grid,
            squares,
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
                self._queue_recovery_patterns(
                    "coarse",
                    self._result.mapping.projector_grid,
                )
            return
        state.selected = selected

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
        state.detection = detection
        state.board_camera_grid = detection.grid_points.copy()
        hypothesis = CalibrationResult(
            board_detection=detection,
            mapping=mapping,
            validation=self._result.validation,
        )
        state.candidate = hypothesis
        self._queue_recovery_patterns("measure", mapping.projector_grid)

    def _finish_measure_recovery(self) -> None:
        state = self._recovery
        if (
            state is None
            or state.candidate is None
            or state.board_camera_grid is None
        ):
            return
        hypothesis = state.candidate
        response = normalized_recovery_response(
            state.black_reference,
            state.frames["measure-pattern"],
            state.frames["measure-inverse"],
        )
        state.measure_response = response
        warp: FloatArray | None = None
        try:
            warp = measure_pattern_offset(
                response,
                state.board_camera_grid,
                self._recovery_scored_squares(),
            )
            state.measured_residual = pattern_offset_fraction(
                warp,
                state.board_camera_grid,
                self._recovery_scored_squares(),
            )
        except CalibrationError as error:
            state.last_error = str(error)
        if (
            state.measured_residual is not None
            and state.measured_residual <= _RECOVERY_ACCEPTED_RESIDUAL
        ):
            self._complete_recovery(hypothesis, refined=False)
            return
        if warp is not None:
            try:
                state.candidate = self._with_projector_grid(
                    hypothesis,
                    corrected_projector_grid(
                        state.board_camera_grid,
                        hypothesis.mapping.projector_grid,
                        state.board_camera_grid,
                        hypothesis.mapping.projector_grid,
                        warp,
                    ),
                )
            except CalibrationError as error:
                state.last_error = str(error)
        self._queue_recovery_fringes(state.candidate.mapping.projector_grid)

    def _finish_fringe_recovery(self) -> None:
        state = self._recovery
        if (
            state is None
            or state.candidate is None
            or state.board_camera_grid is None
        ):
            return
        try:
            grid, refinement = refine_projector_grid(
                state.black_reference,
                {
                    (axis, step): state.frames[f"fringe-{axis}-{step}"]
                    for axis in (0, 1)
                    for step in range(_RECOVERY_FRINGE_STEPS)
                },
                state.board_camera_grid,
                state.candidate.mapping.projector_grid,
                self._recovery_scored_squares(),
            )
            state.refinement = refinement
            state.candidate = self._with_projector_grid(state.candidate, grid)
        except CalibrationError as error:
            state.refinement_error = str(error)
        self._queue_recovery_patterns(
            "verify",
            state.candidate.mapping.projector_grid,
        )

    def _finish_verify_recovery(self) -> None:
        state = self._recovery
        if (
            state is None
            or state.candidate is None
            or state.board_camera_grid is None
        ):
            return
        squares = self._recovery_scored_squares()
        response = normalized_recovery_response(
            state.black_reference,
            state.frames["verify-pattern"],
            state.frames["verify-inverse"],
        )
        state.verify_response = response
        score = offset_pattern_score(response, state.board_camera_grid, squares)
        residual: float | None = None
        try:
            residual = pattern_offset_fraction(
                measure_pattern_offset(
                    response,
                    state.board_camera_grid,
                    squares,
                ),
                state.board_camera_grid,
                squares,
            )
        except CalibrationError as error:
            state.last_error = str(error)
        state.verification = {
            "score": round(score, 3),
            "residual_square_fraction": (
                None if residual is None else round(residual, 4)
            ),
        }
        if score < _RECOVERY_MINIMUM_SCORE or (
            residual is None or residual > _RECOVERY_VERIFIED_RESIDUAL
        ):
            offset = "unmeasured" if residual is None else f"{residual:.0%}"
            self._fail_recovery(
                "The re-measured alignment failed the final check (score "
                f"{score:.2f}, offset {offset} of a square)."
            )
            return
        self._complete_recovery(state.candidate, refined=True)

    def _with_projector_grid(
        self,
        result: CalibrationResult,
        projector_grid: FloatArray,
    ) -> CalibrationResult:
        grid = np.asarray(projector_grid, dtype=np.float32)
        width, height = self._projector_size
        if (
            not np.all(np.isfinite(grid))
            or np.any(grid < -0.5)
            or np.any(grid[..., 0] > width - 0.5)
            or np.any(grid[..., 1] > height - 0.5)
        ):
            raise CalibrationError(
                "The re-measured board grid lies outside the projector image."
            )
        camera_grid = result.mapping.camera_grid
        homography, _mask = cv2.findHomography(
            camera_grid.reshape(-1, 2),
            grid.reshape(-1, 2),
            method=0,
        )
        if homography is None:
            raise CalibrationError(
                "The camera-to-projector homography could not be refitted."
            )
        return replace(
            result,
            mapping=replace(
                result.mapping,
                projector_grid=grid.copy(),
                camera_to_projector_homography=np.asarray(
                    homography,
                    dtype=result.mapping.camera_to_projector_homography.dtype,
                ),
            ),
        )

    def _complete_recovery(
        self,
        result: CalibrationResult,
        *,
        refined: bool,
    ) -> None:
        state = self._recovery
        if state is None or state.selected is None:
            return
        selected = state.selected
        details = self._recovery_details(state, selected)
        details["refined"] = refined
        images = self._recovery_images(state)
        self._result = result
        self._tracker.reset(result.board_detection)
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
        if refined and state.verification is not None:
            message = (
                f"Alignment recovered: re-measured after {description}; "
                f"final check score {state.verification['score']:.2f}."
            )
        else:
            message = (
                f"Alignment recovered: {description}; verification pattern "
                f"score {selected.score:.2f}."
            )
        self.status_changed.emit(message)
        self.recovery_completed.emit(result, message, details, images)

    def _fail_recovery(self, message: str) -> None:
        state = self._recovery
        details: dict[str, object] = {}
        images: dict[str, ImageArray] = {}
        if state is not None:
            details = self._recovery_details(state, state.selected)
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
            "stage": state.stage,
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
            "measured_residual_square_fraction": (
                None
                if state.measured_residual is None
                else round(state.measured_residual, 4)
            ),
            "accepted_residual_square_fraction": _RECOVERY_ACCEPTED_RESIDUAL,
            "refinement": state.refinement,
            "refinement_error": state.refinement_error,
            "verification": state.verification,
        }

    @staticmethod
    def _recovery_images(state: _RecoveryState) -> dict[str, ImageArray]:
        images: dict[str, ImageArray] = {
            "recovery-black": state.black_reference,
        }
        for key, name in (
            ("pattern", "recovery-pattern"),
            ("inverse", "recovery-inverse"),
            ("fringe-0-0", "recovery-fringe-columns"),
            ("fringe-1-0", "recovery-fringe-rows"),
        ):
            if key in state.frames:
                images[name] = state.frames[key]
        for response, name in (
            (state.last_response, "recovery-response"),
            (state.measure_response, "recovery-measure-response"),
            (state.verify_response, "recovery-verify-response"),
        ):
            if response is not None:
                images[name] = np.clip(response * 127 + 128, 0, 255).astype(
                    np.uint8
                )
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
