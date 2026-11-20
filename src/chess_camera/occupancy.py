from collections import deque
from dataclasses import dataclass
from typing import Literal, TypeAlias

import cv2
import numpy as np
from numpy.typing import NDArray

from .board_detector import FloatArray, ImageArray

OccupancyLabel = Literal["empty", "occupied", "unknown"]
OccupancyGrid: TypeAlias = NDArray[np.str_]


@dataclass(frozen=True)
class OccupancyResult:
    labels: OccupancyGrid
    scores: FloatArray

    @property
    def occupied_squares(self) -> frozenset[tuple[int, int]]:
        return frozenset(
            (row, column)
            for row in range(8)
            for column in range(8)
            if self.labels[row, column] == "occupied"
        )

    @property
    def occupied_count(self) -> int:
        return int(np.count_nonzero(self.labels == "occupied"))

    @property
    def empty_count(self) -> int:
        return int(np.count_nonzero(self.labels == "empty"))

    @property
    def unknown_count(self) -> int:
        return int(np.count_nonzero(self.labels == "unknown"))


@dataclass(frozen=True)
class OccupancyUpdate:
    state: Literal["stabilizing", "settling", "stable", "motion"]
    observation_count: int
    required_observations: int
    result: OccupancyResult | None


@dataclass(frozen=True)
class _ParityModel:
    color_center: FloatArray
    color_threshold: float
    feature_center: FloatArray
    feature_scale: FloatArray


class SquareOccupancyDetector:
    def __init__(
        self,
        *,
        canonical_size: int = 800,
        required_observations: int = 3,
        required_agreement: int = 2,
        settling_observations: int = 8,
    ) -> None:
        if canonical_size < 320 or canonical_size % 8 != 0:
            raise ValueError(
                "Canonical size must be a multiple of eight and at least 320."
            )
        if required_observations < 1:
            raise ValueError("At least one occupancy observation is required.")
        if not 1 <= required_agreement <= required_observations:
            raise ValueError(
                "Required agreement must fit within the observation window."
            )
        if settling_observations < 1:
            raise ValueError("At least one quiet observation is required.")
        self._canonical_size = canonical_size
        self._required_observations = required_observations
        self._required_agreement = required_agreement
        self._settling_observations = settling_observations
        self._settling_remaining = 0
        self._models: dict[int, _ParityModel] = {}
        self._history: deque[OccupancyGrid] = deque(
            maxlen=required_observations
        )
        self._previous_motion_image: NDArray[np.uint8] | None = None

    def reset(self) -> None:
        self._models.clear()
        self._history.clear()
        self._previous_motion_image = None
        self._settling_remaining = 0

    def observe(
        self,
        frame: ImageArray,
        camera_grid: FloatArray,
    ) -> OccupancyUpdate:
        canonical = rectify_board(
            frame,
            camera_grid,
            self._canonical_size,
        )
        motion_image = cv2.resize(
            cv2.cvtColor(canonical, cv2.COLOR_BGR2GRAY),
            (100, 100),
            interpolation=cv2.INTER_AREA,
        )
        motion_image = cv2.GaussianBlur(motion_image, (5, 5), 0)
        if self._previous_motion_image is not None:
            difference = cv2.absdiff(
                motion_image,
                self._previous_motion_image,
            )
            changed_ratio = float(np.mean(difference > 15))
            if float(np.mean(difference)) > 6.0 and changed_ratio > 0.04:
                self._history.clear()
                self._previous_motion_image = motion_image
                self._settling_remaining = self._settling_observations
                return OccupancyUpdate(
                    "motion",
                    0,
                    self._required_observations,
                    None,
                )
        self._previous_motion_image = motion_image
        if self._settling_remaining:
            self._history.clear()
            self._settling_remaining -= 1
            return OccupancyUpdate(
                "settling",
                self._settling_observations - self._settling_remaining,
                self._settling_observations,
                None,
            )

        square_regions = extract_square_regions(canonical)
        if not self._models:
            self._models = _build_empty_models(square_regions)
        labels, scores = _classify_squares(square_regions, self._models)
        labels = suppress_piece_spillover(labels, scores)
        labels = resolve_binary_occupancy(labels, scores)
        self._history.append(labels)
        observation_count = len(self._history)
        if observation_count < self._required_observations:
            return OccupancyUpdate(
                "stabilizing",
                observation_count,
                self._required_observations,
                OccupancyResult(labels, scores),
            )

        stabilized = np.full((8, 8), "unknown", dtype="<U8")
        history = np.asarray(self._history)
        for label in ("empty", "occupied"):
            counts = np.count_nonzero(history == label, axis=0)
            stabilized[counts >= self._required_agreement] = label
        return OccupancyUpdate(
            "stable",
            observation_count,
            self._required_observations,
            OccupancyResult(stabilized, scores),
        )


def rectify_board(
    frame: ImageArray,
    camera_grid: FloatArray,
    size: int = 800,
) -> ImageArray:
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Occupancy detection requires a BGR camera frame.")
    if camera_grid.shape != (9, 9, 2):
        raise ValueError("Camera grid must have shape (9, 9, 2).")
    if size < 8 or size % 8 != 0:
        raise ValueError("Rectified board size must be divisible by eight.")
    source = np.asarray(
        camera_grid[[0, 0, 8, 8], [0, 8, 8, 0]],
        dtype=np.float32,
    )
    destination = np.asarray(
        [[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(source, destination)
    return np.asarray(
        cv2.warpPerspective(frame, transform, (size, size)),
        dtype=np.uint8,
    )


def extract_square_regions(
    canonical: ImageArray,
    *,
    margin_fraction: float = 0.15,
) -> list[list[ImageArray]]:
    if canonical.ndim != 3 or canonical.shape[2] != 3:
        raise ValueError("Canonical board must be a BGR image.")
    height, width = canonical.shape[:2]
    if height != width or width % 8 != 0:
        raise ValueError("Canonical board must be square and divisible by eight.")
    if not 0 <= margin_fraction < 0.4:
        raise ValueError("Square margin must be between 0% and 40%.")
    square_size = width // 8
    margin = round(square_size * margin_fraction)
    return [
        [
            canonical[
                row * square_size + margin : (row + 1) * square_size - margin,
                column * square_size
                + margin : (column + 1) * square_size
                - margin,
            ]
            for column in range(8)
        ]
        for row in range(8)
    ]


def detect_starting_position_orientation(
    frame: ImageArray,
    camera_grid: FloatArray,
    occupied_squares: frozenset[tuple[int, int]],
    parity_turns: int,
    *,
    minimum_piece_contrast: float = 15.0,
) -> int:
    if parity_turns not in (0, 1, 2, 3):
        raise ValueError("Board orientation must be a quarter-turn count.")
    if minimum_piece_contrast <= 0:
        raise ValueError("Minimum piece contrast must be positive.")
    base_turns = parity_turns % 2
    candidate = base_turns
    candidate_side = frozenset(
        _rotate_square(row, column, candidate)
        for row in (0, 1)
        for column in range(8)
    )
    opposite_side = frozenset(
        _rotate_square(row, column, candidate)
        for row in (6, 7)
        for column in range(8)
    )
    if (
        len(occupied_squares & candidate_side) < 8
        or len(occupied_squares & opposite_side) < 8
    ):
        raise ValueError(
            "White-side orientation requires at least eight occupied squares "
            "on each starting side."
        )

    canonical = rectify_board(frame, camera_grid)
    regions = extract_square_regions(canonical)

    def side_luminance(
        turns: int,
        ui_rows: tuple[int, int],
    ) -> float:
        values: list[float] = []
        for ui_row in ui_rows:
            for ui_column in range(8):
                row, column = _rotate_square(
                    ui_row,
                    ui_column,
                    turns,
                )
                if (row, column) not in occupied_squares:
                    continue
                lab = cv2.cvtColor(
                    regions[row][column],
                    cv2.COLOR_BGR2LAB,
                )
                height, width = lab.shape[:2]
                center = lab[
                    height // 4 : 3 * height // 4,
                    width // 4 : 3 * width // 4,
                    0,
                ]
                values.append(float(np.median(center)))
        return float(np.median(values))

    opposite = (candidate + 2) % 4
    candidate_bottom = side_luminance(candidate, (6, 7))
    candidate_top = side_luminance(candidate, (0, 1))
    contrast = candidate_bottom - candidate_top
    if abs(contrast) < minimum_piece_contrast:
        raise ValueError(
            "White and black starting sides do not have enough contrast."
        )
    return candidate if contrast > 0 else opposite


def suppress_piece_spillover(
    labels: OccupancyGrid,
    scores: FloatArray,
    *,
    weak_score_limit: float = 8.0,
    strong_score_minimum: float = 12.0,
    minimum_score_ratio: float = 2.0,
) -> OccupancyGrid:
    if labels.shape != (8, 8) or scores.shape != (8, 8):
        raise ValueError("Occupancy labels and scores must be 8x8.")
    if (
        weak_score_limit <= 0
        or strong_score_minimum <= weak_score_limit
        or minimum_score_ratio <= 1
    ):
        raise ValueError("Piece-spillover thresholds are invalid.")
    filtered = labels.copy()
    for row in range(8):
        for column in range(8):
            if labels[row, column] != "occupied":
                continue
            score = float(scores[row, column])
            if score > weak_score_limit:
                continue
            neighbors = (
                (row - 1, column),
                (row + 1, column),
                (row, column - 1),
                (row, column + 1),
            )
            for neighbor_row, neighbor_column in neighbors:
                if not (
                    0 <= neighbor_row < 8
                    and 0 <= neighbor_column < 8
                    and labels[neighbor_row, neighbor_column] == "occupied"
                ):
                    continue
                neighbor_score = float(
                    scores[neighbor_row, neighbor_column]
                )
                if (
                    neighbor_score >= strong_score_minimum
                    and neighbor_score
                    >= max(score, 0.1) * minimum_score_ratio
                ):
                    filtered[row, column] = "empty"
                    break
    return filtered


def resolve_binary_occupancy(
    labels: OccupancyGrid,
    scores: FloatArray,
    *,
    uncertain_occupied_threshold: float = 8.0,
) -> OccupancyGrid:
    if labels.shape != (8, 8) or scores.shape != (8, 8):
        raise ValueError("Occupancy labels and scores must be 8x8.")
    if uncertain_occupied_threshold <= 0:
        raise ValueError("Uncertain occupied threshold must be positive.")
    resolved = labels.copy()
    unresolved = resolved == "unknown"
    resolved[np.logical_and(unresolved, scores >= uncertain_occupied_threshold)] = (
        "occupied"
    )
    resolved[np.logical_and(unresolved, scores < uncertain_occupied_threshold)] = (
        "empty"
    )
    return resolved


def _rotate_square(
    row: int,
    column: int,
    clockwise_quarter_turns: int,
) -> tuple[int, int]:
    for _turn in range(clockwise_quarter_turns % 4):
        row, column = column, 7 - row
    return row, column


def _build_empty_models(
    square_regions: list[list[ImageArray]],
) -> dict[int, _ParityModel]:
    models: dict[int, _ParityModel] = {}
    lab_regions = [
        [cv2.cvtColor(region, cv2.COLOR_BGR2LAB) for region in row]
        for row in square_regions
    ]
    for parity in (0, 1):
        references = [
            lab_regions[row][column]
            for row in range(2, 6)
            for column in range(8)
            if (row + column) % 2 == parity
        ]
        pixels = np.concatenate(
            [region.reshape(-1, 3) for region in references],
            axis=0,
        ).astype(np.float32)
        color_center = np.median(pixels, axis=0).astype(np.float32)
        distances = np.linalg.norm(pixels - color_center, axis=1)
        color_threshold = max(8.0, float(np.percentile(distances, 97.5)))
        features = np.asarray(
            [
                _square_features(
                    region,
                    color_center,
                    color_threshold,
                )
                for region in references
            ],
            dtype=np.float32,
        )
        feature_center = np.median(features, axis=0).astype(np.float32)
        deviations = np.median(
            np.abs(features - feature_center),
            axis=0,
        )
        feature_scale = np.maximum(
            deviations * 1.4826,
            np.asarray([0.025, 0.008, 2.5, 2.5], dtype=np.float32),
        ).astype(np.float32)
        models[parity] = _ParityModel(
            color_center,
            color_threshold,
            feature_center,
            feature_scale,
        )
    return models


def _square_features(
    lab_region: ImageArray,
    color_center: FloatArray,
    color_threshold: float,
) -> FloatArray:
    pixels = lab_region.reshape(-1, 3).astype(np.float32)
    distances = np.linalg.norm(pixels - color_center, axis=1)
    anomaly_fraction = float(np.mean(distances > color_threshold))
    luminance = lab_region[:, :, 0]
    edges = cv2.Canny(luminance, 35, 100)
    edge_density = float(np.mean(edges > 0))
    luminance_spread = float(np.std(luminance))
    height, width = luminance.shape
    center = lab_region[
        height // 4 : 3 * height // 4,
        width // 4 : 3 * width // 4,
    ].reshape(-1, 3)
    border_mask = np.ones((height, width), dtype=bool)
    border_mask[
        height // 4 : 3 * height // 4,
        width // 4 : 3 * width // 4,
    ] = False
    border = lab_region[border_mask]
    center_border_distance = float(
        np.linalg.norm(
            np.median(center, axis=0).astype(np.float32)
            - np.median(border, axis=0).astype(np.float32)
        )
    )
    return np.asarray(
        [
            anomaly_fraction,
            edge_density,
            luminance_spread,
            center_border_distance,
        ],
        dtype=np.float32,
    )


def _classify_squares(
    square_regions: list[list[ImageArray]],
    models: dict[int, _ParityModel],
) -> tuple[OccupancyGrid, FloatArray]:
    labels = np.full((8, 8), "unknown", dtype="<U8")
    scores = np.zeros((8, 8), dtype=np.float32)
    for row in range(8):
        for column in range(8):
            parity = (row + column) % 2
            model = models[parity]
            lab_region = cv2.cvtColor(
                square_regions[row][column],
                cv2.COLOR_BGR2LAB,
            )
            features = _square_features(
                lab_region,
                model.color_center,
                model.color_threshold,
            )
            normalized = np.maximum(
                (features - model.feature_center) / model.feature_scale,
                0,
            )
            ordered = np.sort(normalized)
            score = float(np.mean(ordered[-2:]))
            scores[row, column] = score
            support = int(np.count_nonzero(normalized >= 3.0))
            luminance = lab_region[:, :, 0]
            unusable_ratio = float(
                np.mean(np.logical_or(luminance <= 3, luminance >= 252))
            )
            if unusable_ratio > 0.35:
                continue
            if support >= 2 and score >= 3.5:
                labels[row, column] = "occupied"
            elif support == 0 and score < 2.0:
                labels[row, column] = "empty"
    return labels, scores
