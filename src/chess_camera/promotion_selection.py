from dataclasses import dataclass
from typing import Literal, TypeAlias

import cv2
import numpy as np
from numpy.typing import NDArray

from .board_detector import FloatArray, ImageArray
from .legal_game import PromotionPiece

PromotionEdge = Literal["top", "right", "bottom", "left"]
PromotionPolygons: TypeAlias = NDArray[np.float32]

PROMOTION_CHOICES: tuple[PromotionPiece, ...] = ("n", "b", "r", "q")


def promotion_choice_polygons(
    grid: FloatArray,
    edge: PromotionEdge,
) -> PromotionPolygons:
    points = np.asarray(grid, dtype=np.float32)
    if points.shape != (9, 9, 2) or not np.isfinite(points).all():
        raise ValueError("Promotion choices require a finite 9x9 grid.")
    if edge == "top":
        board_edge = points[0]
        outer_edge = board_edge + (board_edge - points[1])
        first_line, second_line = outer_edge, board_edge
        polygons = _line_choice_polygons(first_line, second_line)
    elif edge == "bottom":
        board_edge = points[8]
        outer_edge = board_edge + (board_edge - points[7])
        first_line, second_line = board_edge, outer_edge
        polygons = _line_choice_polygons(first_line, second_line)
    elif edge == "left":
        board_edge = points[:, 0]
        outer_edge = board_edge + (board_edge - points[:, 1])
        polygons = _line_choice_polygons(
            outer_edge,
            board_edge,
        )
    elif edge == "right":
        board_edge = points[:, 8]
        outer_edge = board_edge + (board_edge - points[:, 7])
        polygons = _line_choice_polygons(
            board_edge,
            outer_edge,
        )
    else:
        raise ValueError(f"Unsupported promotion edge: {edge}")
    return np.asarray(polygons, dtype=np.float32)


def _line_choice_polygons(
    first_line: FloatArray,
    second_line: FloatArray,
) -> list[tuple[NDArray[np.float32], ...]]:
    first_column = 2
    return [
        (
            first_line[index],
            first_line[index + 1],
            second_line[index + 1],
            second_line[index],
        )
        for index in range(
            first_column,
            first_column + len(PROMOTION_CHOICES),
        )
    ]


def validate_promotion_polygons(
    polygons: PromotionPolygons,
    width: int,
    height: int,
) -> None:
    if polygons.shape != (len(PROMOTION_CHOICES), 4, 2):
        raise ValueError("Promotion choice polygons have an invalid shape.")
    _validate_polygons_in_frame(polygons, width, height)


def _validate_polygons_in_frame(
    polygons: PromotionPolygons,
    width: int,
    height: int,
) -> None:
    if polygons.ndim != 3 or polygons.shape[1:] != (4, 2):
        raise ValueError("Promotion polygons have an invalid shape.")
    if width <= 0 or height <= 0:
        raise ValueError("Promotion output dimensions must be positive.")
    if (
        np.any(polygons[:, :, 0] < 0)
        or np.any(polygons[:, :, 0] >= width)
        or np.any(polygons[:, :, 1] < 0)
        or np.any(polygons[:, :, 1] >= height)
    ):
        raise ValueError(
            "The promotion choice row is outside the camera or projector."
        )


@dataclass(frozen=True)
class PromotionChoiceObservation:
    occupied_choices: tuple[PromotionPiece, ...]
    changed_fractions: tuple[float, ...]

    @property
    def selected_piece(self) -> PromotionPiece | None:
        if len(self.occupied_choices) != 1:
            return None
        return self.occupied_choices[0]


def observe_promotion_choices(
    reference: ImageArray,
    current: ImageArray,
    polygons: PromotionPolygons,
    *,
    pixel_difference_threshold: int = 20,
    occupied_fraction: float = 0.08,
) -> PromotionChoiceObservation:
    if reference.shape != current.shape:
        raise ValueError("Promotion reference and current frames must match.")
    if reference.ndim != 3 or reference.shape[2] != 3:
        raise ValueError("Promotion selection requires BGR camera frames.")
    if not 1 <= pixel_difference_threshold <= 255:
        raise ValueError("Promotion pixel threshold is invalid.")
    if not 0 < occupied_fraction <= 1:
        raise ValueError("Promotion occupied fraction is invalid.")
    height, width = reference.shape[:2]
    validate_promotion_polygons(polygons, width, height)
    reference_gray = cv2.GaussianBlur(
        cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY),
        (5, 5),
        0,
    )
    current_gray = cv2.GaussianBlur(
        cv2.cvtColor(current, cv2.COLOR_BGR2GRAY),
        (5, 5),
        0,
    )
    changed_fractions: list[float] = []
    occupied: list[PromotionPiece] = []
    for piece, polygon in zip(PROMOTION_CHOICES, polygons, strict=True):
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.rint(polygon).astype(np.int32), 255)
        mask_values = mask != 0
        reference_values = reference_gray[mask_values].astype(np.float32)
        current_values = current_gray[mask_values].astype(np.float32)
        brightness_shift = float(
            np.median(current_values - reference_values)
        )
        normalized = np.clip(
            current_values - brightness_shift,
            0,
            255,
        )
        changed_fraction = float(
            np.mean(
                np.abs(reference_values - normalized)
                >= pixel_difference_threshold
            )
        )
        changed_fractions.append(changed_fraction)
        if changed_fraction >= occupied_fraction:
            occupied.append(piece)
    return PromotionChoiceObservation(
        tuple(occupied),
        tuple(changed_fractions),
    )


class ManipulationSettlingDetector:
    def __init__(
        self,
        *,
        activity_fraction: float = 0.02,
        pixel_difference_threshold: int = 18,
        arming_observations: int = 8,
        activity_observations: int = 2,
        quiet_observations: int = 5,
    ) -> None:
        if not 0 < activity_fraction <= 1:
            raise ValueError("Activity fraction is invalid.")
        if not 1 <= pixel_difference_threshold <= 255:
            raise ValueError("Activity pixel threshold is invalid.")
        if (
            arming_observations < 1
            or activity_observations < 1
            or quiet_observations < 1
        ):
            raise ValueError("Observation requirements must be positive.")
        self._activity_fraction = activity_fraction
        self._pixel_difference_threshold = pixel_difference_threshold
        self._arming_observations = arming_observations
        self._activity_observations = activity_observations
        self._quiet_observations = quiet_observations
        self.reset()

    def reset(self) -> None:
        self._previous: NDArray[np.uint8] | None = None
        self._arming_count = 0
        self._activity_count = 0
        self._activity_seen = False
        self._quiet_count = 0

    def observe(
        self,
        frame: ImageArray,
        polygons: PromotionPolygons,
    ) -> bool:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Promotion activity requires BGR frames.")
        height, width = frame.shape[:2]
        _validate_polygons_in_frame(polygons, width, height)
        gray = cv2.GaussianBlur(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
            (5, 5),
            0,
        )
        mask = np.zeros((height, width), dtype=np.uint8)
        for polygon in polygons:
            cv2.fillConvexPoly(
                mask,
                np.rint(polygon).astype(np.int32),
                255,
            )
        if self._previous is None:
            self._previous = gray
            return False
        changed_fraction = float(
            np.mean(
                cv2.absdiff(self._previous, gray)[mask != 0]
                >= self._pixel_difference_threshold
            )
        )
        self._previous = gray
        if self._arming_count < self._arming_observations:
            self._arming_count += 1
            self._activity_count = 0
            return False
        if changed_fraction >= self._activity_fraction:
            self._activity_count += 1
            if self._activity_count >= self._activity_observations:
                self._activity_seen = True
            self._quiet_count = 0
            return False
        self._activity_count = 0
        if not self._activity_seen:
            return False
        self._quiet_count += 1
        return self._quiet_count >= self._quiet_observations
