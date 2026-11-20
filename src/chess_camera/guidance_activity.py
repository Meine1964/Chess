import cv2
import numpy as np

from .board_detector import FloatArray, ImageArray


class GuidanceActivityDetector:
    def __init__(
        self,
        *,
        arming_observations: int = 10,
        confirmation_observations: int = 2,
        pixel_difference_threshold: int = 18,
        minimum_square_fraction: float = 0.06,
    ) -> None:
        if arming_observations < 1 or confirmation_observations < 1:
            raise ValueError("Observation requirements must be positive.")
        if not 1 <= pixel_difference_threshold <= 255:
            raise ValueError("The pixel difference threshold is invalid.")
        if not 0 < minimum_square_fraction <= 1:
            raise ValueError("The minimum square fraction is invalid.")
        self._arming_observations = arming_observations
        self._confirmation_observations = confirmation_observations
        self._pixel_difference_threshold = pixel_difference_threshold
        self._minimum_square_fraction = minimum_square_fraction
        self._baseline: np.ndarray | None = None
        self._arming_count = 0
        self._activity_count = 0

    def reset(self) -> None:
        self._baseline = None
        self._arming_count = 0
        self._activity_count = 0

    def observe(
        self,
        frame: ImageArray,
        camera_grid: FloatArray,
    ) -> bool:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Guidance activity requires a BGR camera frame.")
        grid = np.asarray(camera_grid, dtype=np.float32)
        if grid.shape != (9, 9, 2) or not np.isfinite(grid).all():
            raise ValueError("The camera grid must contain finite 9x9 points.")
        gray = cv2.GaussianBlur(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
            (5, 5),
            0,
        )
        if self._arming_count < self._arming_observations:
            self._baseline = gray
            self._arming_count += 1
            self._activity_count = 0
            return False
        if self._baseline is None:
            raise RuntimeError("Guidance activity baseline is unavailable.")

        mask = _guidance_region_mask(frame.shape[:2], grid)
        baseline_values = self._baseline[mask != 0].astype(np.float32)
        current_values = gray[mask != 0].astype(np.float32)
        brightness_shift = float(
            np.median(current_values - baseline_values)
        )
        normalized = np.clip(
            gray.astype(np.float32) - brightness_shift,
            0,
            255,
        ).astype(np.uint8)
        difference = cv2.absdiff(self._baseline, normalized)
        changed = np.where(
            (difference >= self._pixel_difference_threshold) & (mask != 0),
            255,
            0,
        ).astype(np.uint8)
        changed = cv2.morphologyEx(
            changed,
            cv2.MORPH_OPEN,
            np.ones((3, 3), dtype=np.uint8),
        )
        changed = cv2.dilate(
            changed,
            np.ones((5, 5), dtype=np.uint8),
            iterations=1,
        )
        component_count, _labels, statistics, _centroids = (
            cv2.connectedComponentsWithStats(changed)
        )
        largest_component = (
            0
            if component_count <= 1
            else int(np.max(statistics[1:, cv2.CC_STAT_AREA]))
        )
        minimum_component = (
            _median_square_area(grid) * self._minimum_square_fraction
        )
        if largest_component >= minimum_component:
            self._activity_count += 1
        else:
            self._activity_count = 0
        return self._activity_count >= self._confirmation_observations


def _guidance_region_mask(
    frame_shape: tuple[int, int],
    camera_grid: FloatArray,
) -> np.ndarray:
    height, width = frame_shape
    top_left = (
        camera_grid[0, 0]
        + (camera_grid[0, 0] - camera_grid[1, 0])
        + (camera_grid[0, 0] - camera_grid[0, 1])
    )
    top_right = (
        camera_grid[0, 8]
        + (camera_grid[0, 8] - camera_grid[1, 8])
        + (camera_grid[0, 8] - camera_grid[0, 7])
    )
    bottom_right = (
        camera_grid[8, 8]
        + (camera_grid[8, 8] - camera_grid[7, 8])
        + (camera_grid[8, 8] - camera_grid[8, 7])
    )
    bottom_left = (
        camera_grid[8, 0]
        + (camera_grid[8, 0] - camera_grid[7, 0])
        + (camera_grid[8, 0] - camera_grid[8, 1])
    )
    polygon = np.rint(
        np.asarray(
            [top_left, top_right, bottom_right, bottom_left],
            dtype=np.float32,
        )
    ).astype(np.int32)
    polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, polygon, 255)
    return mask


def _median_square_area(camera_grid: FloatArray) -> float:
    areas = [
        abs(
            cv2.contourArea(
                np.asarray(
                    [
                        camera_grid[row, column],
                        camera_grid[row, column + 1],
                        camera_grid[row + 1, column + 1],
                        camera_grid[row + 1, column],
                    ],
                    dtype=np.float32,
                )
            )
        )
        for row in range(8)
        for column in range(8)
    ]
    median_area = float(np.median(areas))
    if median_area <= 0:
        raise ValueError("The camera grid contains degenerate squares.")
    return median_area
