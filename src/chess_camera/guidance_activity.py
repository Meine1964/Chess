from collections.abc import Callable
from dataclasses import dataclass
from time import monotonic
from typing import Literal

import cv2
import numpy as np

from .board_detector import FloatArray, ImageArray
from .occupancy import rectify_board


BlackoutSettlingState = Literal["waiting", "stable", "timeout"]
GuidanceLifecycleState = Literal[
    "arming",
    "entering",
    "manipulating",
    "exiting",
    "departed",
]


@dataclass(frozen=True)
class BlackoutSettlingUpdate:
    state: BlackoutSettlingState
    stable_observations: int
    required_observations: int
    elapsed_milliseconds: int


class BlackoutSettlingDetector:
    def __init__(
        self,
        *,
        minimum_milliseconds: int = 400,
        maximum_milliseconds: int = 1000,
        required_observations: int = 3,
        maximum_mean_pixel_change: float = 2.5,
        maximum_square_change: float = 4.0,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        if (
            minimum_milliseconds < 0
            or maximum_milliseconds <= minimum_milliseconds
        ):
            raise ValueError("Blackout settling durations are invalid.")
        if required_observations < 1:
            raise ValueError("Stable observation count must be positive.")
        if maximum_mean_pixel_change <= 0 or maximum_square_change <= 0:
            raise ValueError("Blackout stability thresholds must be positive.")
        self._minimum_seconds = minimum_milliseconds / 1000.0
        self._maximum_seconds = maximum_milliseconds / 1000.0
        self._required_observations = required_observations
        self._maximum_mean_pixel_change = maximum_mean_pixel_change
        self._maximum_square_change = maximum_square_change
        self._clock = clock
        self._started_at: float | None = None
        self._previous: np.ndarray | None = None
        self._stable_observations = 0

    def reset(self) -> None:
        self._started_at = self._clock()
        self._previous = None
        self._stable_observations = 0

    def observe(
        self,
        frame: ImageArray,
        camera_grid: FloatArray,
    ) -> BlackoutSettlingUpdate:
        if self._started_at is None:
            raise RuntimeError("Blackout settling detector is not active.")
        elapsed = self._clock() - self._started_at
        elapsed_milliseconds = max(0, round(elapsed * 1000))
        if elapsed >= self._maximum_seconds:
            return BlackoutSettlingUpdate(
                "timeout",
                self._stable_observations,
                self._required_observations,
                elapsed_milliseconds,
            )
        canonical = rectify_board(frame, camera_grid, 160)
        current = cv2.GaussianBlur(
            cv2.cvtColor(canonical, cv2.COLOR_BGR2GRAY),
            (5, 5),
            0,
        )
        previous = self._previous
        self._previous = current
        if previous is None or elapsed < self._minimum_seconds:
            self._stable_observations = 0
            return BlackoutSettlingUpdate(
                "waiting",
                0,
                self._required_observations,
                elapsed_milliseconds,
            )
        difference = cv2.absdiff(previous, current)
        square_changes = tuple(
            float(
                np.mean(
                    difference[
                        row * 20 : (row + 1) * 20,
                        column * 20 : (column + 1) * 20,
                    ]
                )
            )
            for row in range(8)
            for column in range(8)
        )
        stable = (
            float(np.mean(difference)) <= self._maximum_mean_pixel_change
            and max(square_changes, default=0.0)
            <= self._maximum_square_change
        )
        self._stable_observations = (
            self._stable_observations + 1 if stable else 0
        )
        state: BlackoutSettlingState = (
            "stable"
            if self._stable_observations >= self._required_observations
            else "waiting"
        )
        return BlackoutSettlingUpdate(
            state,
            self._stable_observations,
            self._required_observations,
            elapsed_milliseconds,
        )


class GuidanceActivityDetector:
    def __init__(
        self,
        *,
        arming_observations: int = 10,
        confirmation_observations: int = 2,
        quiet_observations: int = 5,
        pixel_difference_threshold: int = 18,
        minimum_square_fraction: float = 0.06,
        rolling_baseline_before_activity: bool = False,
        require_halo_activity: bool = False,
    ) -> None:
        if (
            arming_observations < 1
            or confirmation_observations < 1
            or quiet_observations < 1
        ):
            raise ValueError("Observation requirements must be positive.")
        if not 1 <= pixel_difference_threshold <= 255:
            raise ValueError("The pixel difference threshold is invalid.")
        if not 0 < minimum_square_fraction <= 1:
            raise ValueError("The minimum square fraction is invalid.")
        self._arming_observations = arming_observations
        self._confirmation_observations = confirmation_observations
        self._quiet_observations = quiet_observations
        self._pixel_difference_threshold = pixel_difference_threshold
        self._minimum_square_fraction = minimum_square_fraction
        self._rolling_baseline_before_activity = (
            rolling_baseline_before_activity
        )
        self._require_halo_activity = require_halo_activity
        self._baseline: np.ndarray | None = None
        self._arming_count = 0
        self._activity_count = 0
        self._manipulation_active = False
        self._quiet_count = 0
        self._lifecycle_state: GuidanceLifecycleState = "arming"
        self._guard_band_pixels = 0
        self._largest_component_pixels = 0
        self._largest_component_halo_pixels = 0
        self._largest_component_inside_pixels = 0

    @property
    def manipulation_active(self) -> bool:
        return self._manipulation_active

    @property
    def lifecycle_state(self) -> GuidanceLifecycleState:
        return self._lifecycle_state

    @property
    def guard_band_pixels(self) -> int:
        return self._guard_band_pixels

    @property
    def largest_component_pixels(self) -> int:
        return self._largest_component_pixels

    @property
    def largest_component_halo_pixels(self) -> int:
        return self._largest_component_halo_pixels

    @property
    def largest_component_inside_pixels(self) -> int:
        return self._largest_component_inside_pixels

    def reset(self) -> None:
        self._baseline = None
        self._arming_count = 0
        self._activity_count = 0
        self._manipulation_active = False
        self._quiet_count = 0
        self._lifecycle_state = "arming"
        self._guard_band_pixels = 0
        self._largest_component_pixels = 0
        self._largest_component_halo_pixels = 0
        self._largest_component_inside_pixels = 0

    def observe(
        self,
        frame: ImageArray,
        camera_grid: FloatArray,
        ignored_squares: tuple[tuple[int, int], ...] = (),
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
            self._lifecycle_state = "arming"
            return False
        if self._baseline is None:
            raise RuntimeError("Guidance activity baseline is unavailable.")

        guidance_mask = _guidance_region_mask(frame.shape[:2], grid)
        board_mask = _board_region_mask(frame.shape[:2], grid)
        outside_mask = cv2.bitwise_and(
            guidance_mask,
            cv2.bitwise_not(board_mask),
        )
        mask = (
            board_mask.copy()
            if self._require_halo_activity and self._manipulation_active
            else guidance_mask.copy()
        )
        ignored_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for row, column in ignored_squares:
            if not 0 <= row < 8 or not 0 <= column < 8:
                raise ValueError("Ignored squares must be inside the board.")
            polygon = np.rint(
                np.asarray(
                    (
                        grid[row, column],
                        grid[row, column + 1],
                        grid[row + 1, column + 1],
                        grid[row + 1, column],
                    ),
                    dtype=np.float32,
                )
            ).astype(np.int32)
            cv2.fillConvexPoly(ignored_mask, polygon, 255)
        if ignored_squares:
            guard_ratio = 0.5 if self._manipulation_active else 0.15
            guard_band = max(
                1,
                round(_median_square_area(grid) ** 0.5 * guard_ratio),
            )
            self._guard_band_pixels = guard_band
            ignored_mask = cv2.dilate(
                ignored_mask,
                np.ones(
                    (guard_band * 2 + 1, guard_band * 2 + 1),
                    dtype=np.uint8,
                ),
                iterations=1,
            )
            mask[ignored_mask != 0] = 0
        else:
            self._guard_band_pixels = 0
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
        illumination_delta = (
            normalized.astype(np.float32)
            - self._baseline.astype(np.float32)
        )
        illumination_delta[mask == 0] = 0
        illumination_kernel = max(
            3,
            round(_median_square_area(grid) ** 0.5 * 2.0),
        )
        if illumination_kernel % 2 == 0:
            illumination_kernel += 1
        smooth_illumination = _smooth_guidance_illumination(
            illumination_delta,
            mask,
            board_mask,
            illumination_kernel,
        )
        photometric_normalized = np.clip(
            normalized.astype(np.float32) - smooth_illumination,
            0,
            255,
        ).astype(np.uint8)
        difference = cv2.absdiff(
            self._baseline,
            photometric_normalized,
        )
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
        component_count, labels, statistics, _centroids = (
            cv2.connectedComponentsWithStats(changed)
        )
        minimum_component = (
            _median_square_area(grid) * self._minimum_square_fraction
        )
        qualifying_components = [
            (
                int(statistics[component, cv2.CC_STAT_AREA]),
                int(
                    np.count_nonzero(
                        (labels == component) & (outside_mask != 0)
                    )
                ),
                int(
                    np.count_nonzero(
                        (labels == component) & (board_mask != 0)
                    )
                ),
            )
            for component in range(1, component_count)
            if (
                not self._require_halo_activity
                or self._manipulation_active
                or (
                    np.count_nonzero(
                        (labels == component) & (outside_mask != 0)
                    )
                    >= minimum_component
                    and np.count_nonzero(
                        (labels == component) & (board_mask != 0)
                    )
                    >= minimum_component
                )
            )
        ]
        (
            largest_component,
            largest_halo_component,
            largest_inside_component,
        ) = max(
            qualifying_components,
            default=(0, 0, 0),
        )
        self._largest_component_pixels = largest_component
        self._largest_component_halo_pixels = largest_halo_component
        self._largest_component_inside_pixels = largest_inside_component
        changed_materially = largest_component >= minimum_component
        if not self._manipulation_active and changed_materially:
            self._activity_count += 1
            self._lifecycle_state = "entering"
            if self._activity_count >= self._confirmation_observations:
                self._manipulation_active = True
                self._quiet_count = 0
                self._lifecycle_state = "manipulating"
            return False
        if not self._manipulation_active:
            self._activity_count = 0
            self._lifecycle_state = "arming"
            if self._rolling_baseline_before_activity:
                self._baseline = gray
            return False

        if changed_materially:
            self._quiet_count = 0
            self._lifecycle_state = "manipulating"
        else:
            self._quiet_count += 1
            self._lifecycle_state = "exiting"
        departed = self._quiet_count >= self._quiet_observations
        if departed:
            self._lifecycle_state = "departed"
        return departed


class BoardEdgeActivityDetector:
    def __init__(
        self,
        *,
        arming_observations: int = 10,
        confirmation_observations: int = 1,
        quiet_observations: int = 1,
        pixel_difference_threshold: int = 18,
        minimum_square_fraction: float = 0.06,
    ) -> None:
        if (
            arming_observations < 1
            or confirmation_observations < 1
            or quiet_observations < 1
        ):
            raise ValueError("Observation requirements must be positive.")
        if not 1 <= pixel_difference_threshold <= 255:
            raise ValueError("The pixel difference threshold is invalid.")
        if not 0 < minimum_square_fraction <= 1:
            raise ValueError("The minimum square fraction is invalid.")
        self._arming_observations = arming_observations
        self._confirmation_observations = confirmation_observations
        self._quiet_observations = quiet_observations
        self._pixel_difference_threshold = pixel_difference_threshold
        self._minimum_square_fraction = minimum_square_fraction
        self._baseline: np.ndarray | None = None
        self._arming_count = 0
        self._crossing_count = 0
        self._quiet_count = 0
        self._active = False
        self._crossing_detected = False
        self._largest_component_pixels = 0
        self._largest_component_inside_pixels = 0
        self._largest_component_outside_pixels = 0
        self._largest_component_outer_halo_pixels = 0
        self._largest_component_bounds: tuple[int, int, int, int] | None = None

    @property
    def active(self) -> bool:
        return self._active

    @property
    def crossing_detected(self) -> bool:
        return self._crossing_detected

    @property
    def largest_component_pixels(self) -> int:
        return self._largest_component_pixels

    @property
    def largest_component_inside_pixels(self) -> int:
        return self._largest_component_inside_pixels

    @property
    def largest_component_outside_pixels(self) -> int:
        return self._largest_component_outside_pixels

    @property
    def largest_component_outer_halo_pixels(self) -> int:
        return self._largest_component_outer_halo_pixels

    @property
    def largest_component_bounds(self) -> tuple[int, int, int, int] | None:
        return self._largest_component_bounds

    def reset(self) -> None:
        self._baseline = None
        self._arming_count = 0
        self._crossing_count = 0
        self._quiet_count = 0
        self._active = False
        self._crossing_detected = False
        self._largest_component_pixels = 0
        self._largest_component_inside_pixels = 0
        self._largest_component_outside_pixels = 0
        self._largest_component_outer_halo_pixels = 0
        self._largest_component_bounds = None

    def observe(
        self,
        frame: ImageArray,
        camera_grid: FloatArray,
    ) -> bool:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Board-edge activity requires a BGR camera frame.")
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
            self._crossing_detected = False
            self._largest_component_pixels = 0
            self._largest_component_inside_pixels = 0
            self._largest_component_outside_pixels = 0
            self._largest_component_outer_halo_pixels = 0
            self._largest_component_bounds = None
            return False
        if self._baseline is None:
            raise RuntimeError("Board-edge activity baseline is unavailable.")

        guidance_mask = _guidance_region_mask(frame.shape[:2], grid)
        board_mask = _board_region_mask(frame.shape[:2], grid)
        outside_mask = cv2.bitwise_and(
            guidance_mask,
            cv2.bitwise_not(board_mask),
        )
        square_area = _median_square_area(grid)
        outer_rim_width = max(1, round(square_area**0.5 * 0.15))
        outer_halo_mask = cv2.bitwise_and(
            guidance_mask,
            cv2.bitwise_not(
                cv2.erode(
                    guidance_mask,
                    np.ones(
                        (
                            outer_rim_width * 2 + 1,
                            outer_rim_width * 2 + 1,
                        ),
                        dtype=np.uint8,
                    ),
                    iterations=1,
                )
            ),
        )
        baseline_values = self._baseline[guidance_mask != 0].astype(
            np.float32
        )
        current_values = gray[guidance_mask != 0].astype(np.float32)
        brightness_shift = float(
            np.median(current_values - baseline_values)
        )
        normalized = np.clip(
            gray.astype(np.float32) - brightness_shift,
            0,
            255,
        ).astype(np.uint8)
        illumination_delta = (
            normalized.astype(np.float32)
            - self._baseline.astype(np.float32)
        )
        illumination_delta[guidance_mask == 0] = 0
        illumination_kernel = max(
            3,
            round(_median_square_area(grid) ** 0.5 * 2.0),
        )
        if illumination_kernel % 2 == 0:
            illumination_kernel += 1
        smooth_illumination = _smooth_guidance_illumination(
            illumination_delta,
            guidance_mask,
            board_mask,
            illumination_kernel,
        )
        photometric_normalized = np.clip(
            normalized.astype(np.float32) - smooth_illumination,
            0,
            255,
        ).astype(np.uint8)
        difference = cv2.absdiff(
            self._baseline,
            photometric_normalized,
        )
        changed = np.where(
            (difference >= self._pixel_difference_threshold)
            & (guidance_mask != 0),
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
        component_count, labels, statistics, _centroids = (
            cv2.connectedComponentsWithStats(changed)
        )
        minimum_component = square_area * self._minimum_square_fraction
        components = [
            (
                int(statistics[component, cv2.CC_STAT_AREA]),
                int(
                    np.count_nonzero(
                        (labels == component) & (board_mask != 0)
                    )
                ),
                int(
                    np.count_nonzero(
                        (labels == component) & (outside_mask != 0)
                    )
                ),
                int(
                    np.count_nonzero(
                        (labels == component) & (outer_halo_mask != 0)
                    )
                ),
                (
                    int(statistics[component, cv2.CC_STAT_LEFT]),
                    int(statistics[component, cv2.CC_STAT_TOP]),
                    int(statistics[component, cv2.CC_STAT_WIDTH]),
                    int(statistics[component, cv2.CC_STAT_HEIGHT]),
                ),
            )
            for component in range(1, component_count)
        ]
        (
            self._largest_component_pixels,
            self._largest_component_inside_pixels,
            self._largest_component_outside_pixels,
            self._largest_component_outer_halo_pixels,
            self._largest_component_bounds,
        ) = max(
            components,
            key=lambda candidate: (
                min(candidate[1], candidate[2]),
                candidate[0],
            ),
            default=(0, 0, 0, 0, None),
        )
        crossing = (
            self._largest_component_inside_pixels >= minimum_component
            and self._largest_component_outside_pixels >= minimum_component
            and self._largest_component_outer_halo_pixels
            >= max(1.0, minimum_component * 0.1)
        )
        self._crossing_detected = crossing
        if crossing:
            self._crossing_count += 1
            self._quiet_count = 0
            if self._crossing_count >= self._confirmation_observations:
                self._active = True
            return self._active

        self._crossing_count = 0
        if self._active:
            self._quiet_count += 1
            if self._quiet_count < self._quiet_observations:
                return True
        self._active = False
        self._quiet_count = 0
        self._baseline = gray
        return False


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


def _board_region_mask(
    frame_shape: tuple[int, int],
    camera_grid: FloatArray,
) -> np.ndarray:
    height, width = frame_shape
    polygon = np.rint(
        np.asarray(
            (
                camera_grid[0, 0],
                camera_grid[0, 8],
                camera_grid[8, 8],
                camera_grid[8, 0],
            ),
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


def _smooth_guidance_illumination(
    illumination_delta: np.ndarray,
    activity_mask: np.ndarray,
    board_mask: np.ndarray,
    kernel_size: int,
) -> np.ndarray:
    smooth = np.zeros_like(illumination_delta, dtype=np.float32)
    for region_mask in (
        cv2.bitwise_and(activity_mask, board_mask),
        cv2.bitwise_and(
            activity_mask,
            cv2.bitwise_not(board_mask),
        ),
    ):
        weights = (region_mask != 0).astype(np.float32)
        weighted_delta = cv2.GaussianBlur(
            illumination_delta * weights,
            (kernel_size, kernel_size),
            0,
        )
        smoothed_weights = cv2.GaussianBlur(
            weights,
            (kernel_size, kernel_size),
            0,
        )
        valid = (region_mask != 0) & (smoothed_weights > 1e-6)
        smooth[valid] = weighted_delta[valid] / smoothed_weights[valid]
    return smooth
