from collections.abc import Callable
from dataclasses import dataclass
from time import monotonic, perf_counter
from typing import Literal

import cv2
import numpy as np

from .board_detector import FloatArray, ImageArray
from .image_statistics import median_uint8_difference, offset_uint8
from .occupancy import rectify_board

_EDGE_CROP_MARGIN = 24
_EDGE_CROP_REQUIRED_MARGIN = 4


BlackoutSettlingState = Literal["waiting", "stable", "timeout"]
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
        self._observation_count = 0
        self._last_observation_seconds = 0.0
        self._crop: tuple[int, int, int, int] | None = None
        self._crop_frame_shape: tuple[int, int] | None = None
        self._geometry: _EdgeGeometry | None = None

    @property
    def active(self) -> bool:
        return self._active

    @property
    def observation_count(self) -> int:
        return self._observation_count

    @property
    def last_observation_seconds(self) -> float:
        return self._last_observation_seconds

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
        started = perf_counter()
        try:
            return self._observe(frame, camera_grid)
        finally:
            self._observation_count += 1
            self._last_observation_seconds = perf_counter() - started

    def _observe(
        self,
        frame: ImageArray,
        camera_grid: FloatArray,
    ) -> bool:
        geometry, gray = self._prepare(frame, camera_grid)
        if self._arming_count < self._arming_observations:
            self._baseline = gray
            self._arming_count += 1
            self._clear_measurements()
            return False
        if self._baseline is None:
            raise RuntimeError("Board-edge activity baseline is unavailable.")

        crossing = self._measure_crossing(gray, geometry)
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

    def _clear_measurements(self) -> None:
        self._crossing_detected = False
        self._largest_component_pixels = 0
        self._largest_component_inside_pixels = 0
        self._largest_component_outside_pixels = 0
        self._largest_component_outer_halo_pixels = 0
        self._largest_component_bounds = None

    def _prepare(
        self,
        frame: ImageArray,
        camera_grid: FloatArray,
    ) -> tuple["_EdgeGeometry", np.ndarray]:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Board-edge activity requires a BGR camera frame.")
        grid = np.asarray(camera_grid, dtype=np.float32)
        if grid.shape != (9, 9, 2) or not np.isfinite(grid).all():
            raise ValueError("The camera grid must contain finite 9x9 points.")
        geometry = self._geometry_for(frame.shape[:2], grid)
        left, top, right, bottom = geometry.crop
        gray = cv2.GaussianBlur(
            cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_BGR2GRAY),
            (5, 5),
            0,
        )
        return geometry, gray

    def _measure_crossing(
        self,
        gray: np.ndarray,
        geometry: "_EdgeGeometry",
    ) -> bool:
        _labels, components = self._changed_components(gray, geometry)
        minimum_component = geometry.square_area * self._minimum_square_fraction
        self._record_component(
            max(
                components,
                key=lambda candidate: (
                    min(candidate[1], candidate[2]),
                    candidate[0],
                ),
                default=(0, 0, 0, 0, None),
            )
        )
        crossing = (
            self._largest_component_inside_pixels >= minimum_component
            and self._largest_component_outside_pixels >= minimum_component
            and self._largest_component_outer_halo_pixels
            >= max(1.0, minimum_component * 0.1)
        )
        return crossing

    def _record_component(self, component: "_Component") -> None:
        (
            self._largest_component_pixels,
            self._largest_component_inside_pixels,
            self._largest_component_outside_pixels,
            self._largest_component_outer_halo_pixels,
            self._largest_component_bounds,
        ) = component

    def _changed_components(
        self,
        gray: np.ndarray,
        geometry: "_EdgeGeometry",
    ) -> tuple[np.ndarray, list["_Component"]]:
        if self._baseline is None:
            raise RuntimeError("Board-edge activity baseline is unavailable.")
        left, top = geometry.crop[:2]
        guidance = geometry.guidance
        brightness_shift = median_uint8_difference(
            gray[guidance],
            self._baseline[guidance],
        )
        normalized = offset_uint8(gray, -brightness_shift)
        illumination_delta = (
            normalized.astype(np.float32)
            - self._baseline.astype(np.float32)
        )
        illumination_delta[~guidance] = 0
        smooth_illumination = np.zeros_like(illumination_delta)
        kernel = (geometry.illumination_kernel, geometry.illumination_kernel)
        for valid, weights, smoothed_weights in geometry.smoothing_regions:
            weighted_delta = cv2.GaussianBlur(
                illumination_delta * weights,
                kernel,
                0,
                borderType=cv2.BORDER_CONSTANT,
            )
            smooth_illumination[valid] = (
                weighted_delta[valid] / smoothed_weights[valid]
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
            (difference >= self._pixel_difference_threshold) & guidance,
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
        inside_counts = np.bincount(
            labels[geometry.board],
            minlength=component_count,
        )
        outside_counts = np.bincount(
            labels[geometry.outside],
            minlength=component_count,
        )
        outer_halo_counts = np.bincount(
            labels[geometry.outer_halo],
            minlength=component_count,
        )
        components: list[_Component] = [
            (
                int(statistics[component, cv2.CC_STAT_AREA]),
                int(inside_counts[component]),
                int(outside_counts[component]),
                int(outer_halo_counts[component]),
                (
                    int(statistics[component, cv2.CC_STAT_LEFT]) + left,
                    int(statistics[component, cv2.CC_STAT_TOP]) + top,
                    int(statistics[component, cv2.CC_STAT_WIDTH]),
                    int(statistics[component, cv2.CC_STAT_HEIGHT]),
                ),
            )
            for component in range(1, component_count)
        ]
        return labels, components

    def _geometry_for(
        self,
        frame_shape: tuple[int, int],
        grid: FloatArray,
    ) -> "_EdgeGeometry":
        height, width = frame_shape
        guidance_polygon = _guidance_region_polygon(frame_shape, grid)
        x, y, polygon_width, polygon_height = cv2.boundingRect(
            guidance_polygon
        )

        def expanded(margin: int) -> tuple[int, int, int, int]:
            return (
                max(0, x - margin),
                max(0, y - margin),
                min(width, x + polygon_width + margin),
                min(height, y + polygon_height + margin),
            )

        required = expanded(_EDGE_CROP_REQUIRED_MARGIN)
        crop = self._crop
        if (
            crop is None
            or self._crop_frame_shape != frame_shape
            or required[0] < crop[0]
            or required[1] < crop[1]
            or required[2] > crop[2]
            or required[3] > crop[3]
        ):
            if crop is not None:
                self.reset()
            crop = expanded(_EDGE_CROP_MARGIN)
            self._crop = crop
            self._crop_frame_shape = frame_shape
        geometry = self._geometry
        grid_key = grid.tobytes()
        if (
            geometry is None
            or geometry.crop != crop
            or geometry.grid_key != grid_key
        ):
            geometry = _edge_geometry(
                crop,
                grid_key,
                guidance_polygon,
                _board_region_polygon(frame_shape, grid),
                _median_square_area(grid),
                _inner_square_polygons(frame_shape, grid),
            )
            self._geometry = geometry
        return geometry


class BoardEdgeHandGate(BoardEdgeActivityDetector):
    """Blocks registration while an arm crosses the edge of a clean board."""

    def __init__(
        self,
        *,
        still_release_seconds: float = 5.0,
        still_pixel_fraction: float = 0.01,
        minimum_square_coverage: float = 0.1,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        super().__init__()
        if still_release_seconds <= 0:
            raise ValueError("The stillness release duration must be positive.")
        if not 0 <= still_pixel_fraction < 1:
            raise ValueError("The stillness pixel fraction is invalid.")
        if not 0 < minimum_square_coverage <= 1:
            raise ValueError("The minimum square coverage is invalid.")
        self._still_release_seconds = still_release_seconds
        self._still_pixel_fraction = still_pixel_fraction
        self._minimum_square_coverage = minimum_square_coverage
        self._largest_square_coverage = 0.0
        self._clock = clock
        self._baseline_grid_key: bytes | None = None
        self._previous_gray: np.ndarray | None = None
        self._still_since: float | None = None
        self._released = False

    @property
    def has_baseline(self) -> bool:
        return self._baseline is not None

    @property
    def largest_square_coverage(self) -> float:
        """Largest inner-square fraction covered by the reported component."""
        return self._largest_square_coverage

    @property
    def released(self) -> bool:
        """Whether the latest observation released a still crossing."""
        return self._released

    def _clear_measurements(self) -> None:
        super()._clear_measurements()
        self._largest_square_coverage = 0.0

    def reset(self) -> None:
        super().reset()
        self._largest_square_coverage = 0.0
        self._baseline_grid_key = None
        self._previous_gray = None
        self._still_since = None
        self._released = False

    def capture_baseline(
        self,
        frame: ImageArray,
        camera_grid: FloatArray,
    ) -> None:
        geometry, gray = self._prepare(frame, camera_grid)
        self._baseline = gray
        self._baseline_grid_key = geometry.grid_key
        self._previous_gray = None
        self._still_since = None
        self._released = False
        self._active = False
        self._clear_measurements()

    def _observe(
        self,
        frame: ImageArray,
        camera_grid: FloatArray,
    ) -> bool:
        self._released = False
        geometry, gray = self._prepare(frame, camera_grid)
        if (
            self._baseline is None
            or self._baseline_grid_key != geometry.grid_key
            or self._baseline.shape != gray.shape
        ):
            self.reset()
            return False
        crossing = self._measure_crossing(gray, geometry)
        self._crossing_detected = crossing
        previous = self._previous_gray
        self._previous_gray = gray
        if not crossing:
            self._active = False
            self._still_since = None
            return False
        now = self._clock()
        if previous is None or not self._is_still(previous, gray, geometry):
            self._still_since = now
        elif self._still_since is None:
            self._still_since = now
        if now - self._still_since >= self._still_release_seconds:
            self._baseline = gray
            self._still_since = None
            self._active = False
            self._released = True
            return False
        self._active = True
        return True

    def _measure_crossing(
        self,
        gray: np.ndarray,
        geometry: "_EdgeGeometry",
    ) -> bool:
        labels, components = self._changed_components(gray, geometry)
        if not components:
            self._clear_measurements()
            return False
        component_count = len(components) + 1
        square_pixels = np.bincount(
            labels[geometry.square_inner].astype(np.intp) * 64
            + geometry.square_inner_labels,
            minlength=component_count * 64,
        ).reshape(component_count, 64)
        coverages = (square_pixels / geometry.square_inner_pixels).max(axis=1)
        minimum_component = geometry.square_area * self._minimum_square_fraction
        minimum_rim = max(1.0, minimum_component * 0.1)

        def ranking(index: int) -> tuple[bool, float, int]:
            _area, _inside, outside, rim, _bounds = components[index]
            coverage = float(coverages[index + 1])
            qualifies = (
                coverage >= self._minimum_square_coverage
                and outside >= minimum_component
                and rim >= minimum_rim
            )
            return qualifies, coverage, outside

        selected = max(range(len(components)), key=ranking)
        self._record_component(components[selected])
        self._largest_square_coverage = float(coverages[selected + 1])
        return ranking(selected)[0]

    def _is_still(
        self,
        previous: np.ndarray,
        current: np.ndarray,
        geometry: "_EdgeGeometry",
    ) -> bool:
        bounds = self._largest_component_bounds
        if bounds is None:
            return False
        left, top = geometry.crop[:2]
        margin = round(geometry.square_area**0.5)
        height, width = current.shape
        x0 = max(0, bounds[0] - left - margin)
        y0 = max(0, bounds[1] - top - margin)
        x1 = min(width, bounds[0] - left + bounds[2] + margin)
        y1 = min(height, bounds[1] - top + bounds[3] + margin)
        region = geometry.guidance[y0:y1, x0:x1]
        region_pixels = np.count_nonzero(region)
        if region_pixels == 0:
            return False
        previous_region = previous[y0:y1, x0:x1]
        current_region = current[y0:y1, x0:x1]
        shift = median_uint8_difference(
            current_region[region],
            previous_region[region],
        )
        difference = cv2.absdiff(
            previous_region,
            offset_uint8(current_region, -shift),
        )
        changed = np.count_nonzero(
            (difference >= self._pixel_difference_threshold) & region
        )
        return changed <= region_pixels * self._still_pixel_fraction


_Component = tuple[int, int, int, int, tuple[int, int, int, int] | None]
_SQUARE_SCORE_INSET = 0.12


@dataclass(frozen=True)
class _EdgeGeometry:
    crop: tuple[int, int, int, int]
    grid_key: bytes
    guidance: np.ndarray
    board: np.ndarray
    outside: np.ndarray
    outer_halo: np.ndarray
    square_area: float
    illumination_kernel: int
    smoothing_regions: tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...]
    square_inner: np.ndarray
    square_inner_labels: np.ndarray
    square_inner_pixels: np.ndarray


def _edge_geometry(
    crop: tuple[int, int, int, int],
    grid_key: bytes,
    guidance_polygon: np.ndarray,
    board_polygon: np.ndarray,
    square_area: float,
    inner_square_polygons: list[np.ndarray],
) -> _EdgeGeometry:
    left, top, right, bottom = crop
    offset = np.asarray([left, top], dtype=np.int32)
    guidance_mask = np.zeros((bottom - top, right - left), dtype=np.uint8)
    cv2.fillConvexPoly(guidance_mask, guidance_polygon - offset, 255)
    board_mask = np.zeros_like(guidance_mask)
    cv2.fillConvexPoly(board_mask, board_polygon - offset, 255)
    square_labels = np.full(guidance_mask.shape, -1, dtype=np.int16)
    for index, polygon in enumerate(inner_square_polygons):
        cv2.fillConvexPoly(square_labels, polygon - offset, index)
    square_inner = square_labels >= 0
    square_inner_labels = square_labels[square_inner].astype(np.intp)
    square_inner_pixels = np.maximum(
        np.bincount(square_inner_labels, minlength=64),
        1,
    ).astype(np.float64)
    outer_rim_width = max(1, round(square_area**0.5 * 0.15))
    eroded = cv2.erode(
        guidance_mask,
        np.ones(
            (outer_rim_width * 2 + 1, outer_rim_width * 2 + 1),
            dtype=np.uint8,
        ),
        iterations=1,
    )
    guidance = guidance_mask != 0
    board = board_mask != 0
    illumination_kernel = max(3, round(square_area**0.5 * 2.0))
    if illumination_kernel % 2 == 0:
        illumination_kernel += 1
    smoothing_regions = []
    for region in (guidance & board, guidance & ~board):
        weights = region.astype(np.float32)
        smoothed_weights = cv2.GaussianBlur(
            weights,
            (illumination_kernel, illumination_kernel),
            0,
            borderType=cv2.BORDER_CONSTANT,
        )
        smoothing_regions.append(
            (region & (smoothed_weights > 1e-6), weights, smoothed_weights)
        )
    return _EdgeGeometry(
        crop=crop,
        grid_key=grid_key,
        guidance=guidance,
        board=board,
        outside=guidance & ~board,
        outer_halo=guidance & (eroded == 0),
        square_area=square_area,
        illumination_kernel=illumination_kernel,
        smoothing_regions=tuple(smoothing_regions),
        square_inner=square_inner,
        square_inner_labels=square_inner_labels,
        square_inner_pixels=square_inner_pixels,
    )


def _inner_square_polygons(
    frame_shape: tuple[int, int],
    camera_grid: FloatArray,
) -> list[np.ndarray]:
    height, width = frame_shape
    low = _SQUARE_SCORE_INSET
    high = 1.0 - _SQUARE_SCORE_INSET
    polygons = []
    for row in range(8):
        for column in range(8):
            top_left = camera_grid[row, column]
            top_right = camera_grid[row, column + 1]
            bottom_right = camera_grid[row + 1, column + 1]
            bottom_left = camera_grid[row + 1, column]

            def point(u: float, v: float) -> np.ndarray:
                return np.asarray(
                    (1 - u) * (1 - v) * top_left
                    + u * (1 - v) * top_right
                    + u * v * bottom_right
                    + (1 - u) * v * bottom_left,
                    dtype=np.float32,
                )

            polygon = np.rint(
                np.asarray(
                    [
                        point(low, low),
                        point(high, low),
                        point(high, high),
                        point(low, high),
                    ]
                )
            ).astype(np.int32)
            polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
            polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
            polygons.append(polygon)
    return polygons


def _guidance_region_polygon(
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
    return polygon


def _board_region_polygon(
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
    return polygon


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
