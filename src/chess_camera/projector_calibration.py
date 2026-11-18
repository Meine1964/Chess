from dataclasses import dataclass
from typing import Literal, TypeAlias

import cv2
import numpy as np
from numpy.typing import NDArray

from .board_detector import BoardDetection, FloatArray, ImageArray

GrayImage: TypeAlias = NDArray[np.uint8]
IntArray: TypeAlias = NDArray[np.int32]
BoolArray: TypeAlias = NDArray[np.bool_]
Axis = Literal["x", "y"]


class CalibrationError(RuntimeError):
    pass


@dataclass(frozen=True)
class CoverageResult:
    board_mask: BoolArray
    response_mask: BoolArray
    coverage_ratio: float
    saturation_ratio: float


@dataclass(frozen=True)
class MappingResult:
    camera_grid: FloatArray
    projector_grid: FloatArray
    camera_to_projector_homography: FloatArray
    coverage_ratio: float
    decoded_ratio: float
    inlier_ratio: float
    rms_error: float
    maximum_error: float


@dataclass(frozen=True)
class ValidationResult:
    measured_points: FloatArray
    expected_points: FloatArray
    errors: FloatArray
    mean_error: float
    rms_error: float
    maximum_error: float
    maximum_square_fraction: float


def gray_code_bit_count(length: int) -> int:
    if length <= 1:
        raise ValueError("Gray-code dimension must be greater than one.")
    return (length - 1).bit_length()


def gray_code_pattern(
    width: int,
    height: int,
    axis: Axis,
    bit_index: int,
    *,
    inverse: bool = False,
    intensity: int = 180,
) -> GrayImage:
    if width <= 1 or height <= 1:
        raise ValueError("Pattern dimensions must be greater than one.")
    if axis not in ("x", "y"):
        raise ValueError('Pattern axis must be "x" or "y".')
    if not 1 <= intensity <= 255:
        raise ValueError("Pattern intensity must be between 1 and 255.")

    length = width if axis == "x" else height
    bit_count = gray_code_bit_count(length)
    if not 0 <= bit_index < bit_count:
        raise ValueError(
            f"Bit index must be between 0 and {bit_count - 1}."
        )

    coordinates = np.arange(length, dtype=np.int32)
    gray_values = np.bitwise_xor(coordinates, coordinates >> 1)
    shift = bit_count - bit_index - 1
    active = ((gray_values >> shift) & 1).astype(bool)
    if inverse:
        active = np.logical_not(active)

    line = np.where(active, intensity, 0).astype(np.uint8)
    if axis == "x":
        return np.broadcast_to(line, (height, width)).copy()
    return np.broadcast_to(line[:, np.newaxis], (height, width)).copy()


def board_mask(
    image_shape: tuple[int, int],
    detection: BoardDetection,
    *,
    erosion_pixels: int = 2,
) -> BoolArray:
    height, width = image_shape
    if height <= 0 or width <= 0:
        raise ValueError("Image dimensions must be positive.")
    if erosion_pixels < 0:
        raise ValueError("Erosion must not be negative.")

    mask = np.zeros((height, width), dtype=np.uint8)
    corners = np.ascontiguousarray(
        np.rint(detection.outer_corners),
        dtype=np.int32,
    )
    cv2.fillConvexPoly(mask, corners, 255)
    if erosion_pixels:
        kernel_size = 2 * erosion_pixels + 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        mask = cv2.erode(mask, kernel)
    return mask > 0


def measure_coverage(
    black: GrayImage,
    illuminated: GrayImage,
    detection: BoardDetection,
    *,
    minimum_response: int = 8,
    maximum_saturation_ratio: float = 0.2,
) -> CoverageResult:
    if black.shape != illuminated.shape or black.ndim != 2:
        raise ValueError("Coverage images must be equally sized grayscale images.")
    if minimum_response < 1:
        raise ValueError("Minimum response must be positive.")

    mask = board_mask(black.shape, detection)
    response = illuminated.astype(np.int16) - black.astype(np.int16)
    response_mask = np.logical_and(mask, response >= minimum_response)
    board_pixels = int(np.count_nonzero(mask))
    if board_pixels == 0:
        raise CalibrationError("The detected board mask is empty.")

    coverage_ratio = float(np.count_nonzero(response_mask) / board_pixels)
    saturation_ratio = float(
        np.count_nonzero(np.logical_and(mask, illuminated >= 250))
        / board_pixels
    )
    if coverage_ratio < 0.7:
        raise CalibrationError(
            f"The projector illuminates only {coverage_ratio:.0%} of the board."
        )
    if saturation_ratio > maximum_saturation_ratio:
        raise CalibrationError(
            f"{saturation_ratio:.0%} of the board is saturated."
        )

    for row in range(8):
        for column in range(8):
            polygon = np.ascontiguousarray(
                np.rint(
                    detection.grid_points[
                        [row, row, row + 1, row + 1],
                        [column, column + 1, column + 1, column],
                    ]
                ),
                dtype=np.int32,
            )
            square_mask = np.zeros_like(black, dtype=np.uint8)
            cv2.fillConvexPoly(square_mask, polygon, 255)
            square_pixels = np.logical_and(mask, square_mask > 0)
            count = int(np.count_nonzero(square_pixels))
            if count == 0:
                raise CalibrationError(
                    f"Board square ({column}, {row}) has no measurable area."
                )
            square_coverage = float(
                np.count_nonzero(
                    np.logical_and(response_mask, square_pixels)
                )
                / count
            )
            if square_coverage < 0.45:
                raise CalibrationError(
                    f"Projector response is insufficient on board square "
                    f"({column}, {row}): {square_coverage:.0%} coverage."
                )

    return CoverageResult(
        board_mask=mask,
        response_mask=response_mask,
        coverage_ratio=coverage_ratio,
        saturation_ratio=saturation_ratio,
    )


def decode_gray_code(
    horizontal_pairs: list[tuple[GrayImage, GrayImage]],
    vertical_pairs: list[tuple[GrayImage, GrayImage]],
    usable_mask: BoolArray,
    *,
    minimum_difference: int = 8,
    horizontal_omitted_bits: int = 0,
    vertical_omitted_bits: int = 0,
) -> tuple[IntArray, IntArray, BoolArray]:
    if not horizontal_pairs or not vertical_pairs:
        raise ValueError("Both horizontal and vertical Gray-code pairs are required.")
    if usable_mask.ndim != 2:
        raise ValueError("The usable mask must be two-dimensional.")
    if minimum_difference < 1:
        raise ValueError("Minimum difference must be positive.")
    if horizontal_omitted_bits < 0 or vertical_omitted_bits < 0:
        raise ValueError("Omitted Gray-code bit counts must not be negative.")

    decoded: list[IntArray] = []
    valid = usable_mask.copy()
    for pairs, omitted_bits in (
        (horizontal_pairs, horizontal_omitted_bits),
        (vertical_pairs, vertical_omitted_bits),
    ):
        gray_value = np.zeros(usable_mask.shape, dtype=np.int32)
        for normal, inverse in pairs:
            if normal.shape != usable_mask.shape or inverse.shape != usable_mask.shape:
                raise ValueError("All Gray-code images must match the usable mask.")
            difference = normal.astype(np.int16) - inverse.astype(np.int16)
            valid = np.logical_and(valid, np.abs(difference) >= minimum_difference)
            gray_value = (gray_value << 1) | (difference > 0).astype(np.int32)

        binary_value = gray_value.copy()
        shift = 1
        while shift < 32:
            binary_value ^= binary_value >> shift
            shift <<= 1
        if omitted_bits:
            scale = 1 << omitted_bits
            binary_value = binary_value * scale + scale // 2
        decoded.append(binary_value)

    return decoded[0], decoded[1], valid


def fit_projector_mapping(
    detection: BoardDetection,
    projector_x: IntArray,
    projector_y: IntArray,
    valid_mask: BoolArray,
    projector_size: tuple[int, int],
    coverage_ratio: float,
) -> MappingResult:
    if (
        projector_x.shape != projector_y.shape
        or projector_x.shape != valid_mask.shape
    ):
        raise ValueError("Decoded coordinate arrays and mask must have equal shapes.")
    projector_width, projector_height = projector_size
    valid = np.logical_and(
        valid_mask,
        np.logical_and(
            np.logical_and(projector_x >= 0, projector_x < projector_width),
            np.logical_and(projector_y >= 0, projector_y < projector_height),
        ),
    )
    board_area = abs(
        float(
            cv2.contourArea(
                np.ascontiguousarray(detection.outer_corners, dtype=np.float32)
            )
        )
    )
    valid_count = int(np.count_nonzero(valid))
    decoded_ratio = valid_count / max(board_area, 1.0)
    if valid_count < 500 or decoded_ratio < 0.25:
        raise CalibrationError(
            f"Only {decoded_ratio:.0%} of the board produced valid "
            "projector coordinates."
        )

    camera_y, camera_x = np.nonzero(valid)
    stride = max(1, valid_count // 50_000)
    sample_indices = np.arange(0, valid_count, stride)
    camera_points = np.column_stack(
        (camera_x[sample_indices], camera_y[sample_indices])
    ).astype(np.float32)
    projector_points = np.column_stack(
        (
            projector_x[camera_y[sample_indices], camera_x[sample_indices]],
            projector_y[camera_y[sample_indices], camera_x[sample_indices]],
        )
    ).astype(np.float32)

    homography, inlier_mask = cv2.findHomography(
        camera_points,
        projector_points,
        cv2.RANSAC,
        2.5,
    )
    if homography is None or inlier_mask is None:
        raise CalibrationError("Could not fit a camera-to-projector homography.")

    predicted = cv2.perspectiveTransform(
        camera_points.reshape(-1, 1, 2),
        homography,
    ).reshape(-1, 2)
    residuals = np.linalg.norm(predicted - projector_points, axis=1)
    inliers = np.asarray(inlier_mask, dtype=bool).reshape(-1)
    inlier_residuals = residuals[inliers]
    if inlier_residuals.size == 0:
        raise CalibrationError("The projector mapping has no geometric inliers.")

    inlier_ratio = float(np.mean(inliers))
    rms_error = float(np.sqrt(np.mean(np.square(inlier_residuals))))
    maximum_error = float(np.max(inlier_residuals))
    if inlier_ratio < 0.7 or rms_error > 2.0:
        raise CalibrationError(
            f"Projector mapping is unreliable: {inlier_ratio:.0%} inliers, "
            f"{rms_error:.2f} px RMS error."
        )

    typed_homography = np.asarray(homography, dtype=np.float32)
    global_grid = np.asarray(
        cv2.perspectiveTransform(
            detection.grid_points.reshape(-1, 1, 2),
            typed_homography,
        ),
        dtype=np.float32,
    ).reshape(9, 9, 2)
    projector_grid = _apply_local_grid_corrections(
        detection.grid_points,
        global_grid,
        projector_x,
        projector_y,
        valid,
        typed_homography,
    )
    _validate_projector_grid(
        projector_grid,
        projector_width,
        projector_height,
    )

    return MappingResult(
        camera_grid=detection.grid_points.copy(),
        projector_grid=projector_grid,
        camera_to_projector_homography=typed_homography,
        coverage_ratio=coverage_ratio,
        decoded_ratio=decoded_ratio,
        inlier_ratio=inlier_ratio,
        rms_error=rms_error,
        maximum_error=maximum_error,
    )


def _apply_local_grid_corrections(
    camera_grid: FloatArray,
    global_grid: FloatArray,
    projector_x: IntArray,
    projector_y: IntArray,
    valid: BoolArray,
    homography: FloatArray,
) -> FloatArray:
    height, width = valid.shape
    corrected = global_grid.copy()
    for row in range(9):
        for column in range(9):
            camera_point = camera_grid[row, column]
            x = int(round(float(camera_point[0])))
            y = int(round(float(camera_point[1])))
            radius = 12
            left = max(0, x - radius)
            right = min(width, x + radius + 1)
            top = max(0, y - radius)
            bottom = min(height, y + radius + 1)
            local_valid = valid[top:bottom, left:right]
            local_y, local_x = np.nonzero(local_valid)
            if local_x.size < 20:
                continue
            absolute_x = local_x + left
            absolute_y = local_y + top
            camera_samples = np.column_stack(
                (absolute_x, absolute_y)
            ).astype(np.float32)
            global_samples = cv2.perspectiveTransform(
                camera_samples.reshape(-1, 1, 2),
                homography,
            ).reshape(-1, 2)
            measured_samples = np.column_stack(
                (
                    projector_x[absolute_y, absolute_x],
                    projector_y[absolute_y, absolute_x],
                )
            ).astype(np.float32)
            offsets = measured_samples - global_samples
            median_offset = np.median(offsets, axis=0)
            corrected[row, column] += median_offset.astype(np.float32)
    return corrected


def _validate_projector_grid(
    grid: FloatArray,
    width: int,
    height: int,
) -> None:
    if grid.shape != (9, 9, 2) or not np.all(np.isfinite(grid)):
        raise CalibrationError("The projector grid is incomplete or non-finite.")
    if (
        np.min(grid[:, :, 0]) < 0
        or np.max(grid[:, :, 0]) > width - 1
        or np.min(grid[:, :, 1]) < 0
        or np.max(grid[:, :, 1]) > height - 1
    ):
        raise CalibrationError(
            "The calculated board grid extends beyond the projector display."
        )

    signed_areas: list[float] = []
    for row in range(8):
        for column in range(8):
            polygon = np.asarray(
                [
                    grid[row, column],
                    grid[row, column + 1],
                    grid[row + 1, column + 1],
                    grid[row + 1, column],
                ],
                dtype=np.float32,
            )
            signed_areas.append(float(cv2.contourArea(polygon, oriented=True)))
    areas = np.asarray(signed_areas)
    if np.min(np.abs(areas)) < 4.0 or not (
        np.all(areas > 0) or np.all(areas < 0)
    ):
        raise CalibrationError("The calculated projector grid is folded or invalid.")


def square_polygons(
    projector_grid: FloatArray,
    *,
    inset_fraction: float = 0.05,
) -> list[FloatArray]:
    if projector_grid.shape != (9, 9, 2):
        raise ValueError("The projector grid must have shape (9, 9, 2).")
    if not 0 <= inset_fraction < 0.5:
        raise ValueError("Inset fraction must be between 0 and 0.5.")

    polygons: list[FloatArray] = []
    low = inset_fraction
    high = 1.0 - inset_fraction
    for row in range(8):
        for column in range(8):
            corners = np.asarray(
                [
                    projector_grid[row, column],
                    projector_grid[row, column + 1],
                    projector_grid[row + 1, column + 1],
                    projector_grid[row + 1, column],
                ],
                dtype=np.float32,
            )
            polygons.append(
                np.asarray(
                    [
                        _bilinear(corners, low, low),
                        _bilinear(corners, high, low),
                        _bilinear(corners, high, high),
                        _bilinear(corners, low, high),
                    ],
                    dtype=np.float32,
                )
            )
    return polygons


def interpolate_grid_point(grid: FloatArray, x: float, y: float) -> FloatArray:
    if grid.shape != (9, 9, 2):
        raise ValueError("Grid must have shape (9, 9, 2).")
    if not 0 <= x <= 8 or not 0 <= y <= 8:
        raise ValueError("Board coordinates must be between 0 and 8.")
    column = min(int(x), 7)
    row = min(int(y), 7)
    local_x = x - column
    local_y = y - row
    corners = np.asarray(
        [
            grid[row, column],
            grid[row, column + 1],
            grid[row + 1, column + 1],
            grid[row + 1, column],
        ],
        dtype=np.float32,
    )
    return _bilinear(corners, local_x, local_y)


def refine_projector_mesh(
    projector_grid: FloatArray,
    camera_to_projector_homography: FloatArray,
    validation_board_points: FloatArray,
    expected_camera_points: FloatArray,
    measured_camera_points: FloatArray,
    projector_size: tuple[int, int],
) -> FloatArray:
    if projector_grid.shape != (9, 9, 2):
        raise ValueError("The projector grid must have shape (9, 9, 2).")
    if camera_to_projector_homography.shape != (3, 3):
        raise ValueError("The camera-to-projector homography must be 3x3.")
    if (
        validation_board_points.shape != expected_camera_points.shape
        or expected_camera_points.shape != measured_camera_points.shape
        or measured_camera_points.ndim != 2
        or measured_camera_points.shape[1] != 2
        or measured_camera_points.shape[0] != 9
    ):
        raise ValueError(
            "Exactly nine board, expected, and measured validation points "
            "are required."
        )

    expected_projector = np.asarray(
        cv2.perspectiveTransform(
            expected_camera_points.reshape(-1, 1, 2),
            camera_to_projector_homography,
        ),
        dtype=np.float32,
    ).reshape(-1, 2)
    measured_projector = np.asarray(
        cv2.perspectiveTransform(
            measured_camera_points.reshape(-1, 1, 2),
            camera_to_projector_homography,
        ),
        dtype=np.float32,
    ).reshape(-1, 2)
    corrections = expected_projector - measured_projector

    x_nodes = np.unique(validation_board_points[:, 0])
    y_nodes = np.unique(validation_board_points[:, 1])
    if x_nodes.size != 3 or y_nodes.size != 3:
        raise ValueError("Validation points must form a 3x3 board-coordinate grid.")

    correction_grid = np.empty((3, 3, 2), dtype=np.float32)
    for row, y_value in enumerate(y_nodes):
        for column, x_value in enumerate(x_nodes):
            matches = np.logical_and(
                np.isclose(validation_board_points[:, 0], x_value),
                np.isclose(validation_board_points[:, 1], y_value),
            )
            matching_corrections = corrections[matches]
            if matching_corrections.shape != (1, 2):
                raise ValueError(
                    "Every 3x3 validation coordinate must occur exactly once."
                )
            correction_grid[row, column] = matching_corrections[0]

    refined_grid = projector_grid.copy()
    for row in range(9):
        for column in range(9):
            for component in range(2):
                values_at_y = np.asarray(
                    [
                        np.interp(
                            float(column),
                            x_nodes,
                            correction_grid[node_row, :, component],
                        )
                        for node_row in range(3)
                    ],
                    dtype=np.float32,
                )
                refined_grid[row, column, component] = (
                    projector_grid[row, column, component]
                    + np.interp(float(row), y_nodes, values_at_y)
                )

    _validate_projector_grid(
        refined_grid,
        projector_size[0],
        projector_size[1],
    )
    return refined_grid


def _bilinear(corners: FloatArray, x: float, y: float) -> FloatArray:
    top = corners[0] * (1.0 - x) + corners[1] * x
    bottom = corners[3] * (1.0 - x) + corners[2] * x
    return np.asarray(top * (1.0 - y) + bottom * y, dtype=np.float32)


def locate_validation_target(
    black: GrayImage,
    illuminated: GrayImage,
    expected_point: FloatArray,
    search_radius: int,
    *,
    minimum_response: int = 8,
) -> FloatArray:
    if black.shape != illuminated.shape or black.ndim != 2:
        raise ValueError("Validation images must be equally sized grayscale images.")
    if search_radius < 2:
        raise ValueError("Search radius must be at least two pixels.")

    height, width = black.shape
    expected_x = int(round(float(expected_point[0])))
    expected_y = int(round(float(expected_point[1])))
    left = max(0, expected_x - search_radius)
    right = min(width, expected_x + search_radius + 1)
    top = max(0, expected_y - search_radius)
    bottom = min(height, expected_y + search_radius + 1)
    difference = (
        illuminated[top:bottom, left:right].astype(np.int16)
        - black[top:bottom, left:right].astype(np.int16)
    )
    baseline = float(np.median(difference))
    weights = np.maximum(
        difference.astype(np.float32) - baseline - minimum_response,
        0.0,
    )
    if np.count_nonzero(weights) < 5 or float(np.sum(weights)) <= 0:
        raise CalibrationError("A projected validation target was not detected.")

    local_y, local_x = np.indices(weights.shape, dtype=np.float32)
    total = float(np.sum(weights))
    measured_x = left + float(np.sum(local_x * weights) / total)
    measured_y = top + float(np.sum(local_y * weights) / total)
    return np.asarray([measured_x, measured_y], dtype=np.float32)


def validation_result(
    measured_points: FloatArray,
    expected_points: FloatArray,
    square_widths: FloatArray,
) -> ValidationResult:
    if (
        measured_points.shape != expected_points.shape
        or measured_points.ndim != 2
        or measured_points.shape[1] != 2
    ):
        raise ValueError("Measured and expected points must have shape (N, 2).")
    if square_widths.shape != (measured_points.shape[0],):
        raise ValueError("One square width is required for every validation point.")
    if np.any(square_widths <= 0):
        raise ValueError("Square widths must be positive.")

    errors = np.linalg.norm(measured_points - expected_points, axis=1).astype(
        np.float32
    )
    fractions = errors / square_widths
    return ValidationResult(
        measured_points=measured_points.copy(),
        expected_points=expected_points.copy(),
        errors=errors,
        mean_error=float(np.mean(errors)),
        rms_error=float(np.sqrt(np.mean(np.square(errors)))),
        maximum_error=float(np.max(errors)),
        maximum_square_fraction=float(np.max(fractions)),
    )


def bgr_to_gray(frame: ImageArray) -> GrayImage:
    return np.asarray(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), dtype=np.uint8)
