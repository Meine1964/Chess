from dataclasses import dataclass, replace
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


@dataclass(frozen=True)
class CodedPattern:
    image: GrayImage
    marker_corners: dict[int, FloatArray]


def gray_code_bit_count(length: int) -> int:
    if length <= 1:
        raise ValueError("Gray-code dimension must be greater than one.")
    return (length - 1).bit_length()


def coded_marker_pattern(
    width: int,
    height: int,
    *,
    intensity: int = 112,
) -> CodedPattern:
    if width <= 1 or height <= 1:
        raise ValueError("Coded-pattern dimensions must be greater than one.")
    if not 1 <= intensity <= 255:
        raise ValueError("Intensity must be between 1 and 255.")

    marker_size = max(40, min(width // 16, height // 9))
    quiet_zone = max(6, marker_size // 6)
    pitch = marker_size + 2 * quiet_zone
    columns = width // pitch
    rows = height // pitch
    if columns < 4 or rows < 3 or columns * rows > 100:
        raise ValueError(
            "Projector resolution cannot hold enough uniquely coded markers."
        )

    pattern = np.zeros((height, width), dtype=np.uint8)
    marker_corners: dict[int, FloatArray] = {}
    dictionary = cv2.aruco.getPredefinedDictionary(
        cv2.aruco.DICT_4X4_100
    )
    start_x = (width - columns * pitch) // 2
    start_y = (height - rows * pitch) // 2
    marker_id = 0
    for row in range(rows):
        for column in range(columns):
            tile_left = start_x + column * pitch
            tile_top = start_y + row * pitch
            marker_left = tile_left + quiet_zone
            marker_top = tile_top + quiet_zone
            pattern[
                tile_top : tile_top + pitch,
                tile_left : tile_left + pitch,
            ] = intensity
            marker = cv2.aruco.generateImageMarker(
                dictionary,
                marker_id,
                marker_size,
            )
            marker_slice = pattern[
                marker_top : marker_top + marker_size,
                marker_left : marker_left + marker_size,
            ]
            marker_slice[marker == 0] = 0
            marker_corners[marker_id] = np.asarray(
                [
                    [marker_left, marker_top],
                    [marker_left + marker_size - 1, marker_top],
                    [
                        marker_left + marker_size - 1,
                        marker_top + marker_size - 1,
                    ],
                    [marker_left, marker_top + marker_size - 1],
                ],
                dtype=np.float32,
            )
            marker_id += 1
    return CodedPattern(pattern, marker_corners)


def detect_coded_marker_correspondences(
    black: GrayImage,
    illuminated: GrayImage,
    pattern: CodedPattern,
) -> tuple[FloatArray, FloatArray, IntArray]:
    if black.shape != illuminated.shape or black.ndim != 2:
        raise ValueError(
            "Coded-pattern captures must be equally sized grayscale images."
        )
    difference = np.maximum(
        illuminated.astype(np.int16) - black.astype(np.int16),
        0,
    ).astype(np.float32)
    response = float(np.percentile(difference, 99))
    if response < 8:
        raise CalibrationError(
            "The coded pattern produced insufficient camera response."
        )
    normalized = np.clip(
        np.rint(difference * (255.0 / response)),
        0,
        255,
    ).astype(np.uint8)
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100),
        parameters,
    )
    detected_corners, detected_ids, _rejected = detector.detectMarkers(
        normalized
    )
    if detected_ids is None:
        raise CalibrationError("No uniquely coded projector markers were detected.")

    camera_points: list[FloatArray] = []
    projector_points: list[FloatArray] = []
    marker_ids: list[int] = []
    for corners, marker_id_value in zip(
        detected_corners,
        detected_ids.reshape(-1),
        strict=True,
    ):
        marker_id = int(marker_id_value)
        expected_corners = pattern.marker_corners.get(marker_id)
        if expected_corners is None:
            continue
        camera_points.append(
            np.asarray(corners, dtype=np.float32).reshape(4, 2)
        )
        projector_points.append(expected_corners)
        marker_ids.append(marker_id)
    if len(camera_points) < 4:
        raise CalibrationError(
            f"Only {len(camera_points)} coded markers were detected; "
            "at least four are required."
        )
    return (
        np.concatenate(camera_points, axis=0),
        np.concatenate(projector_points, axis=0),
        np.asarray(marker_ids, dtype=np.int32),
    )


def fit_visible_middle_lattice(
    black: GrayImage,
    rough_detection: BoardDetection,
) -> BoardDetection:
    if black.ndim != 2:
        raise ValueError("The board reference must be a grayscale image.")
    flags = (
        cv2.CALIB_CB_EXHAUSTIVE
        | cv2.CALIB_CB_ACCURACY
        | cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    observed_options: list[FloatArray] = []
    found, corners = cv2.findChessboardCornersSB(black, (7, 5), flags=flags)
    if found and corners is not None:
        observed_options.append(
            np.asarray(corners, dtype=np.float32).reshape(5, 7, 2)
        )
    found, corners = cv2.findChessboardCornersSB(black, (5, 7), flags=flags)
    if found and corners is not None:
        observed_options.append(
            np.asarray(corners, dtype=np.float32)
            .reshape(7, 5, 2)
            .transpose(1, 0, 2)
        )
    if not observed_options:
        raise CalibrationError(
            "The visible 7x5 middle-board intersections were not detected."
        )

    best_observed: FloatArray | None = None
    best_expected: FloatArray | None = None
    best_row_start = 0
    best_error = float("inf")
    for observed in observed_options:
        for row_start in range(1, 4):
            expected = rough_detection.grid_points[
                row_start : row_start + 5,
                1:8,
            ]
            for row_flip in (False, True):
                for column_flip in (False, True):
                    candidate = observed
                    if row_flip:
                        candidate = candidate[::-1, :, :]
                    if column_flip:
                        candidate = candidate[:, ::-1, :]
                    error = float(
                        np.sqrt(np.mean(np.square(candidate - expected)))
                    )
                    if error < best_error:
                        best_error = error
                        best_observed = candidate.copy()
                        best_expected = expected.copy()
                        best_row_start = row_start
    if best_observed is None or best_expected is None:
        raise CalibrationError("Could not orient the visible middle lattice.")
    rough_square_width = float(
        np.median(
            np.linalg.norm(
                np.diff(rough_detection.grid_points, axis=1),
                axis=2,
            )
        )
    )
    if best_error > rough_square_width * 0.6:
        raise CalibrationError(
            "The detected middle lattice does not agree with the rough "
            "board location."
        )

    model_points = np.asarray(
        [
            (column, row)
            for row in range(best_row_start, best_row_start + 5)
            for column in range(1, 8)
        ],
        dtype=np.float32,
    )
    camera_points = best_observed.reshape(-1, 2)
    homography, _mask = cv2.findHomography(
        model_points,
        camera_points,
        method=0,
    )
    if homography is None:
        raise CalibrationError("Could not fit the visible middle lattice.")
    outer_model = np.asarray(
        [[0, 0], [8, 0], [8, 8], [0, 8]],
        dtype=np.float32,
    )
    extrapolated_outer = cv2.perspectiveTransform(
        outer_model.reshape(-1, 1, 2),
        homography,
    ).reshape(-1, 2)
    outer_residuals = np.linalg.norm(
        extrapolated_outer - rough_detection.outer_corners,
        axis=1,
    )
    consistent_outer = outer_residuals <= rough_square_width * 0.2
    used_outer_anchors = int(np.count_nonzero(consistent_outer))
    interior_bias = float(
        np.linalg.norm(
            np.mean(
                best_observed - best_expected,
                axis=(0, 1),
            )
        )
    )
    if (
        used_outer_anchors >= 2
        and interior_bias <= rough_square_width * 0.02
    ):
        anchored_model = np.concatenate(
            [model_points, outer_model[consistent_outer]],
            axis=0,
        )
        anchored_camera = np.concatenate(
            [
                camera_points,
                rough_detection.outer_corners[consistent_outer],
            ],
            axis=0,
        )
        anchored_homography, _mask = cv2.findHomography(
            anchored_model,
            anchored_camera,
            method=0,
        )
        if anchored_homography is not None:
            homography = anchored_homography
        else:
            used_outer_anchors = 0
    else:
        used_outer_anchors = 0
    projected = cv2.perspectiveTransform(
        model_points.reshape(-1, 1, 2),
        homography,
    ).reshape(-1, 2)
    residual = float(
        np.sqrt(np.mean(np.square(projected - camera_points)))
    )
    if residual > 2.0:
        raise CalibrationError(
            f"The visible middle lattice fit has {residual:.2f} camera-pixel "
            "RMS error."
        )
    grid_model = np.asarray(
        [(column, row) for row in range(9) for column in range(9)],
        dtype=np.float32,
    )
    grid = np.asarray(
        cv2.perspectiveTransform(
            grid_model.reshape(-1, 1, 2),
            homography,
        ),
        dtype=np.float32,
    ).reshape(9, 9, 2)
    return BoardDetection(
        grid_points=grid,
        confidence=min(0.99, max(0.0, 1.0 - residual / 10.0)),
        reprojection_error=residual,
        method=(
            "visible 7x5 middle lattice"
            if used_outer_anchors < 2
            else "visible 7x5 middle lattice with bounded outer anchors"
        ),
        grid_alignment=rough_detection.grid_alignment,
    )


def fit_coded_pattern_mapping(
    black: GrayImage,
    illuminated: GrayImage,
    rough_detection: BoardDetection,
    pattern: CodedPattern,
    projector_size: tuple[int, int],
) -> tuple[BoardDetection, MappingResult]:
    refined_detection = fit_visible_middle_lattice(
        black,
        rough_detection,
    )
    camera_points, projector_points, marker_ids = (
        detect_coded_marker_correspondences(
            black,
            illuminated,
            pattern,
        )
    )
    board_polygon = np.ascontiguousarray(
        refined_detection.outer_corners,
        dtype=np.float32,
    )
    marker_count = marker_ids.size
    marker_camera_points = camera_points.reshape(marker_count, 4, 2)
    inside_markers = np.asarray(
        [
            cv2.pointPolygonTest(
                board_polygon,
                tuple(np.mean(corners, axis=0)),
                False,
            )
            >= 0
            for corners in marker_camera_points
        ],
        dtype=bool,
    )
    camera_points = marker_camera_points[inside_markers].reshape(-1, 2)
    projector_points = projector_points.reshape(marker_count, 4, 2)[
        inside_markers
    ].reshape(-1, 2)
    inside_marker_count = int(np.count_nonzero(inside_markers))
    if inside_marker_count < 4:
        raise CalibrationError(
            f"Only {inside_marker_count} coded markers were visible on the "
            "rough board plane."
        )

    homography, inlier_mask = cv2.findHomography(
        camera_points,
        projector_points,
        cv2.RANSAC,
        3.0,
    )
    if homography is None or inlier_mask is None:
        raise CalibrationError(
            "Could not fit the coded camera-to-projector plane."
        )
    inliers = np.asarray(inlier_mask, dtype=bool).reshape(-1)
    inlier_ratio = float(np.mean(inliers))
    if np.count_nonzero(inliers) < 16 or inlier_ratio < 0.6:
        raise CalibrationError(
            f"Coded-pattern plane support is unreliable: "
            f"{np.count_nonzero(inliers)} corners, "
            f"{inlier_ratio:.0%} inliers."
        )

    predicted = cv2.perspectiveTransform(
        camera_points.reshape(-1, 1, 2),
        homography,
    ).reshape(-1, 2)
    residuals = np.linalg.norm(predicted - projector_points, axis=1)
    inlier_residuals = residuals[inliers]
    rms_error = float(np.sqrt(np.mean(np.square(inlier_residuals))))
    maximum_error = float(np.max(inlier_residuals))
    if rms_error > 2.5:
        raise CalibrationError(
            f"Coded-pattern plane fit has {rms_error:.2f} projector-pixel "
            "RMS error."
        )

    camera_to_board = cv2.getPerspectiveTransform(
        refined_detection.outer_corners.astype(np.float32),
        np.asarray([[0, 0], [8, 0], [8, 8], [0, 8]], dtype=np.float32),
    )
    inlier_camera_points = camera_points[inliers]
    board_points = cv2.perspectiveTransform(
        inlier_camera_points.reshape(-1, 1, 2),
        camera_to_board,
    ).reshape(-1, 2)
    quadrant_counts = [
        int(
            np.count_nonzero(
                np.logical_and(
                    board_points[:, 0] < 4
                    if column == 0
                    else board_points[:, 0] >= 4,
                    board_points[:, 1] < 4
                    if row == 0
                    else board_points[:, 1] >= 4,
                )
            )
        )
        for row in range(2)
        for column in range(2)
    ]
    if min(quadrant_counts) < 2:
        raise CalibrationError(
            "Coded-pattern plane support does not cover every board quadrant."
        )

    typed_homography = np.asarray(homography, dtype=np.float32)
    projector_grid = np.asarray(
        cv2.perspectiveTransform(
            refined_detection.grid_points.reshape(-1, 1, 2),
            typed_homography,
        ),
        dtype=np.float32,
    ).reshape(9, 9, 2)
    _validate_projector_grid(
        projector_grid,
        projector_size[0],
        projector_size[1],
    )
    mapping = MappingResult(
        camera_grid=refined_detection.grid_points.copy(),
        projector_grid=projector_grid,
        camera_to_projector_homography=typed_homography,
        coverage_ratio=inside_marker_count / max(marker_count, 1),
        decoded_ratio=inside_marker_count / max(
            len(pattern.marker_corners),
            1,
        ),
        inlier_ratio=inlier_ratio,
        rms_error=rms_error,
        maximum_error=maximum_error,
    )
    return refined_detection, mapping


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

    square_coverages = np.empty((8, 8), dtype=np.float32)
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
            square_coverages[row, column] = square_coverage

    quadrant_coverages = [
        float(np.mean(square_coverages[row : row + 4, column : column + 4]))
        for row in (0, 4)
        for column in (0, 4)
    ]
    if min(quadrant_coverages) < 0.35:
        weakest_quadrant = int(np.argmin(quadrant_coverages)) + 1
        raise CalibrationError(
            f"Projector response is insufficient in board quadrant "
            f"{weakest_quadrant}: "
            f"{min(quadrant_coverages):.0%} average coverage."
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
    if valid_count < 500:
        raise CalibrationError(
            f"Only {valid_count} camera pixels produced valid projector "
            "coordinates."
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

    rms_error = float(np.sqrt(np.mean(np.square(inlier_residuals))))
    maximum_error = float(np.max(inlier_residuals))

    all_camera_points = np.column_stack((camera_x, camera_y)).astype(
        np.float32
    )
    all_projector_points = np.column_stack(
        (
            projector_x[camera_y, camera_x],
            projector_y[camera_y, camera_x],
        )
    ).astype(np.float32)
    all_predicted = cv2.perspectiveTransform(
        all_camera_points.reshape(-1, 1, 2),
        homography,
    ).reshape(-1, 2)
    plane_inliers = (
        np.linalg.norm(all_predicted - all_projector_points, axis=1) <= 2.5
    )
    plane_valid = np.zeros_like(valid)
    plane_valid[camera_y[plane_inliers], camera_x[plane_inliers]] = True
    plane_inlier_count = int(np.count_nonzero(plane_inliers))
    inlier_ratio = plane_inlier_count / valid_count
    if inlier_ratio < 0.7 or rms_error > 2.0:
        raise CalibrationError(
            f"Projector mapping is unreliable: {inlier_ratio:.0%} inliers, "
            f"{rms_error:.2f} px RMS error."
        )
    _validate_distributed_plane_support(detection, plane_valid)

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
        plane_valid,
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


def _validate_distributed_plane_support(
    detection: BoardDetection,
    plane_valid: BoolArray,
) -> None:
    camera_y, camera_x = np.nonzero(plane_valid)
    count = camera_x.size
    if count < 500:
        raise CalibrationError(
            "Fewer than 500 decoded pixels are consistent with the board "
            "plane."
        )

    camera_points = np.column_stack((camera_x, camera_y)).astype(np.float32)
    board_corners = np.asarray(
        [[0, 0], [8, 0], [8, 8], [0, 8]],
        dtype=np.float32,
    )
    camera_to_board = cv2.getPerspectiveTransform(
        detection.outer_corners.astype(np.float32),
        board_corners,
    )
    board_points = cv2.perspectiveTransform(
        camera_points.reshape(-1, 1, 2),
        camera_to_board,
    ).reshape(-1, 2)
    board_x = board_points[:, 0]
    board_y = board_points[:, 1]
    inside = np.logical_and.reduce(
        (
            board_x >= 0,
            board_x <= 8,
            board_y >= 0,
            board_y <= 8,
        )
    )
    board_points = board_points[inside]
    if board_points.shape[0] < 500:
        raise CalibrationError(
            "Insufficient decoded board-plane support lies inside the board."
        )

    board_x = board_points[:, 0]
    board_y = board_points[:, 1]
    minimum_quadrant_count = max(20, round(board_points.shape[0] * 0.03))
    quadrant_counts = [
        int(
            np.count_nonzero(
                np.logical_and(
                    board_x < 4 if column == 0 else board_x >= 4,
                    board_y < 4 if row == 0 else board_y >= 4,
                )
            )
        )
        for row in range(2)
        for column in range(2)
    ]
    horizontal_span = float(
        np.percentile(board_x, 99) - np.percentile(board_x, 1)
    )
    vertical_span = float(
        np.percentile(board_y, 99) - np.percentile(board_y, 1)
    )
    hull_area = abs(
        float(
            cv2.contourArea(
                cv2.convexHull(board_points.astype(np.float32))
            )
        )
    )
    if (
        min(quadrant_counts) < minimum_quadrant_count
        or horizontal_span < 3.5
        or vertical_span < 3.5
        or hull_area < 12.0
    ):
        raise CalibrationError(
            "Decoded board-plane support is too concentrated for safe "
            "extrapolation."
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


def detect_board_orientation(
    gray: GrayImage,
    camera_grid: FloatArray,
    *,
    minimum_contrast: float = 3.0,
) -> int:
    if gray.ndim != 2:
        raise ValueError("Board orientation requires a grayscale image.")
    if camera_grid.shape != (9, 9, 2):
        raise ValueError("The camera grid must have shape (9, 9, 2).")
    if minimum_contrast <= 0:
        raise ValueError("Minimum orientation contrast must be positive.")

    parity_brightness: tuple[list[float], list[float]] = ([], [])
    height, width = gray.shape
    for row in range(8):
        for column in range(8):
            corners = np.asarray(
                [
                    camera_grid[row, column],
                    camera_grid[row, column + 1],
                    camera_grid[row + 1, column + 1],
                    camera_grid[row + 1, column],
                ],
                dtype=np.float32,
            )
            center = np.mean(corners, axis=0)
            inset = center + (corners - center) * 0.55
            left = max(0, int(np.floor(np.min(inset[:, 0]))))
            right = min(width, int(np.ceil(np.max(inset[:, 0]))) + 1)
            top = max(0, int(np.floor(np.min(inset[:, 1]))))
            bottom = min(height, int(np.ceil(np.max(inset[:, 1]))) + 1)
            if left >= right or top >= bottom:
                raise CalibrationError(
                    "A board square has no measurable orientation area."
                )
            local_polygon = np.rint(
                inset - np.asarray([left, top], dtype=np.float32)
            ).astype(np.int32)
            mask = np.zeros((bottom - top, right - left), dtype=np.uint8)
            cv2.fillConvexPoly(mask, local_polygon, 1)
            values = gray[top:bottom, left:right][mask > 0]
            if values.size == 0:
                raise CalibrationError(
                    "A board square has no measurable orientation pixels."
                )
            parity_brightness[(row + column) % 2].append(
                float(np.median(values))
            )

    even_brightness = float(np.median(parity_brightness[0]))
    odd_brightness = float(np.median(parity_brightness[1]))
    if abs(even_brightness - odd_brightness) < minimum_contrast:
        raise CalibrationError(
            "The light and dark board squares do not have enough contrast "
            "to determine orientation."
        )
    return 0 if even_brightness > odd_brightness else 1


def remap_projector_mapping(
    mapping: MappingResult,
    current_camera_grid: FloatArray,
    projector_size: tuple[int, int],
) -> MappingResult:
    if current_camera_grid.shape != (9, 9, 2):
        raise ValueError("The current camera grid must have shape (9, 9, 2).")

    old_camera_points = mapping.camera_grid.reshape(-1, 2)
    old_global_points = np.asarray(
        cv2.perspectiveTransform(
            old_camera_points.reshape(-1, 1, 2),
            mapping.camera_to_projector_homography,
        ),
        dtype=np.float32,
    ).reshape(-1, 2)
    local_corrections = (
        mapping.projector_grid.reshape(-1, 2) - old_global_points
    )

    center = np.mean(old_camera_points, axis=0)
    scale = np.std(old_camera_points, axis=0)
    if np.any(scale < 1.0):
        raise CalibrationError(
            "The saved camera grid is too small to remap reliably."
        )

    normalized_old = (old_camera_points - center) / scale
    old_features = _quadratic_features(normalized_old)
    correction_coefficients, _residuals, _rank, _singular_values = (
        np.linalg.lstsq(
            old_features,
            local_corrections,
            rcond=None,
        )
    )

    current_points = current_camera_grid.reshape(-1, 2)
    current_global_points = np.asarray(
        cv2.perspectiveTransform(
            current_points.reshape(-1, 1, 2),
            mapping.camera_to_projector_homography,
        ),
        dtype=np.float32,
    ).reshape(-1, 2)
    normalized_current = (current_points - center) / scale
    predicted_corrections = (
        _quadratic_features(normalized_current) @ correction_coefficients
    )
    current_projector_grid = np.asarray(
        current_global_points + predicted_corrections,
        dtype=np.float32,
    ).reshape(9, 9, 2)
    _validate_projector_grid(
        current_projector_grid,
        projector_size[0],
        projector_size[1],
    )
    return replace(
        mapping,
        camera_grid=current_camera_grid.copy(),
        projector_grid=current_projector_grid,
    )


def quick_realign_mapping(
    mapping: MappingResult,
    board_coordinates: FloatArray,
    measured_camera_points: FloatArray,
    projector_size: tuple[int, int],
) -> MappingResult:
    if (
        board_coordinates.ndim != 2
        or board_coordinates.shape[1] != 2
        or board_coordinates.shape[0] < 4
    ):
        raise ValueError(
            "At least four two-dimensional board coordinates are required."
        )
    if measured_camera_points.shape != board_coordinates.shape:
        raise ValueError(
            "Measured camera points must match the board coordinates."
        )

    old_global_grid = np.asarray(
        cv2.perspectiveTransform(
            mapping.camera_grid.reshape(-1, 1, 2),
            mapping.camera_to_projector_homography,
        ),
        dtype=np.float32,
    ).reshape(9, 9, 2)
    local_corrections = mapping.projector_grid - old_global_grid
    commanded_points = np.asarray(
        [
            interpolate_grid_point(
                mapping.projector_grid,
                float(board_x),
                float(board_y),
            )
            for board_x, board_y in board_coordinates
        ],
        dtype=np.float32,
    )
    fitting_corrections = np.asarray(
        [
            interpolate_grid_point(
                local_corrections,
                float(board_x),
                float(board_y),
            )
            for board_x, board_y in board_coordinates
        ],
        dtype=np.float32,
    )
    global_commands = commanded_points - fitting_corrections
    homography, _mask = cv2.findHomography(
        measured_camera_points,
        global_commands,
        method=0,
    )
    if homography is None or not np.all(np.isfinite(homography)):
        raise CalibrationError(
            "Quick realignment could not fit a camera-to-projector mapping."
        )
    typed_homography = np.asarray(homography, dtype=np.float32)
    new_global_grid = np.asarray(
        cv2.perspectiveTransform(
            mapping.camera_grid.reshape(-1, 1, 2),
            typed_homography,
        ),
        dtype=np.float32,
    ).reshape(9, 9, 2)
    new_projector_grid = np.asarray(
        new_global_grid + local_corrections,
        dtype=np.float32,
    )
    _validate_projector_grid(
        new_projector_grid,
        projector_size[0],
        projector_size[1],
    )

    predicted_commands = np.asarray(
        cv2.perspectiveTransform(
            measured_camera_points.reshape(-1, 1, 2),
            typed_homography,
        ),
        dtype=np.float32,
    ).reshape(-1, 2)
    predicted_commands += fitting_corrections
    errors = np.linalg.norm(predicted_commands - commanded_points, axis=1)
    return replace(
        mapping,
        projector_grid=new_projector_grid,
        camera_to_projector_homography=typed_homography,
        inlier_ratio=1.0,
        rms_error=float(np.sqrt(np.mean(np.square(errors)))),
        maximum_error=float(np.max(errors)),
    )


def _quadratic_features(points: FloatArray) -> FloatArray:
    x_values = points[:, 0]
    y_values = points[:, 1]
    return np.column_stack(
        (
            np.ones(points.shape[0], dtype=np.float32),
            x_values,
            y_values,
            x_values * x_values,
            x_values * y_values,
            y_values * y_values,
        )
    ).astype(np.float32)


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
