import json

import cv2
import numpy as np
import pytest

from src.chess_camera.board_detector import BoardDetection
from src.chess_camera.calibration_controller import CalibrationResult
from src.chess_camera.calibration_store import CalibrationStore
from src.chess_camera.projector_calibration import (
    CalibrationError,
    MappingResult,
    ValidationResult,
    decode_gray_code,
    fit_projector_mapping,
    gray_code_bit_count,
    gray_code_pattern,
    interpolate_grid_point,
    locate_validation_target,
    measure_coverage,
    refine_projector_mesh,
    square_polygons,
    validation_result,
)


def _detection(
    left: float = 10,
    top: float = 10,
    right: float = 90,
    bottom: float = 90,
) -> BoardDetection:
    x_values = np.linspace(left, right, 9, dtype=np.float32)
    y_values = np.linspace(top, bottom, 9, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    grid = np.stack((x_grid, y_grid), axis=2).astype(np.float32)
    return BoardDetection(
        grid_points=grid,
        confidence=1.0,
        reprojection_error=0.0,
        method="test",
        grid_alignment=1.0,
    )


def test_gray_code_patterns_decode_projector_coordinates() -> None:
    width = 13
    height = 9
    horizontal_pairs = [
        (
            gray_code_pattern(width, height, "x", bit),
            gray_code_pattern(width, height, "x", bit, inverse=True),
        )
        for bit in range(gray_code_bit_count(width))
    ]
    vertical_pairs = [
        (
            gray_code_pattern(width, height, "y", bit),
            gray_code_pattern(width, height, "y", bit, inverse=True),
        )
        for bit in range(gray_code_bit_count(height))
    ]

    decoded_x, decoded_y, valid = decode_gray_code(
        horizontal_pairs,
        vertical_pairs,
        np.ones((height, width), dtype=bool),
    )

    expected_x = np.broadcast_to(np.arange(width), (height, width))
    expected_y = np.broadcast_to(
        np.arange(height)[:, np.newaxis],
        (height, width),
    )
    assert np.array_equal(decoded_x, expected_x)
    assert np.array_equal(decoded_y, expected_y)
    assert np.all(valid)

    coarse_x, coarse_y, coarse_valid = decode_gray_code(
        horizontal_pairs[:-1],
        vertical_pairs[:-1],
        np.ones((height, width), dtype=bool),
        horizontal_omitted_bits=1,
        vertical_omitted_bits=1,
    )
    assert np.array_equal(coarse_x, (expected_x // 2) * 2 + 1)
    assert np.array_equal(coarse_y, (expected_y // 2) * 2 + 1)
    assert np.all(coarse_valid)


def test_coverage_requires_response_on_every_square() -> None:
    detection = _detection()
    black = np.full((100, 100), 20, dtype=np.uint8)
    illuminated = np.full((100, 100), 100, dtype=np.uint8)

    result = measure_coverage(black, illuminated, detection)

    assert result.coverage_ratio == pytest.approx(1.0)
    illuminated[10:20, 10:20] = black[10:20, 10:20]
    with pytest.raises(CalibrationError, match="board square"):
        measure_coverage(black, illuminated, detection)


def test_mapping_recovers_projector_grid_from_dense_measurements() -> None:
    detection = _detection(20, 20, 180, 180)
    height = 200
    width = 200
    camera_y, camera_x = np.indices((height, width), dtype=np.float32)
    camera_points = np.stack((camera_x, camera_y), axis=2)
    expected_homography = np.asarray(
        [
            [1.5, 0.08, 12.0],
            [0.03, 1.2, 8.0],
            [0.0002, 0.0001, 1.0],
        ],
        dtype=np.float32,
    )
    projector_points = cv2.perspectiveTransform(
        camera_points.reshape(-1, 1, 2),
        expected_homography,
    ).reshape(height, width, 2)
    projector_x = np.rint(projector_points[:, :, 0]).astype(np.int32)
    projector_y = np.rint(projector_points[:, :, 1]).astype(np.int32)
    valid = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(
        valid,
        np.rint(detection.outer_corners).astype(np.int32),
        1,
    )

    result = fit_projector_mapping(
        detection,
        projector_x,
        projector_y,
        valid.astype(bool),
        (400, 300),
        coverage_ratio=1.0,
    )
    expected_grid = cv2.perspectiveTransform(
        detection.grid_points.reshape(-1, 1, 2),
        expected_homography,
    ).reshape(9, 9, 2)

    assert result.inlier_ratio > 0.99
    assert result.rms_error < 0.6
    assert np.max(np.abs(result.projector_grid - expected_grid)) < 0.6


def test_square_polygons_build_64_inset_quadrilaterals() -> None:
    projector_grid = _detection(0, 0, 800, 800).grid_points

    polygons = square_polygons(projector_grid, inset_fraction=0.05)

    assert len(polygons) == 64
    assert polygons[0] == pytest.approx(
        np.asarray(
            [[5, 5], [95, 5], [95, 95], [5, 95]],
            dtype=np.float32,
        )
    )
    assert interpolate_grid_point(projector_grid, 4.5, 4.5) == pytest.approx(
        np.asarray([450, 450], dtype=np.float32)
    )


def test_refinement_applies_local_validation_corrections() -> None:
    camera_grid = _detection(20, 20, 180, 180).grid_points
    board_points = np.asarray(
        [
            [0.5, 0.5],
            [4.0, 0.5],
            [7.5, 0.5],
            [0.5, 4.0],
            [4.0, 4.0],
            [7.5, 4.0],
            [0.5, 7.5],
            [4.0, 7.5],
            [7.5, 7.5],
        ],
        dtype=np.float32,
    )
    expected_camera = np.asarray(
        [
            interpolate_grid_point(camera_grid, float(x), float(y))
            for x, y in board_points
        ],
        dtype=np.float32,
    )
    measured_camera = expected_camera + np.asarray([10, -5], dtype=np.float32)
    homography = np.asarray(
        [[1.5, 0, 12], [0, 1.2, 8], [0, 0, 1]],
        dtype=np.float32,
    )
    projector_grid = cv2.perspectiveTransform(
        camera_grid.reshape(-1, 1, 2),
        homography,
    ).reshape(9, 9, 2)

    refined = refine_projector_mesh(
        projector_grid,
        homography,
        board_points,
        expected_camera,
        measured_camera,
        (400, 300),
    )

    expected = projector_grid + np.asarray([-15, 6], dtype=np.float32)
    np.testing.assert_allclose(refined, expected, atol=0.01)


def test_validation_target_and_metrics_measure_camera_error() -> None:
    black = np.full((100, 100), 20, dtype=np.uint8)
    illuminated = black.copy()
    cv2.circle(illuminated, (52, 48), 4, 150, -1)
    expected = np.asarray([50, 50], dtype=np.float32)

    measured = locate_validation_target(
        black,
        illuminated,
        expected,
        search_radius=15,
    )
    result = validation_result(
        measured[np.newaxis, :],
        expected[np.newaxis, :],
        np.asarray([40], dtype=np.float32),
    )

    assert measured == pytest.approx(np.asarray([52, 48]), abs=0.1)
    assert result.maximum_error == pytest.approx(np.sqrt(8), abs=0.1)
    assert result.maximum_square_fraction < 0.1


def test_calibration_store_persists_mesh_and_validation(tmp_path) -> None:
    detection = _detection()
    errors = np.asarray([1.0], dtype=np.float32)
    result = CalibrationResult(
        board_detection=detection,
        mapping=MappingResult(
            camera_grid=detection.grid_points,
            projector_grid=detection.grid_points,
            camera_to_projector_homography=np.eye(3, dtype=np.float32),
            coverage_ratio=0.95,
            decoded_ratio=0.9,
            inlier_ratio=0.99,
            rms_error=0.4,
            maximum_error=1.2,
        ),
        validation=ValidationResult(
            measured_points=np.asarray([[51, 50]], dtype=np.float32),
            expected_points=np.asarray([[50, 50]], dtype=np.float32),
            errors=errors,
            mean_error=1.0,
            rms_error=1.0,
            maximum_error=1.0,
            maximum_square_fraction=0.025,
        ),
    )

    saved = CalibrationStore(tmp_path).save(
        result,
        camera_name="Camera",
        camera_id="camera-id",
        camera_resolution=(1920, 1080),
        projector_name="Projector",
        projector_resolution=(1280, 720),
    )
    metadata = json.loads(saved.read_text(encoding="utf-8"))

    assert metadata["projector"]["width"] == 1280
    np.testing.assert_allclose(
        np.asarray(metadata["mapping"]["projector_grid"]),
        detection.grid_points,
    )
    assert metadata["validation"]["maximum_square_fraction"] == 0.025

    loaded = CalibrationStore(tmp_path).load_matching(
        camera_id="camera-id",
        camera_resolution=(1920, 1080),
        projector_name="Projector",
        projector_resolution=(1280, 720),
    )
    assert loaded is not None
    np.testing.assert_allclose(
        loaded.mapping.projector_grid,
        detection.grid_points,
    )
    assert (
        CalibrationStore(tmp_path).load_matching(
            camera_id="different-camera",
            camera_resolution=(1920, 1080),
            projector_name="Projector",
            projector_resolution=(1280, 720),
        )
        is None
    )
