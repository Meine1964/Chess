import json
from pathlib import Path

import cv2
import numpy as np

from src.chess_camera.board_detector import BoardDetector
from src.chess_camera.camera_app import _resolve_camera_index
from src.chess_camera.diagnostics import DiagnosticStore


def _perspective_board() -> tuple[np.ndarray, np.ndarray]:
    square_size = 80
    board_size = square_size * 8
    board = np.zeros((board_size, board_size, 3), dtype=np.uint8)
    for row in range(8):
        for column in range(8):
            color = 225 if (row + column) % 2 else 25
            top_left = (column * square_size, row * square_size)
            bottom_right = (
                (column + 1) * square_size,
                (row + 1) * square_size,
            )
            cv2.rectangle(board, top_left, bottom_right, (color,) * 3, -1)

    source_corners = np.array(
        [[0, 0], [board_size, 0], [board_size, board_size], [0, board_size]],
        dtype=np.float32,
    )
    destination_corners = np.array(
        [[310, 95], [1010, 145], [920, 650], [225, 590]],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(source_corners, destination_corners)
    image = cv2.warpPerspective(
        board,
        transform,
        (1280, 720),
        borderValue=(140, 140, 140),
    )
    return image, destination_corners


def test_detects_full_grid_from_inner_corners() -> None:
    image, expected_corners = _perspective_board()

    detection = BoardDetector().detect(image)

    assert detection is not None
    assert detection.grid_points.shape == (9, 9, 2)
    assert detection.confidence >= 0.8
    corner_error = np.linalg.norm(
        detection.outer_corners - expected_corners,
        axis=1,
    )
    assert float(np.max(corner_error)) < 8.0

    overlay = BoardDetector.draw_overlay(image, detection)

    assert overlay.shape == image.shape
    assert np.any(overlay != image)


def test_returns_none_when_no_board_is_present() -> None:
    image = np.full((720, 1280, 3), 140, dtype=np.uint8)

    assert BoardDetector().detect(image) is None


def test_resolves_saved_camera_before_default() -> None:
    assert (
        _resolve_camera_index(
            ["Logitech Webcam", "Integrated Webcam"],
            ["logitech-id", "integrated-id"],
            default_key="integrated-id",
            saved_key="logitech-id",
            selector=None,
        )
        == 0
    )


def test_resolves_camera_by_unique_name_fragment() -> None:
    assert (
        _resolve_camera_index(
            ["Logitech Webcam", "Integrated Webcam"],
            ["logitech-id", "integrated-id"],
            default_key="integrated-id",
            saved_key="",
            selector="logitech",
        )
        == 0
    )


def test_saves_diagnostics_and_enforces_retention(tmp_path: Path) -> None:
    image, _ = _perspective_board()
    detection = BoardDetector().detect(image)
    assert detection is not None
    overlay = BoardDetector.draw_overlay(image, detection)
    store = DiagnosticStore(tmp_path, max_captures=2)

    first_capture = store.save(
        image,
        overlay,
        detection,
        camera_name="Test Camera",
        camera_id="test-id",
    )
    store.save(image, None, None, "Test Camera", "test-id")
    latest_capture = store.save(
        image,
        overlay,
        detection,
        camera_name="Test Camera",
        camera_id="test-id",
    )

    assert not first_capture.exists()
    assert len(list(tmp_path.iterdir())) == 2
    assert (latest_capture / "original.png").is_file()
    assert (latest_capture / "overlay.png").is_file()
    metadata = json.loads(
        (latest_capture / "detection.json").read_text(encoding="utf-8")
    )
    assert metadata["camera"]["name"] == "Test Camera"
    assert metadata["detection"]["success"] is True
    assert len(metadata["detection"]["grid_points"]) == 9
