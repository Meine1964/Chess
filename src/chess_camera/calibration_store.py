import json
from datetime import datetime
from pathlib import Path

import numpy as np

from .board_detector import BoardDetection
from .calibration_controller import CalibrationResult
from .projector_calibration import MappingResult, ValidationResult


class CalibrationStore:
    def __init__(self, root: Path) -> None:
        self._root = root

    def save(
        self,
        result: CalibrationResult,
        *,
        camera_name: str,
        camera_id: str,
        camera_resolution: tuple[int, int],
        projector_name: str,
        projector_resolution: tuple[int, int],
    ) -> Path:
        saved_at = datetime.now().astimezone()
        metadata = {
            "schema_version": 1,
            "saved_at": saved_at.isoformat(),
            "camera": {
                "name": camera_name,
                "id": camera_id,
                "width": camera_resolution[0],
                "height": camera_resolution[1],
            },
            "projector": {
                "name": projector_name,
                "width": projector_resolution[0],
                "height": projector_resolution[1],
            },
            "board": {
                "confidence": result.board_detection.confidence,
                "method": result.board_detection.method,
                "reprojection_error_pixels": (
                    result.board_detection.reprojection_error
                ),
                "camera_grid": result.mapping.camera_grid.tolist(),
            },
            "mapping": {
                "camera_to_projector_homography": (
                    result.mapping.camera_to_projector_homography.tolist()
                ),
                "projector_grid": result.mapping.projector_grid.tolist(),
                "coverage_ratio": result.mapping.coverage_ratio,
                "decoded_ratio": result.mapping.decoded_ratio,
                "inlier_ratio": result.mapping.inlier_ratio,
                "rms_error_projector_pixels": result.mapping.rms_error,
                "maximum_error_projector_pixels": result.mapping.maximum_error,
            },
            "validation": {
                "expected_camera_points": (
                    result.validation.expected_points.tolist()
                ),
                "measured_camera_points": (
                    result.validation.measured_points.tolist()
                ),
                "errors_camera_pixels": result.validation.errors.tolist(),
                "mean_error_camera_pixels": result.validation.mean_error,
                "rms_error_camera_pixels": result.validation.rms_error,
                "maximum_error_camera_pixels": result.validation.maximum_error,
                "maximum_square_fraction": (
                    result.validation.maximum_square_fraction
                ),
            },
        }

        self._root.mkdir(parents=True, exist_ok=True)
        destination = self._root / "latest.json"
        temporary = self._root / "latest.json.tmp"
        try:
            temporary.write_text(
                json.dumps(metadata, indent=2),
                encoding="utf-8",
            )
            temporary.replace(destination)
        except OSError as error:
            raise OSError(
                f"Could not save projector calibration to {destination}: {error}"
            ) from error
        return destination

    def load_matching(
        self,
        *,
        camera_id: str,
        camera_resolution: tuple[int, int],
        projector_name: str,
        projector_resolution: tuple[int, int],
    ) -> CalibrationResult | None:
        source = self._root / "latest.json"
        if not source.exists():
            return None
        try:
            metadata = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise OSError(
                f"Could not load projector calibration from {source}: {error}"
            ) from error

        try:
            if metadata["schema_version"] != 1:
                raise ValueError("Unsupported calibration schema version.")
            camera = metadata["camera"]
            projector = metadata["projector"]
            if (
                camera["id"] != camera_id
                or (camera["width"], camera["height"]) != camera_resolution
                or projector["name"] != projector_name
                or (
                    projector["width"],
                    projector["height"],
                )
                != projector_resolution
            ):
                return None

            board = metadata["board"]
            mapping = metadata["mapping"]
            validation = metadata["validation"]
            camera_grid = np.asarray(
                board["camera_grid"],
                dtype=np.float32,
            )
            projector_grid = np.asarray(
                mapping["projector_grid"],
                dtype=np.float32,
            )
            homography = np.asarray(
                mapping["camera_to_projector_homography"],
                dtype=np.float32,
            )
            expected_points = np.asarray(
                validation["expected_camera_points"],
                dtype=np.float32,
            )
            measured_points = np.asarray(
                validation["measured_camera_points"],
                dtype=np.float32,
            )
            errors = np.asarray(
                validation["errors_camera_pixels"],
                dtype=np.float32,
            )
            if (
                camera_grid.shape != (9, 9, 2)
                or projector_grid.shape != (9, 9, 2)
                or homography.shape != (3, 3)
                or expected_points.shape != measured_points.shape
                or expected_points.ndim != 2
                or expected_points.shape[1] != 2
                or errors.shape != (expected_points.shape[0],)
            ):
                raise ValueError("Saved calibration arrays have invalid shapes.")
            maximum_square_fraction = float(
                validation["maximum_square_fraction"]
            )
            if maximum_square_fraction >= 0.1:
                raise ValueError(
                    "Saved calibration exceeds the 10% alignment limit."
                )

            detection = BoardDetection(
                grid_points=camera_grid,
                confidence=float(board["confidence"]),
                reprojection_error=(
                    None
                    if board["reprojection_error_pixels"] is None
                    else float(board["reprojection_error_pixels"])
                ),
                method=str(board["method"]),
                grid_alignment=1.0,
            )
            return CalibrationResult(
                board_detection=detection,
                mapping=MappingResult(
                    camera_grid=camera_grid,
                    projector_grid=projector_grid,
                    camera_to_projector_homography=homography,
                    coverage_ratio=float(mapping["coverage_ratio"]),
                    decoded_ratio=float(mapping["decoded_ratio"]),
                    inlier_ratio=float(mapping["inlier_ratio"]),
                    rms_error=float(mapping["rms_error_projector_pixels"]),
                    maximum_error=float(
                        mapping["maximum_error_projector_pixels"]
                    ),
                ),
                validation=ValidationResult(
                    measured_points=measured_points,
                    expected_points=expected_points,
                    errors=errors,
                    mean_error=float(validation["mean_error_camera_pixels"]),
                    rms_error=float(validation["rms_error_camera_pixels"]),
                    maximum_error=float(
                        validation["maximum_error_camera_pixels"]
                    ),
                    maximum_square_fraction=maximum_square_fraction,
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Saved projector calibration in {source} is invalid: {error}"
            ) from error
