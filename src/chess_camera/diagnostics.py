import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import cv2

from .board_detector import BoardDetection, ImageArray


class DiagnosticStore:
    def __init__(self, root: Path, max_captures: int = 100) -> None:
        if max_captures < 1:
            raise ValueError("max_captures must be at least 1.")
        self._root = root
        self._max_captures = max_captures

    def save(
        self,
        original: ImageArray,
        overlay: ImageArray | None,
        detection: BoardDetection | None,
        camera_name: str,
        camera_id: str,
    ) -> Path:
        captured_at = datetime.now().astimezone()
        capture_name = (
            f"{captured_at:%Y-%m-%d_%H-%M-%S_%f}_{uuid4().hex[:8]}"
        )
        capture_directory = self._root / capture_name
        capture_directory.mkdir(parents=True, exist_ok=False)

        self._write_image(capture_directory / "original.png", original)
        if overlay is not None:
            self._write_image(capture_directory / "overlay.png", overlay)

        height, width = original.shape[:2]
        detection_metadata: dict[str, object] = {
            "success": detection is not None,
        }
        if detection is not None:
            detection_metadata.update(
                {
                    "confidence": detection.confidence,
                    "reprojection_error_pixels": detection.reprojection_error,
                    "method": detection.method,
                    "grid_alignment": detection.grid_alignment,
                    "grid_points": detection.grid_points.tolist(),
                }
            )

        metadata = {
            "schema_version": 1,
            "captured_at": captured_at.isoformat(),
            "camera": {
                "name": camera_name,
                "id": camera_id,
            },
            "image": {
                "width": width,
                "height": height,
            },
            "opencv_version": cv2.__version__,
            "detection": detection_metadata,
        }
        metadata_path = capture_directory / "detection.json"
        try:
            metadata_path.write_text(
                json.dumps(metadata, indent=2),
                encoding="utf-8",
            )
        except OSError as error:
            raise OSError(
                f"Could not write diagnostic metadata to {metadata_path}: {error}"
            ) from error

        self._remove_expired_captures()
        return capture_directory

    @staticmethod
    def _write_image(path: Path, image: ImageArray) -> None:
        try:
            written = cv2.imwrite(str(path), image)
        except cv2.error as error:
            raise OSError(f"Could not write diagnostic image to {path}: {error}") from error
        if not written:
            raise OSError(f"Could not write diagnostic image to {path}.")

    def _remove_expired_captures(self) -> None:
        capture_directories = sorted(
            path for path in self._root.iterdir() if path.is_dir()
        )
        for expired_directory in capture_directories[: -self._max_captures]:
            for child in expired_directory.iterdir():
                if not child.is_file():
                    raise OSError(
                        f"Unexpected directory in diagnostic capture: {child}"
                    )
                child.unlink()
            expired_directory.rmdir()
