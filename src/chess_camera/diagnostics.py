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


class RuntimeDiagnosticLog:
    def __init__(
        self,
        root: Path,
        *,
        maximum_bytes: int = 2_000_000,
    ) -> None:
        if maximum_bytes < 10_000:
            raise ValueError("maximum_bytes must be at least 10000.")
        self._root = root
        self._maximum_bytes = maximum_bytes
        self._state_path = root / "runtime-state.json"
        self._events_path = root / "runtime-events.jsonl"

    @property
    def state_path(self) -> Path:
        return self._state_path

    @property
    def events_path(self) -> Path:
        return self._events_path

    def record(
        self,
        event_type: str,
        message: str,
        state: dict[str, object],
        *,
        details: dict[str, object] | None = None,
        images: dict[str, ImageArray] | None = None,
    ) -> None:
        if not event_type.strip():
            raise ValueError("A runtime diagnostic event type is required.")
        payload: dict[str, object] = {
            "schema_version": 1,
            "recorded_at": datetime.now().astimezone().isoformat(),
            "event": {
                "type": event_type,
                "message": message,
                "details": details or {},
            },
            "state": state,
        }
        _reject_sensitive_diagnostic_keys(payload)
        self._root.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(payload, sort_keys=True)
        temporary_path = self._state_path.with_suffix(".json.tmp")
        try:
            temporary_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            temporary_path.replace(self._state_path)
            with self._events_path.open("a", encoding="utf-8") as events:
                events.write(f"{serialized}\n")
            if images is not None:
                for name, image in images.items():
                    self._write_runtime_image(name, image)
            self._truncate_events_if_needed()
        except OSError as error:
            raise OSError(
                f"Could not write runtime diagnostics: {error}"
            ) from error

    def _write_runtime_image(
        self,
        name: str,
        image: ImageArray,
    ) -> None:
        if (
            not name
            or any(character not in "abcdefghijklmnopqrstuvwxyz-" for character in name)
        ):
            raise ValueError("A runtime image name must use lowercase letters.")
        final_path = self._root / f"runtime-{name}.png"
        temporary_path = self._root / f"runtime-{name}.tmp.png"
        try:
            written = cv2.imwrite(str(temporary_path), image)
        except cv2.error as error:
            raise OSError(
                f"Could not write runtime diagnostic image: {error}"
            ) from error
        if not written:
            raise OSError(
                f"Could not write runtime diagnostic image to {temporary_path}."
            )
        temporary_path.replace(final_path)

    def _truncate_events_if_needed(self) -> None:
        if self._events_path.stat().st_size <= self._maximum_bytes:
            return
        lines = self._events_path.read_text(encoding="utf-8").splitlines()
        retained: list[str] = []
        retained_bytes = 0
        target_bytes = self._maximum_bytes // 2
        for line in reversed(lines):
            line_bytes = len(line.encode("utf-8")) + 1
            if retained and retained_bytes + line_bytes > target_bytes:
                break
            retained.append(line)
            retained_bytes += line_bytes
        retained.reverse()
        self._events_path.write_text(
            "".join(f"{line}\n" for line in retained),
            encoding="utf-8",
        )


def _reject_sensitive_diagnostic_keys(payload: object) -> None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            normalized = key.casefold()
            if any(
                sensitive in normalized
                for sensitive in ("token", "authorization", "secret")
            ):
                raise ValueError(
                    f"Sensitive runtime diagnostic key is forbidden: {key}"
                )
            _reject_sensitive_diagnostic_keys(value)
    elif isinstance(payload, (list, tuple)):
        for value in payload:
            _reject_sensitive_diagnostic_keys(value)
