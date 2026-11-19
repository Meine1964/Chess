from pathlib import Path

import numpy as np

from .board_detector import ImageArray
from .chess_position import Position


class ReferenceImageStore:
    def __init__(self, path: Path) -> None:
        self._path = path

    def save(
        self,
        image: ImageArray,
        *,
        camera_id: str,
        camera_resolution: tuple[int, int],
        orientation: int,
        position: Position,
    ) -> None:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Reference image must be a BGR color image.")
        if orientation not in (0, 1, 2, 3):
            raise ValueError("Reference orientation must be a quarter turn.")
        position_rows = np.asarray(
            [
                (row, column, ord(piece))
                for (row, column), piece in sorted(position.items())
            ],
            dtype=np.int16,
        )
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_suffix(".tmp")
        try:
            with temporary.open("wb") as stream:
                np.savez_compressed(
                    stream,
                    image=np.asarray(image, dtype=np.uint8),
                    camera_id=np.asarray(camera_id),
                    camera_resolution=np.asarray(
                        camera_resolution,
                        dtype=np.int32,
                    ),
                    orientation=np.asarray(orientation, dtype=np.int8),
                    position=position_rows,
                )
            temporary.replace(self._path)
        except OSError as error:
            raise OSError(
                f"Could not save move reference to {self._path}: {error}"
            ) from error

    def load_matching(
        self,
        *,
        camera_id: str,
        camera_resolution: tuple[int, int],
        orientation: int,
        position: Position,
    ) -> ImageArray | None:
        if not self._path.exists():
            return None
        try:
            with np.load(self._path, allow_pickle=False) as data:
                stored_camera = str(data["camera_id"].item())
                stored_resolution = tuple(
                    int(value) for value in data["camera_resolution"]
                )
                stored_orientation = int(data["orientation"].item())
                stored_position = {
                    (int(row), int(column)): chr(int(piece))
                    for row, column, piece in data["position"]
                }
                image = np.asarray(data["image"], dtype=np.uint8)
        except (OSError, ValueError, KeyError) as error:
            raise ValueError(
                f"Saved move reference in {self._path} is invalid: {error}"
            ) from error
        if (
            stored_camera != camera_id
            or stored_resolution != camera_resolution
            or stored_orientation != orientation
            or stored_position != position
        ):
            return None
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Saved move reference image has invalid shape.")
        return image.copy()
