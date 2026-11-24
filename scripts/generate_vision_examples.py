"""Generate demonstration-only review records from a camera image and grid."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from src.chess_camera.vision_data import VisionDataset, demonstration_samples
from src.chess_camera.board_detector import BoardDetector


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    geometry = parser.add_mutually_exclusive_group(required=True)
    geometry.add_argument("--grid", type=Path)
    geometry.add_argument(
        "--detect-grid", action="store_true",
        help="Explicitly detect geometry on this recorded image; never modifies calibration.",
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    image = cv2.imread(str(arguments.image))
    if image is None:
        parser.error("Could not read the recorded camera image.")
    image = np.asarray(image, dtype=np.uint8)
    if arguments.detect_grid:
        detection = BoardDetector().detect(image)
        if detection is None:
            parser.error("No board detected on the recorded image; supply matching geometry.")
        grid = detection.grid_points
    else:
        metadata = json.loads(arguments.grid.read_text(encoding="utf-8"))
        grid = np.asarray(metadata["board"]["camera_grid"], dtype=np.float32)
    view, samples = demonstration_samples(image, grid)
    dataset = VisionDataset.create(arguments.output)
    dataset.save(samples, 2_000_000_000)
    (arguments.output / "recorded-grid.json").write_text(
        json.dumps({"board": {"camera_grid": grid.tolist()}}, indent=2),
        encoding="utf-8",
    )
    for filename, image in (
        ("normalized-preview.png", view.image),
        ("normalized-validity.png", view.valid),
    ):
        if not cv2.imwrite(str(arguments.output / filename), image):
            raise OSError(f"Could not write {filename}.")
    print(f"Generated {len(samples)} demonstration-only samples in {dataset.path}")
    print("All samples remain excluded from default training selection after review.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
