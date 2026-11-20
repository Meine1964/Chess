from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
from numpy.typing import NDArray

from .board_detector import ImageArray
from .chess_position import Position, Square

PieceColorLabel = Literal["empty", "white", "black", "uncertain"]
FeatureArray = NDArray[np.float32]


@dataclass(frozen=True)
class PieceColorClassification:
    label: PieceColorLabel
    distance: float
    margin: float


def classify_piece_colors(
    reference: ImageArray,
    candidate: ImageArray,
    position: Position,
    *,
    maximum_distance: float = 0.35,
    minimum_margin: float = 0.04,
) -> dict[Square, PieceColorClassification]:
    if reference.shape != candidate.shape:
        raise ValueError("Reference and candidate images must have equal shape.")
    if maximum_distance <= 0 or minimum_margin <= 0:
        raise ValueError("Classification thresholds must be positive.")
    reference_features = _appearance_features(reference)
    candidate_features = _appearance_features(candidate)
    samples: dict[PieceColorLabel, list[tuple[Square, FeatureArray]]] = {
        "empty": [],
        "white": [],
        "black": [],
        "uncertain": [],
    }
    for row in range(8):
        for column in range(8):
            square = (row, column)
            piece = position.get(square)
            label: PieceColorLabel = (
                "empty"
                if piece is None
                else "white"
                if piece.isupper()
                else "black"
            )
            samples[label].append((square, reference_features[row, column]))

    classifications: dict[Square, PieceColorClassification] = {}
    for row in range(8):
        for column in range(8):
            square = (row, column)
            feature = candidate_features[row, column]
            distances: list[tuple[float, PieceColorLabel]] = []
            for label in ("empty", "white", "black"):
                label_samples = samples[label]
                if label == "empty":
                    label_samples = [
                        sample
                        for sample in label_samples
                        if sum(sample[0]) % 2 == sum(square) % 2
                    ]
                if not label_samples:
                    continue
                distances.append(
                    (
                        min(
                            float(np.linalg.norm(feature - sample_feature))
                            for _sample_square, sample_feature in label_samples
                        ),
                        label,
                    )
                )
            distances.sort(key=lambda item: (item[0], item[1]))
            if not distances:
                classifications[square] = PieceColorClassification(
                    "uncertain",
                    float("inf"),
                    0.0,
                )
                continue
            best_distance, best_label = distances[0]
            runner_up = (
                distances[1][0] if len(distances) > 1 else float("inf")
            )
            margin = runner_up - best_distance
            label = (
                best_label
                if best_distance <= maximum_distance
                and margin >= minimum_margin
                else "uncertain"
            )
            classifications[square] = PieceColorClassification(
                label,
                best_distance,
                margin,
            )
    return classifications


def _appearance_features(image: ImageArray) -> FeatureArray:
    if (
        image.ndim != 3
        or image.shape[2] != 3
        or image.shape[0] != image.shape[1]
        or image.shape[0] % 8 != 0
    ):
        raise ValueError("Board images must be square BGR images.")
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    square_size = image.shape[0] // 8
    edge = max(1, round(square_size * 0.16))
    inner_margin = max(1, round(square_size * 0.1))
    features = np.zeros((8, 8, 8), dtype=np.float32)
    for row in range(8):
        for column in range(8):
            region = lab[
                row * square_size : (row + 1) * square_size,
                column * square_size : (column + 1) * square_size,
            ]
            corners = np.concatenate(
                (
                    region[:edge, :edge].reshape(-1, 3),
                    region[:edge, -edge:].reshape(-1, 3),
                    region[-edge:, :edge].reshape(-1, 3),
                    region[-edge:, -edge:].reshape(-1, 3),
                ),
                axis=0,
            )
            background = np.median(corners, axis=0)
            inner = region[
                inner_margin:-inner_margin,
                inner_margin:-inner_margin,
            ].reshape(-1, 3)
            difference = inner - background
            magnitude = np.linalg.norm(difference, axis=1)
            foreground_threshold = float(np.quantile(magnitude, 0.65))
            foreground = inner[magnitude >= foreground_threshold]
            foreground_difference = foreground - background
            features[row, column] = np.asarray(
                (
                    *np.median(foreground, axis=0),
                    *np.median(foreground_difference, axis=0),
                    np.quantile(magnitude, 0.5),
                    np.quantile(magnitude, 0.85),
                ),
                dtype=np.float32,
            ) / np.float32(255.0)
    return features
