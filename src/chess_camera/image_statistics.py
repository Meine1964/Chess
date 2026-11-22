import cv2
import numpy as np


def median_uint8_difference(
    minuend: np.ndarray,
    subtrahend: np.ndarray,
) -> float:
    """Return np.median(minuend - subtrahend) for uint8 inputs via a histogram."""
    differences = minuend.astype(np.int16) - subtrahend.astype(np.int16)
    counts = np.bincount(differences.ravel() + 255, minlength=511)
    total = int(differences.size)
    if total == 0:
        raise ValueError("The median of an empty selection is undefined.")
    cumulative = np.cumsum(counts)
    lower = int(np.searchsorted(cumulative, (total - 1) // 2, side="right"))
    upper = int(np.searchsorted(cumulative, total // 2, side="right"))
    return (lower + upper - 510) / 2


def offset_uint8(image: np.ndarray, offset: float) -> np.ndarray:
    """Return float32 clip(image + offset, 0, 255) truncated to uint8."""
    table = np.clip(
        np.arange(256, dtype=np.float32) + offset,
        0,
        255,
    ).astype(np.uint8)
    return np.asarray(
        cv2.LUT(np.ascontiguousarray(image, dtype=np.uint8), table),
        dtype=np.uint8,
    )
