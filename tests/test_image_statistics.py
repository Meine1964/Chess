import numpy as np
import pytest

from src.chess_camera.image_statistics import (
    median_uint8_difference,
    offset_uint8,
)


@pytest.mark.parametrize("count", [1, 7, 8, 10_000, 10_001])
def test_uint8_helpers_match_floating_point_reference(count: int) -> None:
    rng = np.random.default_rng(count)
    minuend = rng.integers(0, 256, count, dtype=np.uint8)
    subtrahend = rng.integers(0, 256, count, dtype=np.uint8)

    expected_median = float(
        np.median(minuend.astype(np.float32) - subtrahend)
    )
    median = median_uint8_difference(minuend, subtrahend)

    assert median == expected_median
    expected_offset = np.clip(
        subtrahend.astype(np.float32) + expected_median,
        0,
        255,
    ).astype(np.uint8)
    assert np.array_equal(offset_uint8(subtrahend, median), expected_offset)


@pytest.mark.parametrize("offset", [-300.0, -12.5, 0.0, 0.5, 254.5, 300.0])
def test_offset_uint8_matches_clip_and_truncation(offset: float) -> None:
    image = np.arange(256, dtype=np.uint8).reshape(16, 16)

    expected = np.clip(image.astype(np.float32) + offset, 0, 255).astype(
        np.uint8
    )

    assert np.array_equal(offset_uint8(image, offset), expected)
