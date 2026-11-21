import cv2
import numpy as np

from src.chess_camera.promotion_selection import (
    ManipulationSettlingDetector,
    observe_promotion_choices,
    promotion_choice_polygons,
)


def _grid() -> np.ndarray:
    return np.asarray(
        [
            [[100 + column * 50, 100 + row * 50] for column in range(9)]
            for row in range(9)
        ],
        dtype=np.float32,
    )


def test_promotion_choices_form_extra_square_spaced_rank() -> None:
    polygons = promotion_choice_polygons(_grid(), "top")

    assert polygons.shape == (4, 4, 2)
    assert np.allclose(polygons[0], ((200, 50), (250, 50), (250, 100), (200, 100)))
    centers = np.mean(polygons, axis=1)
    assert np.allclose(np.diff(centers[:, 0]), (50, 50, 50))


def test_promotion_choices_support_vertical_board_edges() -> None:
    polygons = promotion_choice_polygons(_grid(), "left")

    assert np.allclose(
        polygons[0],
        ((50, 200), (50, 250), (100, 250), (100, 200)),
    )
    centers = np.mean(polygons, axis=1)
    assert np.allclose(np.diff(centers[:, 1]), (50, 50, 50))


def test_single_changed_virtual_square_selects_piece() -> None:
    grid = _grid()
    polygons = promotion_choice_polygons(grid, "top")
    reference = np.full((600, 600, 3), 120, dtype=np.uint8)
    current = reference.copy()
    cv2.circle(current, (325, 75), 14, (20, 20, 20), -1)

    observation = observe_promotion_choices(
        reference,
        current,
        polygons,
    )

    assert observation.occupied_choices == ("r",)
    assert observation.selected_piece == "r"


def test_multiple_changed_virtual_squares_are_ambiguous() -> None:
    polygons = promotion_choice_polygons(_grid(), "bottom")
    reference = np.full((600, 600, 3), 120, dtype=np.uint8)
    current = reference.copy()
    for center_x in (225, 275):
        cv2.circle(current, (center_x, 525), 14, (20, 20, 20), -1)

    observation = observe_promotion_choices(
        reference,
        current,
        polygons,
    )

    assert observation.occupied_choices == ("n", "b")
    assert observation.selected_piece is None


def test_manipulation_detector_waits_for_activity_then_quiet() -> None:
    polygons = promotion_choice_polygons(_grid(), "top")
    detector = ManipulationSettlingDetector(
        arming_observations=1,
        activity_observations=1,
        quiet_observations=2,
    )
    baseline = np.full((600, 600, 3), 120, dtype=np.uint8)
    active = baseline.copy()
    cv2.rectangle(active, (190, 45), (310, 105), (10, 10, 10), -1)

    assert not detector.observe(baseline, polygons)
    assert not detector.observe(baseline, polygons)
    assert not detector.observe(active, polygons)
    assert not detector.observe(active, polygons)
    assert detector.observe(active, polygons)
