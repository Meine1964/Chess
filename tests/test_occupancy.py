import cv2
import numpy as np

from src.chess_camera.occupancy import (
    SquareOccupancyDetector,
    detect_starting_position_orientation,
    extract_square_regions,
    rectify_board,
    resolve_binary_occupancy,
    suppress_piece_spillover,
)


def _camera_grid(size: int = 800) -> np.ndarray:
    coordinates = np.linspace(0, size - 1, 9, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(coordinates, coordinates)
    return np.stack((x_grid, y_grid), axis=2)


def _starting_position() -> np.ndarray:
    image = np.zeros((800, 800, 3), dtype=np.uint8)
    for row in range(8):
        for column in range(8):
            color = (
                (190, 210, 225)
                if (row + column) % 2 == 0
                else (65, 90, 120)
            )
            image[
                row * 100 : (row + 1) * 100,
                column * 100 : (column + 1) * 100,
            ] = color
            if row not in (0, 1, 6, 7):
                continue
            piece_color = (
                (225, 225, 225)
                if row >= 6
                else (25, 25, 25)
            )
            center = (column * 100 + 50, row * 100 + 50)
            cv2.circle(
                image,
                center,
                28,
                piece_color,
                -1,
                cv2.LINE_AA,
            )
            cv2.circle(
                image,
                (center[0], center[1] - 10),
                17,
                tuple(min(value + 20, 255) for value in piece_color),
                -1,
                cv2.LINE_AA,
            )
    return image


def _observe_stable(
    detector: SquareOccupancyDetector,
    image: np.ndarray,
) -> object:
    update = None
    for _index in range(15):
        update = detector.observe(image, _camera_grid())
    assert update is not None
    return update


def test_rectifies_and_extracts_all_square_interiors() -> None:
    image = _starting_position()

    canonical = rectify_board(image, _camera_grid())
    regions = extract_square_regions(canonical)

    assert canonical.shape == (800, 800, 3)
    assert len(regions) == 8
    assert all(len(row) == 8 for row in regions)
    assert all(region.shape == (70, 70, 3) for row in regions for region in row)


def test_starting_position_stabilizes_to_32_occupied_squares() -> None:
    detector = SquareOccupancyDetector()

    update = _observe_stable(detector, _starting_position())

    assert update.state == "stable"
    assert update.result is not None
    assert update.result.occupied_count == 32
    assert update.result.empty_count == 32
    assert update.result.unknown_count == 0


def test_first_observation_provides_provisional_occupancy() -> None:
    detector = SquareOccupancyDetector()

    update = detector.observe(_starting_position(), _camera_grid())

    assert update.state == "stabilizing"
    assert update.observation_count == 1
    assert update.required_observations == 3
    assert update.result is not None
    assert update.result.occupied_count == 32


def test_piece_removal_and_move_change_source_and_destination() -> None:
    detector = SquareOccupancyDetector()
    starting = _starting_position()
    initial = _observe_stable(detector, starting)
    assert initial.result is not None

    moved = starting.copy()
    moved[0:100, 0:100] = (190, 210, 225)
    cv2.circle(
        moved,
        (350, 350),
        28,
        (25, 25, 25),
        -1,
        cv2.LINE_AA,
    )
    update = _observe_stable(detector, moved)

    assert update.result is not None
    assert (0, 0) not in update.result.occupied_squares
    assert (3, 3) in update.result.occupied_squares
    assert update.result.occupied_count == 32


def test_large_obstruction_freezes_occupancy_accumulation() -> None:
    detector = SquareOccupancyDetector()
    starting = _starting_position()
    _observe_stable(detector, starting)
    obstructed = starting.copy()
    cv2.rectangle(obstructed, (150, 100), (650, 700), (240, 30, 180), -1)

    update = detector.observe(obstructed, _camera_grid())

    assert update.state == "motion"
    assert update.observation_count == 0
    assert update.result is None


def test_motion_requires_quiet_period_before_occupancy_resumes() -> None:
    detector = SquareOccupancyDetector(
        required_observations=1,
        required_agreement=1,
        settling_observations=3,
    )
    starting = _starting_position()
    detector.observe(starting, _camera_grid())
    obstructed = starting.copy()
    cv2.rectangle(obstructed, (150, 100), (650, 700), (240, 30, 180), -1)

    motion = detector.observe(obstructed, _camera_grid())
    settling = [
        detector.observe(obstructed, _camera_grid())
        for _index in range(3)
    ]
    resumed = detector.observe(obstructed, _camera_grid())

    assert motion.state == "motion"
    assert [update.state for update in settling] == ["settling"] * 3
    assert all(update.result is None for update in settling)
    assert resumed.state == "stable"


def test_start_detector_rebuilds_appearance_model_after_motion() -> None:
    detector = SquareOccupancyDetector(
        required_observations=1,
        required_agreement=1,
        settling_observations=1,
        rebuild_models_after_motion=True,
    )
    starting = _starting_position()
    detector.observe(starting, _camera_grid())
    obstructed = starting.copy()
    cv2.rectangle(obstructed, (150, 100), (650, 700), (240, 30, 180), -1)

    motion = detector.observe(obstructed, _camera_grid())
    settling = detector.observe(obstructed, _camera_grid())
    resumed = detector.observe(obstructed, _camera_grid())

    assert motion.state == "motion"
    assert settling.state == "settling"
    assert resumed.state == "stable"
    assert resumed.result is not None


def test_piece_colors_put_white_starting_side_at_ui_bottom() -> None:
    starting = _starting_position()
    occupied = frozenset(
        (row, column)
        for row in (0, 1, 6, 7)
        for column in range(8)
    )

    white_at_bottom = detect_starting_position_orientation(
        starting,
        _camera_grid(),
        occupied,
        0,
    )
    white_at_top = detect_starting_position_orientation(
        cv2.rotate(starting, cv2.ROTATE_180),
        _camera_grid(),
        occupied,
        0,
    )

    assert white_at_bottom == 0
    assert white_at_top == 2


def test_piece_color_orientation_allows_small_starting_side_gaps() -> None:
    starting = cv2.rotate(_starting_position(), cv2.ROTATE_180)
    occupied = {
        (row, column)
        for row in (0, 1, 6, 7)
        for column in range(8)
    }
    occupied.difference_update({(0, 0), (1, 1), (6, 6), (7, 7)})
    occupied.update({(3, 3), (4, 4)})

    assert (
        detect_starting_position_orientation(
            starting,
            _camera_grid(),
            frozenset(occupied),
            0,
        )
        == 2
    )


def test_piece_color_orientation_requires_evidence_on_both_sides() -> None:
    with np.testing.assert_raises_regex(
        ValueError,
        "at least eight occupied squares",
    ):
        detect_starting_position_orientation(
            _starting_position(),
            _camera_grid(),
            frozenset({(0, 0)}),
            0,
        )


def test_suppresses_weak_adjacent_tall_piece_spillover() -> None:
    labels = np.full((8, 8), "empty", dtype="<U8")
    scores = np.zeros((8, 8), dtype=np.float32)
    labels[3, 2] = "occupied"
    scores[3, 2] = 24.4
    labels[4, 2] = "occupied"
    scores[4, 2] = 6.2

    filtered = suppress_piece_spillover(labels, scores)

    assert filtered[3, 2] == "occupied"
    assert filtered[4, 2] == "empty"


def test_preserves_adjacent_pieces_with_independent_strong_evidence() -> None:
    labels = np.full((8, 8), "empty", dtype="<U8")
    scores = np.zeros((8, 8), dtype=np.float32)
    labels[3, 2] = "occupied"
    scores[3, 2] = 24.4
    labels[4, 2] = "occupied"
    scores[4, 2] = 13.0

    filtered = suppress_piece_spillover(labels, scores)

    assert filtered[3, 2] == "occupied"
    assert filtered[4, 2] == "occupied"


def test_resolves_every_square_to_binary_occupancy() -> None:
    labels = np.full((8, 8), "unknown", dtype="<U8")
    scores = np.zeros((8, 8), dtype=np.float32)
    scores[2, 3] = 7.9
    scores[4, 5] = 8.0

    resolved = resolve_binary_occupancy(labels, scores)

    assert resolved[2, 3] == "empty"
    assert resolved[4, 5] == "occupied"
    assert not np.any(resolved == "unknown")
