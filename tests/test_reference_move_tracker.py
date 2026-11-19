import cv2
import numpy as np

from src.chess_camera.reference_move_tracker import (
    ReferenceMoveTracker,
    square_change_scores,
)


def _camera_grid(size: int = 800) -> np.ndarray:
    coordinates = np.linspace(0, size - 1, 9, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(coordinates, coordinates)
    return np.stack((x_grid, y_grid), axis=2)


def _empty_board() -> np.ndarray:
    image = np.zeros((800, 800, 3), dtype=np.uint8)
    for row in range(8):
        for column in range(8):
            image[
                row * 100 : (row + 1) * 100,
                column * 100 : (column + 1) * 100,
            ] = (
                (190, 210, 225)
                if (row + column) % 2 == 0
                else (65, 90, 120)
            )
    return image


def _draw_piece(
    image: np.ndarray,
    square: tuple[int, int],
    color: tuple[int, int, int] = (20, 20, 20),
) -> None:
    row, column = square
    cv2.circle(
        image,
        (column * 100 + 50, row * 100 + 55),
        28,
        color,
        -1,
        cv2.LINE_AA,
    )
    cv2.circle(
        image,
        (column * 100 + 50, row * 100 + 30),
        18,
        color,
        -1,
        cv2.LINE_AA,
    )


def _accepted_update(
    tracker: ReferenceMoveTracker,
    image: np.ndarray,
    position: dict[tuple[int, int], str],
) -> object:
    update = None
    for _index in range(5):
        update = tracker.observe(image, _camera_grid(), 0, position)
        if update.state == "accepted":
            return update
    assert update is not None
    return update


def test_change_scores_localize_source_and_destination() -> None:
    source = (0, 1)
    destination = (0, 3)
    reference = _empty_board()
    _draw_piece(reference, source)
    moved = _empty_board()
    _draw_piece(moved, destination)

    scores = square_change_scores(reference, moved)
    ranked = sorted(
        (
            (float(scores[row, column]), (row, column))
            for row in range(8)
            for column in range(8)
        ),
        reverse=True,
    )

    assert {ranked[0][1], ranked[1][1]} == {source, destination}
    assert ranked[1][0] > ranked[2][0] + 0.025


def test_accepts_non_legal_single_piece_relocation() -> None:
    source = (0, 1)
    destination = (0, 3)
    reference = _empty_board()
    _draw_piece(reference, source)
    moved = _empty_board()
    _draw_piece(moved, destination)
    tracker = ReferenceMoveTracker()
    tracker.set_reference(reference, _camera_grid(), 0)

    update = _accepted_update(tracker, moved, {source: "n"})

    assert update.state == "accepted"
    assert update.move == (source, destination)
    np.testing.assert_array_equal(tracker.reference, reference)
    tracker.commit_accepted(update.move)
    np.testing.assert_allclose(tracker.reference, moved, atol=3)


def test_discarding_accepted_candidate_preserves_reference() -> None:
    source = (0, 1)
    destination = (0, 3)
    reference = _empty_board()
    _draw_piece(reference, source)
    moved = _empty_board()
    _draw_piece(moved, destination)
    tracker = ReferenceMoveTracker()
    tracker.set_reference(reference, _camera_grid(), 0)

    update = _accepted_update(tracker, moved, {source: "n"})
    assert update.state == "accepted"
    tracker.reset_activity()

    np.testing.assert_array_equal(tracker.reference, reference)


def test_global_exposure_change_does_not_produce_move() -> None:
    reference = _empty_board()
    _draw_piece(reference, (0, 1))
    brighter = cv2.convertScaleAbs(reference, alpha=1.0, beta=18)
    tracker = ReferenceMoveTracker()
    tracker.set_reference(reference, _camera_grid(), 0)

    updates = [
        tracker.observe(brighter, _camera_grid(), 0, {(0, 1): "n"})
        for _index in range(4)
    ]

    assert updates[-1].state == "watching"
    assert all(update.move is None for update in updates)


def test_small_image_translation_does_not_produce_move() -> None:
    reference = _empty_board()
    _draw_piece(reference, (0, 1))
    translated = cv2.warpAffine(
        reference,
        np.float32([[1, 0, 2], [0, 1, 2]]),
        (800, 800),
        borderMode=cv2.BORDER_REFLECT,
    )
    tracker = ReferenceMoveTracker()
    tracker.set_reference(reference, _camera_grid(), 0)

    updates = [
        tracker.observe(translated, _camera_grid(), 0, {(0, 1): "n"})
        for _index in range(4)
    ]

    assert updates[-1].state == "watching"
    assert all(update.move is None for update in updates)


def test_accepts_knight_move_after_small_grid_remap() -> None:
    source = (7, 6)
    destination = (5, 5)
    reference = _empty_board()
    _draw_piece(reference, source)
    moved = _empty_board()
    _draw_piece(moved, destination)
    corners = np.float32(
        [[0, 0], [799, 0], [799, 799], [0, 799]]
    )
    remapped_corners = np.float32(
        [[3, 0], [796, 3], [799, 796], [0, 799]]
    )
    remapped = cv2.warpPerspective(
        moved,
        cv2.getPerspectiveTransform(corners, remapped_corners),
        (800, 800),
        borderMode=cv2.BORDER_REFLECT,
    )
    tracker = ReferenceMoveTracker()
    tracker.set_reference(reference, _camera_grid(), 0)

    update = _accepted_update(tracker, remapped, {source: "N"})

    assert update.state == "accepted"
    assert update.move == (source, destination)


def test_stationary_large_obstruction_is_not_evaluated_as_move() -> None:
    reference = _empty_board()
    _draw_piece(reference, (0, 1))
    obstructed = reference.copy()
    cv2.rectangle(obstructed, (120, 100), (650, 650), (30, 40, 220), -1)
    tracker = ReferenceMoveTracker()
    tracker.set_reference(reference, _camera_grid(), 0)

    activity = tracker.observe(
        obstructed,
        _camera_grid(),
        0,
        {(0, 1): "n"},
    )
    waiting = tracker.observe(
        obstructed,
        _camera_grid(),
        0,
        {(0, 1): "n"},
    )

    assert activity.state in ("activity", "waiting-for-clear")
    assert waiting.state == "waiting-for-clear"
    assert waiting.move is None


def test_ambiguous_multi_square_change_preserves_reference() -> None:
    reference = _empty_board()
    sources = {(0, 1): "n", (1, 1): "p"}
    for square in sources:
        _draw_piece(reference, square)
    changed = _empty_board()
    _draw_piece(changed, (3, 3))
    _draw_piece(changed, (4, 4))
    tracker = ReferenceMoveTracker()
    tracker.set_reference(reference, _camera_grid(), 0)
    original_reference = tracker.reference

    update = _accepted_update(tracker, changed, sources)

    assert update.state == "ambiguous"
    np.testing.assert_array_equal(tracker.reference, original_reference)
