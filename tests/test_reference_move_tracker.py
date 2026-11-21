import cv2
import numpy as np
import pytest

from src.chess_camera.chess_position import (
    PiecePositionSynchronizer,
    starting_occupied_squares,
)
from src.chess_camera.legal_game import LegalGame
from src.chess_camera.reference_move_tracker import (
    ReferenceMoveTracker,
    ReferenceTrackerUpdate,
    _has_large_obstruction,
    _matches_signature_result_state,
    square_change_scores,
)
from src.chess_camera.piece_color_classifier import PieceColorClassification


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
    legal_signatures: tuple = (),
) -> object:
    update = None
    for _index in range(5):
        update = tracker.observe(
            image,
            _camera_grid(),
            0,
            position,
            legal_signatures,
        )
        if update.state in ("accepted", "illegal-relocation"):
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


def _adjacent_queen_residual(
    retained_piece: str,
) -> tuple[np.ndarray, np.ndarray, dict[tuple[int, int], str]]:
    source = (6, 3)
    retained = (7, 3)
    destination = (2, 7)
    reference = _empty_board()
    moved = _empty_board()
    _draw_piece(reference, retained, (235, 235, 235))
    _draw_piece(moved, retained, (235, 235, 235))
    _draw_piece(reference, source)
    _draw_piece(moved, destination)
    cv2.rectangle(
        reference,
        (315, 700),
        (385, 748),
        (20, 20, 20),
        -1,
    )
    return (
        reference,
        moved,
        {source: "q", retained: retained_piece},
    )


def test_retained_opposite_piece_resolves_adjacent_residual() -> None:
    reference, moved, position = _adjacent_queen_residual("Q")
    tracker = ReferenceMoveTracker()
    tracker.set_reference(reference, _camera_grid(), 0)

    update = _accepted_update(tracker, moved, position)

    assert update.state == "accepted"
    assert update.move == ((6, 3), (2, 7))


def test_same_color_adjacent_residual_remains_ambiguous() -> None:
    reference, moved, position = _adjacent_queen_residual("r")
    tracker = ReferenceMoveTracker()
    tracker.set_reference(reference, _camera_grid(), 0)

    update = _accepted_update(tracker, moved, position)

    assert update.state == "ambiguous"
    assert update.move is None


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


@pytest.mark.parametrize(
    ("fen", "source", "destination", "uci", "moving_piece"),
    (
        (
            "4k3/8/8/8/1n6/P7/8/4K3 w - - 0 1",
            (5, 0),
            (4, 1),
            "a3b4",
            "P",
        ),
        (
            "4k3/8/8/8/1n6/3Q4/8/4K3 b - - 0 1",
            (4, 1),
            (5, 3),
            "b4d3",
            "n",
        ),
    ),
)
def test_legal_signature_tracker_accepts_ordinary_capture(
    fen: str,
    source: tuple[int, int],
    destination: tuple[int, int],
    uci: str,
    moving_piece: str,
) -> None:
    game = LegalGame(fen)
    reference = _empty_board()
    for square, piece in game.position.items():
        _draw_piece(
            reference,
            square,
            (225, 225, 225) if piece.isupper() else (20, 20, 20),
        )
    moved = reference.copy()
    empty = _empty_board()
    for square in (source, destination):
        row, column = square
        moved[
            row * 100 : (row + 1) * 100,
            column * 100 : (column + 1) * 100,
        ] = empty[
            row * 100 : (row + 1) * 100,
            column * 100 : (column + 1) * 100,
        ]
    _draw_piece(
        moved,
        destination,
        (225, 225, 225) if moving_piece.isupper() else (20, 20, 20),
    )
    tracker = ReferenceMoveTracker()
    tracker.set_reference(reference, _camera_grid(), 0)

    update = _accepted_update(
        tracker,
        moved,
        game.position,
        game.legal_move_signatures,
    )

    assert update.state == "accepted"
    assert update.move == (source, destination)
    signature = next(
        signature
        for signature in game.legal_move_signatures
        if signature.uci == uci
    )
    assert signature.kind == "capture"


def test_two_action_capture_waits_for_attacker_before_accepting() -> None:
    game = LegalGame("4k3/8/8/8/1n6/P7/8/4K3 w - - 0 1")
    source = (5, 0)
    destination = (4, 1)
    reference = _empty_board()
    for square, piece in game.position.items():
        _draw_piece(
            reference,
            square,
            (225, 225, 225) if piece.isupper() else (20, 20, 20),
        )
    empty = _empty_board()
    captured_piece_removed = reference.copy()
    row, column = destination
    captured_piece_removed[
        row * 100 : (row + 1) * 100,
        column * 100 : (column + 1) * 100,
    ] = empty[
        row * 100 : (row + 1) * 100,
        column * 100 : (column + 1) * 100,
    ]
    tracker = ReferenceMoveTracker()
    tracker.set_reference(reference, _camera_grid(), 0)

    pending = _accepted_update(
        tracker,
        captured_piece_removed,
        game.position,
        game.physical_move_signatures,
    )

    assert pending.state == "pending-capture"
    assert pending.move is None
    assert "removed from b4" in pending.message
    assert "from a3" in pending.message
    assert game.fen == "4k3/8/8/8/1n6/P7/8/4K3 w - - 0 1"
    np.testing.assert_array_equal(tracker.reference, reference)

    restored = _accepted_update(
        tracker,
        reference,
        game.position,
        game.physical_move_signatures,
    )

    assert restored.state == "watching"
    pending_again = _accepted_update(
        tracker,
        captured_piece_removed,
        game.position,
        game.physical_move_signatures,
    )
    assert pending_again.state == "pending-capture"

    completed = captured_piece_removed.copy()
    row, column = source
    completed[
        row * 100 : (row + 1) * 100,
        column * 100 : (column + 1) * 100,
    ] = empty[
        row * 100 : (row + 1) * 100,
        column * 100 : (column + 1) * 100,
    ]
    _draw_piece(completed, destination, (225, 225, 225))

    accepted = _accepted_update(
        tracker,
        completed,
        game.position,
        game.physical_move_signatures,
    )

    assert accepted.state == "accepted"
    assert accepted.move == (source, destination)


def test_king_first_castling_relocation_waits_when_visual_match_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.chess_camera.reference_move_tracker as tracker_module
    from src.chess_camera.legal_signature_matcher import PendingCastlingMatch

    monkeypatch.setattr(
        tracker_module,
        "match_pending_castling",
        lambda *_args, **_kwargs: PendingCastlingMatch(
            None,
            None,
            None,
            "Visual castling match unavailable.",
        ),
    )
    game = LegalGame("4k3/8/8/8/8/8/8/4K2R w K - 0 1")
    king_source = (7, 4)
    king_destination = (7, 6)
    rook_source = (7, 7)
    rook_destination = (7, 5)
    reference = _empty_board()
    for square, piece in game.position.items():
        _draw_piece(
            reference,
            square,
            (225, 225, 225) if piece.isupper() else (20, 20, 20),
        )
    empty = _empty_board()
    king_moved = reference.copy()
    king_moved[
        king_source[0] * 100 : (king_source[0] + 1) * 100,
        king_source[1] * 100 : (king_source[1] + 1) * 100,
    ] = empty[
        king_source[0] * 100 : (king_source[0] + 1) * 100,
        king_source[1] * 100 : (king_source[1] + 1) * 100,
    ]
    _draw_piece(king_moved, king_destination, (225, 225, 225))
    tracker = ReferenceMoveTracker()
    tracker.set_reference(reference, _camera_grid(), 0)

    pending = _accepted_update(
        tracker,
        king_moved,
        game.position,
        game.physical_move_signatures,
    )

    assert pending.state == "pending-castling"
    assert pending.move is None
    assert "rook from h1 to f1" in pending.message
    np.testing.assert_array_equal(tracker.reference, reference)

    completed = king_moved.copy()
    completed[
        rook_source[0] * 100 : (rook_source[0] + 1) * 100,
        rook_source[1] * 100 : (rook_source[1] + 1) * 100,
    ] = empty[
        rook_source[0] * 100 : (rook_source[0] + 1) * 100,
        rook_source[1] * 100 : (rook_source[1] + 1) * 100,
    ]
    _draw_piece(completed, rook_destination, (225, 225, 225))

    accepted = _accepted_update(
        tracker,
        completed,
        game.position,
        game.physical_move_signatures,
    )

    assert accepted.state == "accepted"
    assert accepted.move == (king_source, king_destination)


def test_piece_color_reference_is_cached_until_reference_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.chess_camera.reference_move_tracker as tracker_module

    reference = _empty_board()
    position = {(7, 4): "K", (0, 4): "k"}
    for square, piece in position.items():
        _draw_piece(
            reference,
            square,
            (225, 225, 225) if piece.isupper() else (20, 20, 20),
        )
    original_build = tracker_module.build_piece_color_reference
    build_count = 0

    def counted_build(
        image: np.ndarray,
        model_position: dict[tuple[int, int], str],
    ):
        nonlocal build_count
        build_count += 1
        return original_build(image, model_position)

    monkeypatch.setattr(
        tracker_module,
        "build_piece_color_reference",
        counted_build,
    )
    tracker = ReferenceMoveTracker()
    tracker.set_reference(reference, _camera_grid(), 0)

    tracker.observe(reference, _camera_grid(), 0, position)
    watching = tracker.observe(reference, _camera_grid(), 0, position)
    assert watching.state == "watching"
    assert build_count == 1

    tracker._classify_piece_colors(reference, position)
    tracker._classify_piece_colors(reference, dict(position))
    assert build_count == 1

    tracker.note_camera_grid_remap()
    tracker._classify_piece_colors(reference, position)
    assert build_count == 2


def test_captured_pawn_first_en_passant_waits_for_attacker() -> None:
    game = LegalGame(
        "rnbqkbnr/1pp1pppp/p7/3pP3/8/8/PPPP1PPP/"
        "RNBQKBNR w KQkq d6 0 3"
    )
    source = (3, 4)
    destination = (2, 3)
    captured = (3, 3)
    reference = _empty_board()
    for square, piece in game.position.items():
        _draw_piece(
            reference,
            square,
            (225, 225, 225) if piece.isupper() else (20, 20, 20),
        )
    empty = _empty_board()
    captured_removed = reference.copy()
    captured_removed[
        captured[0] * 100 : (captured[0] + 1) * 100,
        captured[1] * 100 : (captured[1] + 1) * 100,
    ] = empty[
        captured[0] * 100 : (captured[0] + 1) * 100,
        captured[1] * 100 : (captured[1] + 1) * 100,
    ]
    tracker = ReferenceMoveTracker()
    tracker.set_reference(reference, _camera_grid(), 0)

    pending = _accepted_update(
        tracker,
        captured_removed,
        game.position,
        game.physical_move_signatures,
    )

    assert pending.state == "pending-en-passant"
    assert pending.move is None
    assert "from e5 to d6" in pending.message
    np.testing.assert_array_equal(tracker.reference, reference)

    completed = captured_removed.copy()
    completed[
        source[0] * 100 : (source[0] + 1) * 100,
        source[1] * 100 : (source[1] + 1) * 100,
    ] = empty[
        source[0] * 100 : (source[0] + 1) * 100,
        source[1] * 100 : (source[1] + 1) * 100,
    ]
    _draw_piece(completed, destination, (225, 225, 225))

    accepted = _accepted_update(
        tracker,
        completed,
        game.position,
        game.physical_move_signatures,
    )

    assert accepted.state == "accepted"
    assert accepted.move == (source, destination)


def test_captured_piece_first_capture_promotion_waits_for_pawn() -> None:
    game = LegalGame("1r5k/P7/8/8/8/8/8/7K w - - 0 1")
    source = (1, 0)
    destination = (0, 1)
    reference = _empty_board()
    for square, piece in game.position.items():
        _draw_piece(
            reference,
            square,
            (225, 225, 225) if piece.isupper() else (20, 20, 20),
        )
    empty = _empty_board()
    captured_removed = reference.copy()
    captured_removed[
        destination[0] * 100 : (destination[0] + 1) * 100,
        destination[1] * 100 : (destination[1] + 1) * 100,
    ] = empty[
        destination[0] * 100 : (destination[0] + 1) * 100,
        destination[1] * 100 : (destination[1] + 1) * 100,
    ]
    tracker = ReferenceMoveTracker()
    tracker.set_reference(reference, _camera_grid(), 0)

    pending = _accepted_update(
        tracker,
        captured_removed,
        game.position,
        game.physical_move_signatures,
    )

    assert pending.state == "pending-capture"
    assert pending.move is None
    assert "from a7" in pending.message
    np.testing.assert_array_equal(tracker.reference, reference)

    completed = captured_removed.copy()
    completed[
        source[0] * 100 : (source[0] + 1) * 100,
        source[1] * 100 : (source[1] + 1) * 100,
    ] = empty[
        source[0] * 100 : (source[0] + 1) * 100,
        source[1] * 100 : (source[1] + 1) * 100,
    ]
    _draw_piece(completed, destination, (225, 225, 225))

    accepted = _accepted_update(
        tracker,
        completed,
        game.position,
        game.physical_move_signatures,
    )

    assert accepted.state == "accepted"
    assert accepted.move == (source, destination)


def test_capture_explains_residuals_that_preserve_authoritative_state() -> None:
    game = LegalGame(
        "r1bqk1nr/pp1n1ppp/2p1p3/1Bb5/4p3/"
        "P1NP1NP1/1PP2P1P/R1BQK2R w KQkq - 0 8"
    )
    signature = next(
        signature
        for signature in game.physical_move_signatures
        if signature.uci == "d3e4"
    )
    colors = {
        (3, 3): PieceColorClassification("empty", 0.01, 0.2),
        (3, 2): PieceColorClassification("black", 0.01, 0.2),
        (7, 7): PieceColorClassification("white", 0.01, 0.2),
    }

    assert _matches_signature_result_state(
        (3, 3),
        signature,
        game.position,
        colors,
    )
    assert _matches_signature_result_state(
        (3, 2),
        signature,
        game.position,
        colors,
    )
    assert _matches_signature_result_state(
        (7, 7),
        signature,
        game.position,
        colors,
    )


def test_legal_capture_is_not_reset_by_whole_board_motion_gate() -> None:
    game = LegalGame("4k3/8/8/8/1n6/P7/8/4K3 w - - 0 1")
    source = (5, 0)
    destination = (4, 1)
    reference = _empty_board()
    for square, piece in game.position.items():
        _draw_piece(
            reference,
            square,
            (225, 225, 225) if piece.isupper() else (20, 20, 20),
        )
    captured = _empty_board()
    for square, piece in game.position.items():
        if square not in (source, destination):
            _draw_piece(
                captured,
                square,
                (225, 225, 225) if piece.isupper() else (20, 20, 20),
            )
    _draw_piece(captured, destination, (225, 225, 225))
    tracker = ReferenceMoveTracker()
    tracker.set_reference(reference, _camera_grid(), 0)
    tracker._previous_motion_image = np.zeros((100, 100), dtype=np.uint8)

    updates = [
        tracker.observe(
            captured,
            _camera_grid(),
            0,
            game.position,
            game.physical_move_signatures,
        )
        for _index in range(3)
    ]

    assert all(update.state != "activity" for update in updates)
    assert updates[-1].state == "accepted"
    assert updates[-1].move == (source, destination)


def test_wrong_side_relocation_is_reported_as_illegal() -> None:
    game = LegalGame("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1")
    reference = _empty_board()
    for square, piece in game.position.items():
        _draw_piece(
            reference,
            square,
            (225, 225, 225) if piece.isupper() else (20, 20, 20),
        )
    source = (3, 3)
    destination = (2, 3)
    changed = reference.copy()
    empty = _empty_board()
    changed[
        source[0] * 100 : (source[0] + 1) * 100,
        source[1] * 100 : (source[1] + 1) * 100,
    ] = empty[
        source[0] * 100 : (source[0] + 1) * 100,
        source[1] * 100 : (source[1] + 1) * 100,
    ]
    _draw_piece(changed, destination, (20, 20, 20))
    tracker = ReferenceMoveTracker()
    tracker.set_reference(reference, _camera_grid(), 0)

    update = _accepted_update(
        tracker,
        changed,
        game.position,
        game.physical_move_signatures,
    )

    assert update.state == "illegal-relocation"
    assert update.move == (source, destination)
    np.testing.assert_array_equal(tracker.reference, reference)


def test_b2_b4_is_illegal_when_black_is_to_move() -> None:
    game = LegalGame(
        "rnbqkbnr/p3pppp/2p5/1p3P2/2PpP3/3P4/PP4PP/"
        "RNBQKBNR b KQkq - 0 1"
    )
    source = (6, 1)
    destination = (4, 1)
    reference = _empty_board()
    for square, piece in game.position.items():
        _draw_piece(
            reference,
            square,
            (225, 225, 225) if piece.isupper() else (20, 20, 20),
        )
    moved = reference.copy()
    moved[
        source[0] * 100 : (source[0] + 1) * 100,
        source[1] * 100 : (source[1] + 1) * 100,
    ] = _empty_board()[
        source[0] * 100 : (source[0] + 1) * 100,
        source[1] * 100 : (source[1] + 1) * 100,
    ]
    _draw_piece(moved, destination, (225, 225, 225))
    tracker = ReferenceMoveTracker()
    tracker.set_reference(reference, _camera_grid(), 0)

    update = _accepted_update(
        tracker,
        moved,
        game.position,
        game.physical_move_signatures,
    )

    assert update.state == "illegal-relocation"
    assert update.move == (source, destination)
    np.testing.assert_array_equal(tracker.reference, reference)


def test_c4_c5_wrong_side_move_ignores_stable_a2_residual() -> None:
    game = LegalGame(
        "rn2kbnr/p3p1pp/b1p2p2/1p3P2/2P1P3/B2P1N2/P3B1PP/"
        "3K3R b kq - 0 1"
    )
    source = (4, 2)
    destination = (3, 2)
    residual = (6, 0)
    reference = _empty_board()
    for square, piece in game.position.items():
        _draw_piece(
            reference,
            square,
            (225, 225, 225) if piece.isupper() else (20, 20, 20),
        )
    moved = reference.copy()
    moved[
        source[0] * 100 : (source[0] + 1) * 100,
        source[1] * 100 : (source[1] + 1) * 100,
    ] = _empty_board()[
        source[0] * 100 : (source[0] + 1) * 100,
        source[1] * 100 : (source[1] + 1) * 100,
    ]
    _draw_piece(moved, destination, (225, 225, 225))
    row, column = residual
    moved[
        row * 100 : (row + 1) * 100,
        column * 100 : (column + 1) * 100,
    ] = cv2.convertScaleAbs(
        moved[
            row * 100 : (row + 1) * 100,
            column * 100 : (column + 1) * 100,
        ],
        alpha=1.0,
        beta=12,
    )
    tracker = ReferenceMoveTracker()
    tracker.set_reference(reference, _camera_grid(), 0)

    update = _accepted_update(
        tracker,
        moved,
        game.position,
        game.physical_move_signatures,
    )

    assert update.state == "illegal-relocation"
    assert update.move == (source, destination)
    assert update.change_scores is not None
    assert update.change_scores[residual] >= 0.05
    np.testing.assert_array_equal(tracker.reference, reference)


def test_illegal_bishop_vertical_move_is_not_reduced_to_removal() -> None:
    game = LegalGame(
        "rn3bnr/pb1kp1pp/2p2p2/1p3P2/2P1P2P/B2P1NP1/"
        "P3B3/3K3R b - - 0 1"
    )
    source = (1, 1)
    destination = (2, 1)
    reference = _empty_board()
    for square, piece in game.position.items():
        _draw_piece(
            reference,
            square,
            (225, 225, 225) if piece.isupper() else (20, 20, 20),
        )
    moved = reference.copy()
    moved[
        source[0] * 100 : (source[0] + 1) * 100,
        source[1] * 100 : (source[1] + 1) * 100,
    ] = _empty_board()[
        source[0] * 100 : (source[0] + 1) * 100,
        source[1] * 100 : (source[1] + 1) * 100,
    ]
    _draw_piece(moved, destination, (20, 20, 20))
    tracker = ReferenceMoveTracker()
    tracker.set_reference(reference, _camera_grid(), 0)

    update = _accepted_update(
        tracker,
        moved,
        game.position,
        game.physical_move_signatures,
    )

    assert update.state == "illegal-relocation"
    assert update.move == (source, destination)
    np.testing.assert_array_equal(tracker.reference, reference)


def test_uncapturable_opponent_removal_is_marked_illegal() -> None:
    game = LegalGame("4k3/8/8/4P3/1n6/8/8/4K3 w - - 0 1")
    removed_square = (4, 1)
    reference = _empty_board()
    for square, piece in game.position.items():
        _draw_piece(
            reference,
            square,
            (225, 225, 225) if piece.isupper() else (20, 20, 20),
        )
    removed = reference.copy()
    empty = _empty_board()
    row, column = removed_square
    removed[
        row * 100 : (row + 1) * 100,
        column * 100 : (column + 1) * 100,
    ] = empty[
        row * 100 : (row + 1) * 100,
        column * 100 : (column + 1) * 100,
    ]
    tracker = ReferenceMoveTracker()
    tracker.set_reference(reference, _camera_grid(), 0)

    update = _accepted_update(
        tracker,
        removed,
        game.position,
        game.physical_move_signatures,
    )

    assert update.state == "illegal-removal"
    assert update.move == (removed_square, removed_square)
    assert "Illegal removal from b4" in update.message
    np.testing.assert_array_equal(tracker.reference, reference)

    restored = _accepted_update(
        tracker,
        reference,
        game.position,
        game.physical_move_signatures,
    )

    assert tracker.rejected_move_is_restored(
        restored,
        (removed_square, removed_square),
    )


def test_b4_removal_after_b4_e5_is_marked_illegal() -> None:
    game = LegalGame(
        "rnbqkbnr/pppp1ppp/8/4p3/1P6/8/P1PPPPPP/"
        "RNBQKBNR w KQkq - 0 2"
    )
    removed_square = (4, 1)
    reference = _empty_board()
    for square, piece in game.position.items():
        _draw_piece(
            reference,
            square,
            (225, 225, 225) if piece.isupper() else (20, 20, 20),
        )
    removed = reference.copy()
    empty = _empty_board()
    row, column = removed_square
    removed[
        row * 100 : (row + 1) * 100,
        column * 100 : (column + 1) * 100,
    ] = empty[
        row * 100 : (row + 1) * 100,
        column * 100 : (column + 1) * 100,
    ]
    tracker = ReferenceMoveTracker()
    tracker.set_reference(reference, _camera_grid(), 0)

    update = _accepted_update(
        tracker,
        removed,
        game.position,
        game.physical_move_signatures,
    )

    assert update.state == "illegal-removal"
    assert update.move == (removed_square, removed_square)
    np.testing.assert_array_equal(tracker.reference, reference)


def test_illegal_game_move_preserves_position_and_reference() -> None:
    source = (7, 6)
    destination = (5, 6)
    reference = _empty_board()
    position = {
        (row, column): piece
        for (row, column), piece in LegalGame().position.items()
    }
    for square in position:
        _draw_piece(reference, square)
    moved = reference.copy()
    moved[
        source[0] * 100 : (source[0] + 1) * 100,
        source[1] * 100 : (source[1] + 1) * 100,
    ] = _empty_board()[
        source[0] * 100 : (source[0] + 1) * 100,
        source[1] * 100 : (source[1] + 1) * 100,
    ]
    _draw_piece(moved, destination)
    tracker = ReferenceMoveTracker()
    tracker.set_reference(reference, _camera_grid(), 0)
    synchronizer = PiecePositionSynchronizer()
    synchronizer.observe(starting_occupied_squares())
    accepted = _accepted_update(tracker, moved, position)

    assert accepted.state == "accepted"
    with pytest.raises(ValueError, match="g1-g3 is not legal"):
        LegalGame().apply_move(source, destination)
    tracker.reset_activity()

    assert synchronizer.position == position
    np.testing.assert_array_equal(tracker.reference, reference)


def test_capture_promotion_that_does_not_resolve_check_is_detected() -> None:
    game = LegalGame("4k3/3P4/8/8/8/8/4p3/5N1K b - - 0 1")
    source = (6, 4)
    destination = (7, 5)
    reference = _empty_board()
    for square, piece in game.position.items():
        _draw_piece(
            reference,
            square,
            (225, 225, 225) if piece.isupper() else (20, 20, 20),
        )
    moved = reference.copy()
    empty = _empty_board()
    for square in (source, destination):
        row, column = square
        moved[
            row * 100 : (row + 1) * 100,
            column * 100 : (column + 1) * 100,
        ] = empty[
            row * 100 : (row + 1) * 100,
            column * 100 : (column + 1) * 100,
        ]
    _draw_piece(moved, destination, (20, 20, 20))
    tracker = ReferenceMoveTracker()
    tracker.set_reference(reference, _camera_grid(), 0)

    accepted = _accepted_update(
        tracker,
        moved,
        game.position,
        game.physical_move_signatures,
    )

    assert accepted.state == "accepted"
    assert accepted.move == (source, destination)
    with pytest.raises(ValueError, match="e2-f1 is not legal for black"):
        game.apply_move(source, destination)
    tracker.reset_activity()
    np.testing.assert_array_equal(tracker.reference, reference)


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


def test_restored_rejected_move_allows_small_source_residual() -> None:
    tracker = ReferenceMoveTracker()
    scores = np.full((8, 8), 0.07, dtype=np.float32)
    scores[3, 3] = 0.087
    update = ReferenceTrackerUpdate(
        "ambiguous",
        None,
        "Ambiguous settled board change.",
        scores,
    )

    assert tracker.rejected_move_is_restored(
        update,
        ((3, 3), (4, 3)),
    )

    destination_changed = scores.copy()
    destination_changed[4, 3] = 0.09
    assert not tracker.rejected_move_is_restored(
        ReferenceTrackerUpdate(
            "ambiguous",
            None,
            "Ambiguous settled board change.",
            destination_changed,
        ),
        ((3, 3), (4, 3)),
    )


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


def test_multi_edge_remap_residual_is_not_a_hand_after_grid_update() -> None:
    reference = _empty_board()
    remapped = reference.copy()
    cv2.rectangle(remapped, (0, 500), (60, 799), (90, 120, 150), -1)
    cv2.rectangle(remapped, (740, 0), (799, 200), (90, 120, 150), -1)

    assert _has_large_obstruction(reference, remapped)
    assert not _has_large_obstruction(
        reference,
        remapped,
        allow_boundary_remap_residuals=True,
    )


def test_boundary_remap_score_does_not_hide_interior_relocation() -> None:
    tracker = ReferenceMoveTracker()
    scores = np.zeros((8, 8), dtype=np.float32)
    scores[2, 4] = 0.215
    scores[3, 4] = 0.153
    scores[5, 0] = 0.125
    reference = _empty_board()

    blocked = tracker._select_move(
        scores,
        {(2, 4): "p"},
        reference,
        reference,
    )
    selected = tracker._select_move(
        scores,
        {(2, 4): "p"},
        reference,
        reference,
        allow_boundary_residuals=True,
    )

    assert blocked is None
    assert selected == ((2, 4), (3, 4))


def test_interior_obstruction_is_not_ignored_after_grid_update() -> None:
    reference = _empty_board()
    obstructed = reference.copy()
    cv2.rectangle(obstructed, (250, 250), (550, 550), (30, 40, 220), -1)
    cv2.rectangle(obstructed, (0, 0), (60, 200), (90, 120, 150), -1)
    cv2.rectangle(obstructed, (740, 600), (799, 799), (90, 120, 150), -1)

    assert _has_large_obstruction(
        reference,
        obstructed,
        allow_boundary_remap_residuals=True,
    )


def test_two_square_hand_region_blocks_intermediate_move() -> None:
    source = (7, 7)
    destination = (7, 6)
    reference = _empty_board()
    _draw_piece(reference, source)
    held = _empty_board()
    _draw_piece(held, destination)
    cv2.rectangle(held, (610, 700), (799, 799), (80, 120, 190), -1)
    tracker = ReferenceMoveTracker()
    tracker.set_reference(reference, _camera_grid(), 0)

    updates = [
        tracker.observe(held, _camera_grid(), 0, {source: "R"})
        for _index in range(4)
    ]

    assert updates[-1].state == "waiting-for-clear"
    assert all(update.move is None for update in updates)


def test_wide_piece_crossing_square_edge_is_not_an_obstruction() -> None:
    source = (7, 4)
    destination = (6, 4)
    reference = _empty_board()
    moved = _empty_board()
    cv2.ellipse(
        reference,
        (480, 750),
        (38, 48),
        0,
        0,
        360,
        (20, 20, 20),
        -1,
        cv2.LINE_AA,
    )
    cv2.ellipse(
        moved,
        (480, 650),
        (38, 48),
        0,
        0,
        360,
        (20, 20, 20),
        -1,
        cv2.LINE_AA,
    )
    tracker = ReferenceMoveTracker()
    tracker.set_reference(reference, _camera_grid(), 0)

    update = _accepted_update(tracker, moved, {source: "K"})

    assert update.state == "accepted"
    assert update.move == (source, destination)


def test_large_released_king_footprint_is_accepted() -> None:
    source = (7, 4)
    destination = (7, 5)
    reference = _empty_board()
    moved = _empty_board()
    cv2.ellipse(
        reference,
        (450, 750),
        (44, 45),
        0,
        0,
        360,
        (20, 20, 20),
        -1,
        cv2.LINE_AA,
    )
    cv2.ellipse(
        moved,
        (550, 750),
        (44, 45),
        0,
        0,
        360,
        (20, 20, 20),
        -1,
        cv2.LINE_AA,
    )
    tracker = ReferenceMoveTracker()
    tracker.set_reference(reference, _camera_grid(), 0)

    update = _accepted_update(tracker, moved, {source: "K"})

    assert update.state == "accepted"
    assert update.move == (source, destination)


def test_single_square_hand_footprint_blocks_move_until_release() -> None:
    source = (7, 7)
    destination = (7, 6)
    reference = _empty_board()
    _draw_piece(reference, source)
    held = _empty_board()
    cv2.rectangle(held, (605, 705), (695, 795), (80, 120, 190), -1)
    tracker = ReferenceMoveTracker()
    tracker.set_reference(reference, _camera_grid(), 0)

    held_updates = [
        tracker.observe(held, _camera_grid(), 0, {source: "R"})
        for _index in range(6)
    ]

    assert any(
        update.state == "waiting-for-clear" for update in held_updates
    )
    assert all(update.state != "accepted" for update in held_updates)
    assert all(update.move is None for update in held_updates)

    released = _empty_board()
    _draw_piece(released, destination)
    accepted = _accepted_update(tracker, released, {source: "R"})

    assert accepted.state == "accepted"
    assert accepted.move == (source, destination)


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
