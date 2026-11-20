from src.chess_camera.chess_position import standard_position
from src.chess_camera.reconciliation import compare_physical_occupancy


def test_matching_occupancy_completes_reconciliation() -> None:
    position = standard_position()

    result = compare_physical_occupancy(
        position,
        frozenset(position),
    )

    assert result.matches
    assert result.mismatch_squares == ()
    assert result.message == "Physical occupancy matches the accepted position."


def test_reconciliation_names_missing_piece_and_unexpected_square() -> None:
    position = standard_position()
    physical = set(position)
    physical.remove((6, 1))
    physical.add((4, 1))

    result = compare_physical_occupancy(
        position,
        frozenset(physical),
    )

    assert not result.matches
    assert result.mismatch_squares == ((6, 1), (4, 1))
    assert result.discrepancies[0].expected_piece == "P"
    assert not result.discrepancies[0].physically_occupied
    assert result.discrepancies[1].expected_piece is None
    assert result.discrepancies[1].physically_occupied
    assert result.message == (
        "Reconciliation required: restore white pawn to b2; remove the "
        "unexpected piece from b4."
    )
