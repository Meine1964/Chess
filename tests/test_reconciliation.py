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


def test_reconciliation_reports_only_confident_wrong_piece_color() -> None:
    position = standard_position()

    result = compare_physical_occupancy(
        position,
        frozenset(position),
        {
            (6, 0): "black",
            (6, 1): "uncertain",
            (6, 2): "white",
        },
    )

    assert result.mismatch_squares == ((6, 0),)
    assert result.discrepancies[0].observed_color == "black"
    assert result.message == (
        "Reconciliation required: replace the black piece on a2 with "
        "white pawn."
    )


def test_same_color_piece_identity_is_accepted_by_occupancy() -> None:
    position = standard_position()
    position[(6, 0)] = "R"

    result = compare_physical_occupancy(
        position,
        frozenset(position),
        {(6, 0): "white"},
    )

    assert result.matches
