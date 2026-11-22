from src.chess_camera.chess_position import standard_position
from src.chess_camera.legal_game import LegalGame
from src.chess_camera.reconciliation import (
    compare_physical_occupancy,
    matching_legal_successors,
)


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


def test_confident_piece_colors_correct_false_occupancy_labels() -> None:
    position = {
        (0, 0): "r",
        (6, 2): "P",
        (7, 7): "R",
    }

    result = compare_physical_occupancy(
        position,
        frozenset({(0, 0), (3, 3)}),
        {
            (0, 0): "black",
            (3, 3): "empty",
            (6, 2): "white",
            (7, 7): "white",
        },
    )

    assert result.matches


def test_reconciliation_finds_unique_immediate_recapture() -> None:
    game = LegalGame("4k3/3b3p/8/8/8/8/P3B3/4K3 b - - 0 1")
    opponent = game.apply_move((1, 3), (3, 1))
    reply_game = LegalGame(opponent.fen)
    reply = reply_game.apply_move((6, 4), (3, 1))

    matches = matching_legal_successors(
        opponent.fen,
        frozenset(reply.position),
        {(3, 1): "white", (6, 4): "empty"},
        allow_promotion=True,
    )

    assert tuple(match.signature.uci for match in matches) == ("e2b5",)
    assert matches[0].result.fen == reply.fen


def test_reconciliation_does_not_infer_unselected_promotion() -> None:
    game = LegalGame("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    promoted = LegalGame(game.fen).apply_move((1, 0), (0, 0), "q")

    matches = matching_legal_successors(
        game.fen,
        frozenset(promoted.position),
        {(0, 0): "white", (1, 0): "empty"},
        allow_promotion=False,
    )

    assert matches == ()
