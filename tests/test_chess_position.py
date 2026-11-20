from src.chess_camera.chess_position import (
    PiecePositionSynchronizer,
    piece_asset_filename,
    standard_position,
    starting_occupied_squares,
)
from src.chess_camera.legal_game import LegalGame


def test_standard_position_contains_all_expected_pieces() -> None:
    position = standard_position()

    assert len(position) == 32
    assert [position[(0, column)] for column in range(8)] == list(
        "rnbqkbnr"
    )
    assert [position[(1, column)] for column in range(8)] == ["p"] * 8
    assert [position[(6, column)] for column in range(8)] == ["P"] * 8
    assert [position[(7, column)] for column in range(8)] == list(
        "RNBQKBNR"
    )


def test_piece_codes_map_to_cburnett_asset_names() -> None:
    assert piece_asset_filename("K") == "wK.svg"
    assert piece_asset_filename("p") == "bP.svg"


def test_synchronizer_moves_one_piece_from_source_to_destination() -> None:
    synchronizer = PiecePositionSynchronizer()
    starting = starting_occupied_squares()
    initialized = synchronizer.observe(starting)

    moved = synchronizer.accept_move((6, 4), (4, 4))

    assert initialized.state == "synchronized"
    assert moved.state == "moved"
    assert moved.move == ((6, 4), (4, 4))
    assert (6, 4) not in moved.position
    assert moved.position[(4, 4)] == "P"
    assert moved.message == "Moved e2-e4."


def test_synchronizer_diagnostic_move_does_not_require_legality() -> None:
    synchronizer = PiecePositionSynchronizer()
    synchronizer.observe(starting_occupied_squares())

    moved = synchronizer.accept_move((7, 6), (5, 6))

    assert moved.move == ((7, 6), (5, 6))
    assert moved.position[(5, 6)] == "N"


def test_synchronizer_commits_authoritative_capture_position() -> None:
    game = LegalGame("4k3/8/8/8/1n6/P7/8/4K3 w - - 0 1")
    synchronizer = PiecePositionSynchronizer()
    synchronizer.restore(game.position)
    capture = game.apply_move((5, 0), (4, 1))

    moved = synchronizer.accept_authoritative_move(
        (5, 0),
        (4, 1),
        capture.position,
    )

    assert moved.move == ((5, 0), (4, 1))
    assert (5, 0) not in moved.position
    assert moved.position[(4, 1)] == "P"
    assert len(moved.position) == 3


def test_synchronizer_rejects_source_only_capture_evidence() -> None:
    synchronizer = PiecePositionSynchronizer()
    starting = starting_occupied_squares()
    initialized = synchronizer.observe(starting)

    ambiguous = synchronizer.observe(
        frozenset(starting - {(6, 4)})
    )

    assert ambiguous.state == "ambiguous"
    assert ambiguous.move is None
    assert ambiguous.position == initialized.position
    assert "removed [e2], added [none]" in ambiguous.message


def test_synchronizer_never_mutates_position_from_absolute_occupancy() -> None:
    synchronizer = PiecePositionSynchronizer()
    starting = starting_occupied_squares()
    initialized = synchronizer.observe(starting)

    source_only = synchronizer.observe(
        frozenset(starting - {(6, 4)})
    )
    unrelated_changes = synchronizer.observe(
        frozenset((starting - {(6, 4), (1, 0)}) | {(4, 4), (3, 0)})
    )
    restored = synchronizer.observe(starting)

    assert synchronizer.initialized
    assert source_only.state == "ambiguous"
    assert unrelated_changes.state == "ambiguous"
    assert restored.state == "synchronized"
    assert source_only.position == initialized.position
    assert unrelated_changes.position == initialized.position
    assert restored.position == initialized.position

    one_source_one_destination = synchronizer.observe(
        frozenset((starting - {(6, 4)}) | {(4, 4)})
    )

    assert one_source_one_destination.state == "ambiguous"
    assert one_source_one_destination.position == initialized.position


def test_synchronizer_requires_starting_position_for_initialization() -> None:
    synchronizer = PiecePositionSynchronizer()

    update = synchronizer.observe(frozenset({(0, 0)}))

    assert update.state == "uninitialized"
    assert update.position == {}
    assert not synchronizer.initialized


def test_synchronizer_restores_accepted_position() -> None:
    synchronizer = PiecePositionSynchronizer()
    restored = standard_position()
    piece = restored.pop((6, 4))
    restored[(4, 4)] = piece

    synchronizer.restore(restored)
    update = synchronizer.observe(frozenset(restored))

    assert synchronizer.initialized
    assert update.state == "synchronized"
    assert update.position == restored
