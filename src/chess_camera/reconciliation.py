from dataclasses import dataclass

from .chess_position import PieceCode, Position, Square, square_name

_PIECE_NAMES = {
    "K": "white king",
    "Q": "white queen",
    "R": "white rook",
    "B": "white bishop",
    "N": "white knight",
    "P": "white pawn",
    "k": "black king",
    "q": "black queen",
    "r": "black rook",
    "b": "black bishop",
    "n": "black knight",
    "p": "black pawn",
}


@dataclass(frozen=True)
class OccupancyDiscrepancy:
    square: Square
    expected_piece: PieceCode | None
    physically_occupied: bool


@dataclass(frozen=True)
class ReconciliationResult:
    discrepancies: tuple[OccupancyDiscrepancy, ...]
    message: str

    @property
    def matches(self) -> bool:
        return not self.discrepancies

    @property
    def mismatch_squares(self) -> tuple[Square, ...]:
        return tuple(discrepancy.square for discrepancy in self.discrepancies)


def compare_physical_occupancy(
    position: Position,
    physically_occupied: frozenset[Square],
) -> ReconciliationResult:
    expected = frozenset(position)
    missing = tuple(sorted(expected - physically_occupied))
    unexpected = tuple(sorted(physically_occupied - expected))
    discrepancies = tuple(
        OccupancyDiscrepancy(square, position[square], False)
        for square in missing
    ) + tuple(
        OccupancyDiscrepancy(square, None, True)
        for square in unexpected
    )
    if not discrepancies:
        return ReconciliationResult(
            (),
            "Physical occupancy matches the accepted position.",
        )
    instructions = [
        f"restore {_PIECE_NAMES[position[square]]} to {square_name(square)}"
        for square in missing
    ]
    instructions.extend(
        f"remove the unexpected piece from {square_name(square)}"
        for square in unexpected
    )
    return ReconciliationResult(
        discrepancies,
        "Reconciliation required: " + "; ".join(instructions) + ".",
    )
