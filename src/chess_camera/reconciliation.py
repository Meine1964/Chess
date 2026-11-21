from dataclasses import dataclass
from typing import Mapping

from .chess_position import PieceCode, Position, Square, square_name
from .piece_color_classifier import PieceColorLabel

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
    observed_color: PieceColorLabel | None = None


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
    observed_colors: Mapping[Square, PieceColorLabel] | None = None,
) -> ReconciliationResult:
    effective_occupied = set(physically_occupied)
    if observed_colors is not None:
        for square, label in observed_colors.items():
            if label in ("white", "black"):
                effective_occupied.add(square)
            elif label == "empty":
                effective_occupied.discard(square)
    physically_occupied = frozenset(effective_occupied)
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
    wrong_colors: tuple[Square, ...] = ()
    if observed_colors is not None:
        wrong_colors = tuple(
            square
            for square in sorted(expected & physically_occupied)
            if (
                observed_colors.get(square) in ("white", "black")
                and observed_colors[square]
                != ("white" if position[square].isupper() else "black")
            )
        )
        discrepancies += tuple(
            OccupancyDiscrepancy(
                square,
                position[square],
                True,
                observed_colors[square],
            )
            for square in wrong_colors
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
    instructions.extend(
        f"replace the {observed_colors[square]} piece on "
        f"{square_name(square)} with {_PIECE_NAMES[position[square]]}"
        for square in wrong_colors
        if observed_colors is not None
    )
    return ReconciliationResult(
        discrepancies,
        "Reconciliation required: " + "; ".join(instructions) + ".",
    )
