from dataclasses import dataclass
from collections.abc import Mapping
from typing import Literal, TypeAlias

Square: TypeAlias = tuple[int, int]
PieceCode = Literal[
    "K",
    "Q",
    "R",
    "B",
    "N",
    "P",
    "k",
    "q",
    "r",
    "b",
    "n",
    "p",
]
Position: TypeAlias = dict[Square, PieceCode]
SynchronizationState = Literal[
    "uninitialized",
    "synchronized",
    "moved",
    "ambiguous",
]


@dataclass(frozen=True)
class PositionUpdate:
    state: SynchronizationState
    position: Position
    move: tuple[Square, Square] | None
    message: str


def standard_position() -> Position:
    position: Position = {}
    black_back_rank: tuple[PieceCode, ...] = (
        "r",
        "n",
        "b",
        "q",
        "k",
        "b",
        "n",
        "r",
    )
    white_back_rank: tuple[PieceCode, ...] = (
        "R",
        "N",
        "B",
        "Q",
        "K",
        "B",
        "N",
        "R",
    )
    for column, piece in enumerate(black_back_rank):
        position[(0, column)] = piece
        position[(1, column)] = "p"
        position[(6, column)] = "P"
        position[(7, column)] = white_back_rank[column]
    return position


def starting_occupied_squares() -> frozenset[Square]:
    return frozenset(standard_position())


def piece_asset_filename(piece: PieceCode) -> str:
    color = "w" if piece.isupper() else "b"
    return f"{color}{piece.upper()}.svg"


def square_name(square: Square) -> str:
    row, column = square
    if not 0 <= row < 8 or not 0 <= column < 8:
        raise ValueError("Square must be inside the 8x8 board.")
    return f"{chr(ord('a') + column)}{8 - row}"


class PiecePositionSynchronizer:
    def __init__(self) -> None:
        self._position: Position | None = None
        self._occupied_baseline: frozenset[Square] | None = None

    @property
    def position(self) -> Position:
        return {} if self._position is None else self._position.copy()

    @property
    def initialized(self) -> bool:
        return self._position is not None

    def reset(self) -> None:
        self._position = None
        self._occupied_baseline = None

    def restore(self, position: Mapping[Square, PieceCode]) -> None:
        restored: Position = {}
        for square, piece in position.items():
            _validate_squares(frozenset({square}))
            if piece not in frozenset("KQRBNPkqrbnp"):
                raise ValueError(f"Unknown piece code: {piece}")
            restored[square] = piece
        if not restored:
            raise ValueError("A restored piece position cannot be empty.")
        self._position = restored
        self._occupied_baseline = frozenset(restored)

    def accept_move(
        self,
        source: Square,
        destination: Square,
    ) -> PositionUpdate:
        _validate_squares(frozenset({source, destination}))
        if self._position is None:
            raise ValueError("Piece position is not initialized.")
        piece = self._position.get(source)
        if piece is None:
            raise ValueError(
                f"No known UI piece is present on {square_name(source)}."
            )
        if destination in self._position:
            raise ValueError(
                f"{square_name(destination)} is already occupied in the UI."
            )
        updated = self._position.copy()
        del updated[source]
        updated[destination] = piece
        self._position = updated
        self._occupied_baseline = frozenset(updated)
        return PositionUpdate(
            "moved",
            updated.copy(),
            (source, destination),
            f"Moved {square_name(source)}-{square_name(destination)}.",
        )

    def accept_authoritative_move(
        self,
        source: Square,
        destination: Square,
        position: Mapping[Square, PieceCode],
    ) -> PositionUpdate:
        _validate_squares(frozenset({source, destination}))
        if self._position is None:
            raise ValueError("Piece position is not initialized.")
        moving_piece = self._position.get(source)
        if moving_piece is None:
            raise ValueError(
                f"No known UI piece is present on {square_name(source)}."
            )
        updated = _validated_position(position)
        if source in updated:
            raise ValueError(
                f"The authoritative move did not vacate {square_name(source)}."
            )
        destination_piece = updated.get(destination)
        if destination_piece is None:
            raise ValueError(
                "The authoritative move did not occupy "
                f"{square_name(destination)}."
            )
        if moving_piece.isupper() != destination_piece.isupper():
            raise ValueError(
                "The authoritative destination piece has the wrong color."
            )
        self._position = updated
        self._occupied_baseline = frozenset(updated)
        return PositionUpdate(
            "moved",
            updated.copy(),
            (source, destination),
            f"Accepted {square_name(source)}-{square_name(destination)}.",
        )

    def observe(
        self,
        occupied_squares: frozenset[Square],
    ) -> PositionUpdate:
        _validate_squares(occupied_squares)
        if self._position is None or self._occupied_baseline is None:
            if occupied_squares != starting_occupied_squares():
                return PositionUpdate(
                    "uninitialized",
                    {},
                    None,
                    "Physical occupancy is not the standard starting position.",
                )
            self._position = standard_position()
            self._occupied_baseline = occupied_squares
            return PositionUpdate(
                "synchronized",
                self._position.copy(),
                None,
                "Piece position initialized from the standard setup.",
            )

        removed = self._occupied_baseline - occupied_squares
        added = occupied_squares - self._occupied_baseline
        if not removed and not added:
            return PositionUpdate(
                "synchronized",
                self._position.copy(),
                None,
                "Physical and UI piece positions are synchronized.",
            )
        return PositionUpdate(
            "ambiguous",
            self._position.copy(),
            None,
            _ambiguous_message(removed, added),
        )


def _validate_squares(squares: frozenset[Square]) -> None:
    for row, column in squares:
        if not 0 <= row < 8 or not 0 <= column < 8:
            raise ValueError("Occupied squares must be inside the 8x8 board.")


def _validated_position(
    position: Mapping[Square, PieceCode],
) -> Position:
    validated: Position = {}
    for square, piece in position.items():
        _validate_squares(frozenset({square}))
        if piece not in frozenset("KQRBNPkqrbnp"):
            raise ValueError(f"Unknown piece code: {piece}")
        validated[square] = piece
    if not validated:
        raise ValueError("An authoritative piece position cannot be empty.")
    return validated


def _ambiguous_message(
    removed: frozenset[Square],
    added: frozenset[Square],
) -> str:
    removed_names = ", ".join(sorted(square_name(square) for square in removed))
    added_names = ", ".join(sorted(square_name(square) for square in added))
    return (
        "Ambiguous physical change: "
        f"removed [{removed_names or 'none'}], "
        f"added [{added_names or 'none'}]."
    )
