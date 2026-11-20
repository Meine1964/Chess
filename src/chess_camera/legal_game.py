from dataclasses import dataclass
from typing import Literal, cast

import chess

from .chess_position import PieceCode, Position, Square, square_name

PromotionPiece = Literal["q", "r", "b", "n"]
GameMode = Literal["diagnostic", "offline-game", "lichess-game"]


@dataclass(frozen=True)
class GameStatus:
    turn: Literal["white", "black"]
    is_check: bool
    is_checkmate: bool
    is_stalemate: bool
    is_insufficient_material: bool
    can_claim_draw: bool
    is_game_over: bool
    result: str


@dataclass(frozen=True)
class LegalMoveResult:
    uci: str
    san: str
    position: Position
    fen: str
    status: GameStatus


class LegalGame:
    def __init__(self, fen: str = chess.STARTING_FEN) -> None:
        try:
            self._board = chess.Board(fen)
        except ValueError as error:
            raise ValueError(f"Invalid chess position: {error}") from error

    @property
    def fen(self) -> str:
        return self._board.fen()

    @property
    def position(self) -> Position:
        position: Position = {}
        for chess_square, piece in self._board.piece_map().items():
            position[_ui_square(chess_square)] = _piece_code(piece.symbol())
        return position

    @property
    def legal_uci_moves(self) -> tuple[str, ...]:
        return tuple(sorted(move.uci() for move in self._board.legal_moves))

    def legal_destinations(self, source: Square) -> tuple[Square, ...]:
        source_square = _chess_square(source)
        return tuple(
            sorted(
                {
                    _ui_square(move.to_square)
                    for move in self._board.legal_moves
                    if move.from_square == source_square
                }
            )
        )

    @property
    def status(self) -> GameStatus:
        outcome = self._board.outcome(claim_draw=True)
        return GameStatus(
            turn="white" if self._board.turn == chess.WHITE else "black",
            is_check=self._board.is_check(),
            is_checkmate=self._board.is_checkmate(),
            is_stalemate=self._board.is_stalemate(),
            is_insufficient_material=self._board.is_insufficient_material(),
            can_claim_draw=self._board.can_claim_draw(),
            is_game_over=outcome is not None,
            result="*" if outcome is None else outcome.result(),
        )

    def reset(self) -> None:
        self._board.reset()

    def apply_move(
        self,
        source: Square,
        destination: Square,
        promotion: PromotionPiece | None = None,
    ) -> LegalMoveResult:
        move = self._matching_legal_move(source, destination, promotion)
        san = self._board.san(move)
        self._board.push(move)
        return LegalMoveResult(
            move.uci(),
            san,
            self.position,
            self.fen,
            self.status,
        )

    def rollback(self) -> LegalMoveResult:
        if not self._board.move_stack:
            raise ValueError("No legal move is available to roll back.")
        move = self._board.pop()
        san = self._board.san(move)
        return LegalMoveResult(
            move.uci(),
            san,
            self.position,
            self.fen,
            self.status,
        )

    def _matching_legal_move(
        self,
        source: Square,
        destination: Square,
        promotion: PromotionPiece | None,
    ) -> chess.Move:
        source_square = _chess_square(source)
        destination_square = _chess_square(destination)
        candidates = [
            move
            for move in self._board.legal_moves
            if move.from_square == source_square
            and move.to_square == destination_square
        ]
        if not candidates:
            raise ValueError(
                f"{square_name(source)}-{square_name(destination)} is not "
                f"legal for {self.status.turn}."
            )
        if any(move.promotion is not None for move in candidates):
            if promotion is None:
                raise ValueError(
                    f"Promotion choice is required for "
                    f"{square_name(source)}-{square_name(destination)}."
                )
            promotion_type = {
                "q": chess.QUEEN,
                "r": chess.ROOK,
                "b": chess.BISHOP,
                "n": chess.KNIGHT,
            }[promotion]
            candidates = [
                move
                for move in candidates
                if move.promotion == promotion_type
            ]
            if not candidates:
                raise ValueError("The selected promotion is not legal.")
        elif promotion is not None:
            raise ValueError("Promotion was supplied for a non-promotion move.")
        if len(candidates) != 1:
            raise ValueError("The legal move candidate is ambiguous.")
        return candidates[0]


def _chess_square(square: Square) -> chess.Square:
    row, column = square
    if not 0 <= row < 8 or not 0 <= column < 8:
        raise ValueError("Square must be inside the 8x8 board.")
    return chess.square(column, 7 - row)


def _ui_square(square: chess.Square) -> Square:
    return 7 - chess.square_rank(square), chess.square_file(square)


def _piece_code(symbol: str) -> PieceCode:
    if symbol not in "KQRBNPkqrbnp":
        raise ValueError(f"Unknown chess piece symbol: {symbol}")
    return cast(PieceCode, symbol)
