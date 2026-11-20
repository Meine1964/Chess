from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, cast

import chess

from .chess_position import PieceCode, Position, Square, square_name

PromotionPiece = Literal["q", "r", "b", "n"]
GameMode = Literal["diagnostic", "offline-game", "lichess-game"]
PhysicalMoveKind = Literal[
    "ordinary",
    "capture",
    "castling",
    "en-passant",
    "promotion",
    "capture-promotion",
]


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


@dataclass(frozen=True)
class PhysicalSquareChange:
    square: Square
    before: PieceCode | None
    after: PieceCode | None


@dataclass(frozen=True)
class LegalMoveSignature:
    uci: str
    san: str
    kind: PhysicalMoveKind
    source: Square
    destination: Square
    promotion: PromotionPiece | None
    changes: tuple[PhysicalSquareChange, ...]


class LegalGame:
    def __init__(self, fen: str = chess.STARTING_FEN) -> None:
        try:
            self._board = chess.Board(fen)
        except ValueError as error:
            raise ValueError(f"Invalid chess position: {error}") from error
        self._signature_cache: tuple[LegalMoveSignature, ...] | None = None
        self._physical_signature_cache: (
            tuple[LegalMoveSignature, ...] | None
        ) = None

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

    @property
    def legal_move_signatures(self) -> tuple[LegalMoveSignature, ...]:
        if self._signature_cache is None:
            self._signature_cache = self._physical_signatures(
                self._board.legal_moves
            )
        return self._signature_cache

    @property
    def physical_move_signatures(self) -> tuple[LegalMoveSignature, ...]:
        if self._physical_signature_cache is None:
            self._physical_signature_cache = self._physical_signatures(
                self._board.pseudo_legal_moves
            )
        return self._physical_signature_cache

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
        self._invalidate_signature_caches()

    def apply_move(
        self,
        source: Square,
        destination: Square,
        promotion: PromotionPiece | None = None,
    ) -> LegalMoveResult:
        move = self._matching_legal_move(source, destination, promotion)
        san = self._board.san(move)
        self._board.push(move)
        self._invalidate_signature_caches()
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
        self._invalidate_signature_caches()
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
            promotion_type = {
                "q": chess.QUEEN,
                "r": chess.ROOK,
                "b": chess.BISHOP,
                "n": chess.KNIGHT,
            }["q" if promotion is None else promotion]
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

    def _physical_signatures(
        self,
        moves: Iterable[chess.Move],
    ) -> tuple[LegalMoveSignature, ...]:
        return tuple(
            self._physical_signature(move)
            for move in sorted(
                (
                    candidate
                    for candidate in moves
                    if candidate.promotion in (None, chess.QUEEN)
                ),
                key=lambda candidate: candidate.uci(),
            )
        )

    def _invalidate_signature_caches(self) -> None:
        self._signature_cache = None
        self._physical_signature_cache = None

    def _physical_signature(
        self,
        move: chess.Move,
    ) -> LegalMoveSignature:
        before = self._board.piece_map()
        after_board = self._board.copy(stack=False)
        san = self._board.san(move)
        is_capture = self._board.is_capture(move)
        is_castling = self._board.is_castling(move)
        is_en_passant = self._board.is_en_passant(move)
        after_board.push(move)
        after = after_board.piece_map()
        source = _ui_square(move.from_square)
        destination = _ui_square(move.to_square)
        changed_squares = {
            square
            for square in set(before) | set(after)
            if before.get(square) != after.get(square)
        }
        ordered_squares = [
            move.from_square,
            move.to_square,
            *sorted(
                changed_squares - {move.from_square, move.to_square},
                key=_ui_square,
            ),
        ]
        changes = tuple(
            PhysicalSquareChange(
                _ui_square(square),
                _optional_piece_code(before.get(square)),
                _optional_piece_code(after.get(square)),
            )
            for square in ordered_squares
            if square in changed_squares
        )
        promotion = _promotion_piece(move.promotion)
        if is_castling:
            kind: PhysicalMoveKind = "castling"
        elif is_en_passant:
            kind = "en-passant"
        elif promotion is not None and is_capture:
            kind = "capture-promotion"
        elif promotion is not None:
            kind = "promotion"
        elif is_capture:
            kind = "capture"
        else:
            kind = "ordinary"
        return LegalMoveSignature(
            move.uci(),
            san,
            kind,
            source,
            destination,
            promotion,
            changes,
        )


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


def _optional_piece_code(piece: chess.Piece | None) -> PieceCode | None:
    return None if piece is None else _piece_code(piece.symbol())


def _promotion_piece(piece_type: chess.PieceType | None) -> PromotionPiece | None:
    if piece_type is None:
        return None
    return {
        chess.QUEEN: "q",
        chess.ROOK: "r",
        chess.BISHOP: "b",
        chess.KNIGHT: "n",
    }[piece_type]
