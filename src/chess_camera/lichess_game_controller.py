from dataclasses import dataclass
from typing import Literal, cast

import chess

from .chess_position import Position, Square, square_name
from .legal_game import (
    LegalGame,
    LegalMoveResult,
    LegalMoveSignature,
    PromotionPiece,
)
from .lichess_client import PlayerColor

LichessSynchronizationState = Literal[
    "attaching",
    "reconciliation-required",
    "synchronized",
    "guidance",
    "correction",
    "game-complete",
]


@dataclass(frozen=True)
class LichessSynchronizationUpdate:
    state: LichessSynchronizationState
    appended_moves: tuple[str, ...]
    pending_moves: tuple[str, ...]
    message: str


class LichessGameController:
    def __init__(
        self,
        game_id: str,
        initial_fen: str,
        player_color: PlayerColor,
        opponent_name: str,
        *,
        physical_fen: str | None = None,
        maximum_initial_queue: int = 2,
    ) -> None:
        if not game_id.strip():
            raise ValueError("A Lichess game ID is required.")
        if maximum_initial_queue < 0:
            raise ValueError("The initial queue limit cannot be negative.")
        self._game_id = game_id
        normalized_initial_fen = (
            chess.STARTING_FEN if initial_fen == "startpos" else initial_fen
        )
        self._initial_fen = LegalGame(normalized_initial_fen).fen
        self._player_color = player_color
        self._opponent_name = opponent_name
        self._server_game = LegalGame(self._initial_fen)
        self._physical_game = LegalGame(
            self._initial_fen if physical_fen is None else physical_fen
        )
        self._maximum_initial_queue = maximum_initial_queue
        self._server_moves: tuple[str, ...] = ()
        self._pending_moves: list[str] = []
        self._state: LichessSynchronizationState = "attaching"
        self._attached = False

    @property
    def game_id(self) -> str:
        return self._game_id

    @property
    def state(self) -> LichessSynchronizationState:
        return self._state

    @property
    def server_moves(self) -> tuple[str, ...]:
        return self._server_moves

    @property
    def pending_moves(self) -> tuple[str, ...]:
        return tuple(self._pending_moves)

    @property
    def server_position(self) -> Position:
        return self._server_game.position

    @property
    def server_fen(self) -> str:
        return self._server_game.fen

    @property
    def physical_position(self) -> Position:
        return self._physical_game.position

    @property
    def physical_fen(self) -> str:
        return self._physical_game.fen

    @property
    def expected_signature(self) -> LegalMoveSignature | None:
        if not self._pending_moves:
            return None
        expected_uci = self._pending_moves[0]
        return next(
            (
                signature
                for signature in self._physical_game.legal_move_signatures
                if signature.uci == expected_uci
                and signature.kind in ("ordinary", "capture", "castling")
            ),
            None,
        )

    @property
    def pending_result(self) -> LegalMoveResult | None:
        signature = self.expected_signature
        if signature is None:
            return None
        preview = LegalGame(self._physical_game.fen)
        return preview.apply_move(
            signature.source,
            signature.destination,
            signature.promotion,
        )

    @property
    def local_move_signatures(self) -> tuple[LegalMoveSignature, ...]:
        if (
            self._state != "synchronized"
            or self._server_game.fen != self._physical_game.fen
            or self._server_game.status.turn != self._player_color
        ):
            return ()
        return tuple(
            signature
            for signature in self._physical_game.legal_move_signatures
            if signature.kind in ("ordinary", "capture", "castling")
        )

    def prepare_local_move(
        self,
        move: tuple[Square, Square],
    ) -> tuple[str, LegalMoveResult]:
        signature = next(
            (
                candidate
                for candidate in self.local_move_signatures
                if (candidate.source, candidate.destination) == move
            ),
            None,
        )
        if signature is None:
            raise ValueError(
                "The physical move is not a supported local Lichess move."
            )
        preview = LegalGame(self._physical_game.fen)
        return (
            signature.uci,
            preview.apply_move(
                signature.source,
                signature.destination,
                signature.promotion,
            ),
        )

    def commit_local_move(
        self,
        move: tuple[Square, Square],
    ) -> LegalMoveResult:
        uci, _preview = self.prepare_local_move(move)
        server_result = _apply_uci(self._server_game, uci)
        physical_result = _apply_uci(self._physical_game, uci)
        if server_result.fen != physical_result.fen:
            raise ValueError(
                "Lichess server and physical move commits diverged."
            )
        self._server_moves = (*self._server_moves, uci)
        self._state = (
            "game-complete"
            if self._server_game.status.is_game_over
            else "synchronized"
        )
        return physical_result

    def require_reconciliation(
        self,
        message: str,
    ) -> LichessSynchronizationUpdate:
        self._state = "reconciliation-required"
        return self._update((), message)

    def accept_server_history(
        self,
        moves: tuple[str, ...],
    ) -> LichessSynchronizationUpdate:
        if self._state == "reconciliation-required":
            return self._update(
                (),
                "Lichess synchronization requires physical reconciliation.",
            )
        if self._attached and moves[: len(self._server_moves)] != (
            self._server_moves
        ):
            self._state = "reconciliation-required"
            return self._update(
                (),
                "Lichess move history was shortened or rewritten; "
                "physical reconciliation is required.",
            )
        appended = moves[len(self._server_moves) :]
        try:
            server_game = LegalGame(self._initial_fen)
            normalized_moves = tuple(
                _apply_uci(server_game, uci).uci for uci in moves
            )
        except ValueError as error:
            self._state = "reconciliation-required"
            return self._update(
                (),
                f"Lichess move history is invalid: {error}",
            )
        if normalized_moves != moves:
            self._state = "reconciliation-required"
            return self._update(
                (),
                "Lichess move history changed during normalization; "
                "physical reconciliation is required.",
            )
        self._server_game = server_game
        if not self._attached:
            self._attached = True
            self._server_moves = moves
            if self._physical_game.fen == self._server_game.fen:
                self._state = (
                    "game-complete"
                    if self._server_game.status.is_game_over
                    else "synchronized"
                )
                return self._update((), self._synchronized_message())
            if (
                self._physical_game.fen == self._initial_fen
                and len(moves) <= self._maximum_initial_queue
            ):
                self._pending_moves.extend(moves)
                return self._activate_pending(moves)
            self._state = "reconciliation-required"
            return self._update(
                (),
                "The active Lichess position does not match the accepted "
                "physical position; start a fresh game or reconcile.",
            )
        self._server_moves = moves
        self._pending_moves.extend(appended)
        if not appended:
            return self._update((), self._current_message())
        return self._activate_pending(appended)

    def confirm_expected_move(
        self,
        move: tuple[Square, Square],
    ) -> LichessSynchronizationUpdate:
        signature = self.expected_signature
        if signature is None:
            self._state = "reconciliation-required"
            return self._update(
                (),
                "The pending Lichess move is not supported by the ordinary, "
                "capture, and castling synchronization slice.",
            )
        expected = (signature.source, signature.destination)
        if move != expected:
            self._state = "correction"
            return self._update(
                (),
                f"Expected {square_name(expected[0])}-"
                f"{square_name(expected[1])}; restore the physical board "
                "or complete that move.",
            )
        _apply_uci(self._physical_game, self._pending_moves.pop(0))
        return self._activate_pending(())

    def note_correction_restored(self) -> LichessSynchronizationUpdate:
        if self._state != "correction":
            return self._update((), self._current_message())
        return self._activate_pending(())

    def _activate_pending(
        self,
        appended: tuple[str, ...],
    ) -> LichessSynchronizationUpdate:
        if not self._pending_moves:
            self._state = (
                "game-complete"
                if self._server_game.status.is_game_over
                else "synchronized"
            )
            return self._update(appended, self._synchronized_message())
        signature = self.expected_signature
        if signature is None:
            self._state = "reconciliation-required"
            return self._update(
                appended,
                "The next Lichess move requires a special-move Phase 7 "
                "slice that is not implemented yet.",
            )
        self._state = "guidance"
        actor = (
            "your Lichess move"
            if _piece_color(signature) == self._player_color
            else f"{self._opponent_name}'s move"
        )
        return self._update(
            appended,
            f"Reproduce {actor} {square_name(signature.source)}-"
            f"{square_name(signature.destination)} on the physical board.",
        )

    def _synchronized_message(self) -> str:
        if self._server_game.status.is_game_over:
            return (
                "Lichess and physical positions are synchronized; "
                f"game result {self._server_game.status.result}."
            )
        return "Lichess and physical positions are synchronized."

    def _current_message(self) -> str:
        if self._state == "correction":
            return "Restore the physical board or complete the expected move."
        if self._state == "guidance":
            signature = self.expected_signature
            if signature is not None:
                return (
                    f"Reproduce {square_name(signature.source)}-"
                    f"{square_name(signature.destination)} physically."
                )
        if self._state == "reconciliation-required":
            return "Physical reconciliation is required."
        return self._synchronized_message()

    def _update(
        self,
        appended: tuple[str, ...],
        message: str,
    ) -> LichessSynchronizationUpdate:
        return LichessSynchronizationUpdate(
            self._state,
            appended,
            tuple(self._pending_moves),
            message,
        )


def _apply_uci(game: LegalGame, uci: str) -> LegalMoveResult:
    try:
        move = chess.Move.from_uci(uci)
    except ValueError as error:
        raise ValueError(f"Invalid UCI move '{uci}'.") from error
    source = _ui_square(move.from_square)
    destination = _ui_square(move.to_square)
    promotion: PromotionPiece | None = (
        None
        if move.promotion is None
        else cast(PromotionPiece, chess.piece_symbol(move.promotion))
    )
    return game.apply_move(source, destination, promotion)


def _ui_square(chess_square: int) -> Square:
    return 7 - chess.square_rank(chess_square), chess.square_file(chess_square)


def _piece_color(signature: LegalMoveSignature) -> PlayerColor:
    moving_piece = next(
        change.before
        for change in signature.changes
        if change.square == signature.source
    )
    if moving_piece is None:
        raise ValueError("A move signature source must contain a piece.")
    return "white" if moving_piece.isupper() else "black"
