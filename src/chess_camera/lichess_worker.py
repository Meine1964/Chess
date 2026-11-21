from collections.abc import Callable, Iterator
from dataclasses import replace
from threading import Event, Lock
from time import monotonic, time
from typing import Literal

import chess
from PySide6.QtCore import QObject, Signal, Slot

from .lichess_client import (
    AccountGameEvent,
    AiChallengeColor,
    GameStreamEvent,
    LichessConnectionError,
    LichessError,
    LichessGameFull,
    LichessGameState,
    LichessHTTPError,
    LichessPlayer,
    LichessPollingSnapshot,
    LichessProtocolError,
    LichessReadOnlyClient,
    OngoingGame,
    PlayerColor,
    reconnect_delays,
)

_POLLING_INTERVAL_SECONDS = 5


def _elapsed_clock_state(
    state: LichessGameState,
    elapsed_ms: int,
) -> LichessGameState:
    if state.result != "*" or elapsed_ms <= 0:
        return state
    if state.turn == "white":
        return replace(
            state,
            white_time_ms=max(0, state.white_time_ms - elapsed_ms),
        )
    return replace(
        state,
        black_time_ms=max(0, state.black_time_ms - elapsed_ms),
    )


def _completed_turn_clock_state(
    previous: LichessGameState,
    current: LichessGameState,
) -> LichessGameState:
    white_time_ms = previous.white_time_ms
    black_time_ms = previous.black_time_ms
    if previous.turn == "white":
        white_time_ms += previous.white_increment_ms
    else:
        black_time_ms += previous.black_increment_ms
    return replace(
        current,
        white_time_ms=white_time_ms,
        black_time_ms=black_time_ms,
        white_increment_ms=previous.white_increment_ms,
        black_increment_ms=previous.black_increment_ms,
        white_draw_offer=previous.white_draw_offer,
        black_draw_offer=previous.black_draw_offer,
    )


def _resumed_clock_state(
    *,
    current: LichessGameState,
    export_turn: chess.Color,
    missing_plies: int,
    local_color: PlayerColor,
    local_time_ms: int,
    raw_white_time_ms: int,
    raw_black_time_ms: int,
    white_increment_ms: int,
    black_increment_ms: int,
    elapsed_since_export_ms: int,
) -> LichessGameState:
    white_moves = 0
    black_moves = 0
    turn = export_turn
    for _ in range(missing_plies):
        if turn == chess.WHITE:
            white_moves += 1
        else:
            black_moves += 1
        turn = not turn
    white_budget_ms = (
        raw_white_time_ms + white_moves * white_increment_ms
    )
    black_budget_ms = (
        raw_black_time_ms + black_moves * black_increment_ms
    )
    if local_color == "white":
        local_elapsed_ms = max(0, white_budget_ms - local_time_ms)
        black_time_ms = max(
            0,
            black_budget_ms
            - max(0, elapsed_since_export_ms - local_elapsed_ms),
        )
        white_time_ms = local_time_ms
    else:
        local_elapsed_ms = max(0, black_budget_ms - local_time_ms)
        white_time_ms = max(
            0,
            white_budget_ms
            - max(0, elapsed_since_export_ms - local_elapsed_ms),
        )
        black_time_ms = local_time_ms
    return replace(
        current,
        white_time_ms=white_time_ms,
        black_time_ms=black_time_ms,
        white_increment_ms=white_increment_ms,
        black_increment_ms=black_increment_ms,
    )


class LichessMoveSubmissionWorker(QObject):
    accepted = Signal(str, str)
    rejected = Signal(str, str, str)
    uncertain = Signal(str, str, str)
    finished = Signal()

    def __init__(
        self,
        client: LichessReadOnlyClient,
        game_id: str,
        uci: str,
        confirmed_moves: tuple[str, ...],
        *,
        confirmed_fen: str | None = None,
    ) -> None:
        super().__init__()
        self._client = client
        self._game_id = game_id
        self._uci = uci
        self._confirmed_moves = confirmed_moves
        self._confirmed_fen = confirmed_fen

    @Slot()
    def run(self) -> None:
        try:
            self._client.submit_move(self._game_id, self._uci)
        except LichessHTTPError as error:
            self.rejected.emit(
                self._game_id,
                self._uci,
                f"Lichess rejected the move with HTTP {error.status}.",
            )
        except (
            LichessConnectionError,
            LichessProtocolError,
            OSError,
        ) as error:
            self._resolve_uncertain_outcome(str(error))
        except LichessError as error:
            self.rejected.emit(self._game_id, self._uci, str(error))
        else:
            self._resolve_uncertain_outcome(
                "The move request succeeded, but authoritative confirmation "
                "is still required.",
                successful_response=True,
            )
        finally:
            self.finished.emit()

    def _resolve_uncertain_outcome(
        self,
        submission_error: str,
        *,
        successful_response: bool = False,
    ) -> None:
        expected_fen: str | None = None
        if self._confirmed_fen is not None:
            try:
                confirmed_board = chess.Board(self._confirmed_fen)
                confirmed_fen = confirmed_board.fen()
                confirmed_board.push_uci(self._uci)
            except ValueError:
                confirmed_fen = None
            else:
                expected_fen = confirmed_board.fen()
        else:
            try:
                confirmed_board = chess.Board()
                for confirmed_move in self._confirmed_moves:
                    confirmed_board.push_uci(confirmed_move)
                confirmed_fen = confirmed_board.fen()
                confirmed_board.push_uci(self._uci)
            except ValueError:
                confirmed_fen = None
            else:
                expected_fen = confirmed_board.fen()
        try:
            ongoing_games = self._client.get_ongoing_games()
        except (LichessError, OSError):
            ongoing_games = None
        if ongoing_games is not None:
            ongoing = next(
                (
                    game
                    for game in ongoing_games
                    if game.game_id == self._game_id
                ),
                None,
            )
            if ongoing is not None:
                try:
                    authoritative_fen = chess.Board(
                        ongoing.current_fen
                    ).fen()
                except ValueError:
                    self.uncertain.emit(
                        self._game_id,
                        self._uci,
                        "The active Lichess position is invalid while "
                        "resolving the submission outcome.",
                    )
                    return
                if (
                    expected_fen is not None
                    and authoritative_fen == expected_fen
                ):
                    self.accepted.emit(self._game_id, self._uci)
                    return
                if (
                    confirmed_fen is not None
                    and authoritative_fen == confirmed_fen
                ):
                    self.uncertain.emit(
                        self._game_id,
                        self._uci,
                        f"{submission_error} The active Lichess position "
                        "does not contain the move yet.",
                    )
                    return
                self.uncertain.emit(
                    self._game_id,
                    self._uci,
                    "The active Lichess position changed unexpectedly while "
                    "resolving the submission outcome.",
                )
                return
        try:
            snapshot = self._client.get_game_snapshot(self._game_id)
        except (LichessError, OSError) as refresh_error:
            self.uncertain.emit(
                self._game_id,
                self._uci,
                f"{submission_error} Authoritative refresh also failed: "
                f"{refresh_error}",
            )
            return
        server_moves = snapshot.game.state.moves
        authoritative_fen = chess.Board(snapshot.game.state.fen).fen()
        move_confirmed = (
            authoritative_fen == expected_fen
            if expected_fen is not None
            else server_moves == (*self._confirmed_moves, self._uci)
        )
        history_unchanged = (
            authoritative_fen == confirmed_fen
            if confirmed_fen is not None
            else server_moves == self._confirmed_moves
        )
        if move_confirmed:
            self.accepted.emit(self._game_id, self._uci)
            return
        if history_unchanged:
            if (
                successful_response
                and snapshot.game.state.result == "*"
            ):
                self.uncertain.emit(
                    self._game_id,
                    self._uci,
                    "The move request succeeded, but authoritative history "
                    "does not contain the move yet.",
                )
                return
            if snapshot.game.state.result != "*":
                self.rejected.emit(
                    self._game_id,
                    self._uci,
                    "Lichess ended the game before the move was recorded "
                    f"(status {snapshot.game.state.status}; result "
                    f"{snapshot.game.state.result}).",
                )
                return
            self.rejected.emit(
                self._game_id,
                self._uci,
                "Lichess history confirms that the move was not accepted.",
            )
            return
        self.uncertain.emit(
            self._game_id,
            self._uci,
            "Lichess history changed unexpectedly while resolving the "
            "submission outcome.",
        )


LichessGameAction = Literal[
    "offer-draw",
    "accept-draw",
    "decline-draw",
    "abort",
    "resign",
]


class LichessGameActionWorker(QObject):
    succeeded = Signal(str, str)
    rejected = Signal(str, str, str)
    uncertain = Signal(str, str, str)
    finished = Signal()

    def __init__(
        self,
        client: LichessReadOnlyClient,
        game_id: str,
        action: LichessGameAction,
    ) -> None:
        super().__init__()
        self._client = client
        self._game_id = game_id
        self._action = action

    @Slot()
    def run(self) -> None:
        try:
            if self._action in ("offer-draw", "accept-draw"):
                self._client.offer_draw(self._game_id)
            elif self._action == "decline-draw":
                self._client.decline_draw(self._game_id)
            elif self._action == "abort":
                self._client.abort_game(self._game_id)
            else:
                self._client.resign_game(self._game_id)
        except LichessHTTPError as error:
            self.rejected.emit(
                self._game_id,
                self._action,
                f"Lichess rejected the request with HTTP {error.status}.",
            )
        except (
            LichessConnectionError,
            LichessProtocolError,
            OSError,
        ) as error:
            self.uncertain.emit(
                self._game_id,
                self._action,
                f"{error} Refresh the game before repeating this request.",
            )
        except LichessError as error:
            self.rejected.emit(
                self._game_id,
                self._action,
                str(error),
            )
        else:
            self.succeeded.emit(self._game_id, self._action)
        finally:
            self.finished.emit()


class LichessAiGameCreationWorker(QObject):
    created = Signal(str)
    rejected = Signal(str)
    uncertain = Signal(str)
    finished = Signal()

    def __init__(
        self,
        client: LichessReadOnlyClient,
        *,
        level: int,
        color: AiChallengeColor,
        clock_limit_seconds: int | None,
        clock_increment_seconds: int | None,
    ) -> None:
        super().__init__()
        self._client = client
        self._level = level
        self._color = color
        self._clock_limit_seconds = clock_limit_seconds
        self._clock_increment_seconds = clock_increment_seconds

    @Slot()
    def run(self) -> None:
        try:
            game_id = self._client.create_ai_game(
                level=self._level,
                color=self._color,
                clock_limit_seconds=self._clock_limit_seconds,
                clock_increment_seconds=self._clock_increment_seconds,
            )
        except LichessHTTPError as error:
            self.rejected.emit(
                f"Lichess rejected AI game creation with HTTP {error.status}."
            )
        except (LichessConnectionError, OSError) as error:
            self.uncertain.emit(
                f"{error} Check ongoing games before creating another game."
            )
        except (LichessProtocolError, LichessError, ValueError) as error:
            self.rejected.emit(str(error))
        else:
            self.created.emit(game_id)
        finally:
            self.finished.emit()


class LichessConnectionWorker(QObject):
    status_changed = Signal(str)
    account_event_received = Signal(object)
    game_event_received = Signal(object)
    ongoing_games_received = Signal(object)
    failed = Signal(str)
    finished = Signal()

    def __init__(
        self,
        client: LichessReadOnlyClient,
        *,
        wait: Callable[[float], bool] | None = None,
    ) -> None:
        super().__init__()
        self._client = client
        self._stop_requested = Event()
        self._selection_lock = Lock()
        self._selected_game_id: str | None = None
        self._polling_game_id: str | None = None
        self._polling_initial_fen: str | None = None
        self._polling_fen: str | None = None
        self._polling_moves: tuple[str, ...] = ()
        self._polling_clock_state: LichessGameState | None = None
        self._polling_clock_monotonic: float | None = None
        self._polling_lock = Lock()
        self._wait = (
            self._stop_requested.wait if wait is None else wait
        )

    def request_stop(self) -> None:
        self._stop_requested.set()

    def select_game(self, game_id: str | None) -> None:
        selected = None if game_id is None else game_id.strip() or None
        with self._selection_lock:
            self._selected_game_id = selected

    def _selected_game(self) -> str | None:
        with self._selection_lock:
            return self._selected_game_id

    def note_local_move(
        self,
        game_id: str,
        uci: str,
        resulting_fen: str,
    ) -> bool:
        try:
            resulting_board = chess.Board(resulting_fen)
        except ValueError:
            return False
        normalized_result = resulting_board.fen()
        with self._polling_lock:
            if self._polling_game_id != game_id:
                return False
            if self._polling_fen == normalized_result:
                return True
            if self._polling_fen is None:
                return False
            try:
                previous_board = chess.Board(self._polling_fen)
                move = previous_board.parse_uci(uci)
                previous_board.push(move)
            except ValueError:
                return False
            if previous_board.fen() != normalized_result:
                return False
            self._polling_moves = (*self._polling_moves, move.uci())
            self._polling_fen = normalized_result
            return True

    @Slot()
    def run(self) -> None:
        delays = reconnect_delays(64)
        reconnect_attempt = 0
        polling_mode = False
        polled_game_id: str | None = None
        try:
            while not self._stop_requested.is_set():
                try:
                    if polling_mode:
                        polled_game_id = self._poll_ongoing_game(
                            polled_game_id
                        )
                        reconnect_attempt = 0
                        if self._wait(_POLLING_INTERVAL_SECONDS):
                            break
                        continue

                    def reset_reconnect_attempt() -> None:
                        nonlocal reconnect_attempt
                        reconnect_attempt = 0

                    game_id = self._wait_for_active_game(
                        reset_reconnect_attempt
                    )
                    if self._stop_requested.is_set():
                        break
                    if game_id is None:
                        raise LichessConnectionError(
                            "The Lichess account stream closed."
                        )
                    reconnect_attempt = 0
                    game_finished = self._follow_game(
                        game_id,
                        reset_reconnect_attempt,
                    )
                    if self._stop_requested.is_set():
                        break
                    if game_finished:
                        reconnect_attempt = 0
                        continue
                    raise LichessConnectionError(
                        "The active Lichess game stream closed."
                    )
                except LichessHTTPError as error:
                    if error.status in (401, 403):
                        self.failed.emit(
                            "Lichess authentication was rejected. Check "
                            "LICHESS_TOKEN and its board:play scope."
                        )
                        break
                    delay = (
                        error.retry_after_seconds
                        if error.status == 429
                        and error.retry_after_seconds is not None
                        else delays[min(reconnect_attempt, len(delays) - 1)]
                    )
                    reconnect_attempt += 1
                    self.status_changed.emit(
                        f"Lichess HTTP {error.status}; reconnecting in "
                        f"{delay} seconds."
                    )
                    if self._wait(delay):
                        break
                except LichessProtocolError as error:
                    self.failed.emit(str(error))
                    break
                except LichessConnectionError as error:
                    if not polling_mode:
                        polling_mode = True
                        reconnect_attempt = 0
                        self.status_changed.emit(
                            "Live Lichess stream unavailable; using "
                            "current-position polling instead."
                        )
                        continue
                    delay = delays[min(reconnect_attempt, len(delays) - 1)]
                    reconnect_attempt += 1
                    self.status_changed.emit(
                        f"Current-position polling failed: {error} "
                        f"Retrying in {delay} seconds."
                    )
                    if self._wait(delay):
                        break
                except LichessError as error:
                    delay = delays[min(reconnect_attempt, len(delays) - 1)]
                    reconnect_attempt += 1
                    self.status_changed.emit(
                        f"{error} Reconnecting in "
                        f"{delay} seconds."
                    )
                    if self._wait(delay):
                        break
                except OSError:
                    delay = delays[min(reconnect_attempt, len(delays) - 1)]
                    reconnect_attempt += 1
                    self.status_changed.emit(
                        "A Lichess socket operation failed; reconnecting in "
                        f"{delay} seconds."
                    )
                    if self._wait(delay):
                        break
        finally:
            self.finished.emit()

    def _poll_ongoing_game(
        self,
        previous_game_id: str | None,
    ) -> str | None:
        games = self._client.get_ongoing_games()
        if not games:
            if previous_game_id is not None:
                snapshot = self._client.get_game_snapshot(previous_game_id)
                if snapshot.game.state.result == "*":
                    self.status_changed.emit(
                        "Lichess game ended; waiting for its final "
                        "authoritative snapshot."
                    )
                    return previous_game_id
                final_snapshot = self._final_polling_snapshot(
                    previous_game_id,
                    snapshot,
                )
                self.game_event_received.emit(final_snapshot)
            self.ongoing_games_received.emit(games)
            self.select_game(None)
            self._reset_polling_cursor()
            self.status_changed.emit(
                "Connected to Lichess by polling; waiting for a game."
            )
            return None
        self.ongoing_games_received.emit(games)
        selected_game_id = self._selected_game()
        if len(games) == 1:
            game = games[0]
            self.select_game(game.game_id)
        else:
            game = next(
                (
                    candidate
                    for candidate in games
                    if candidate.game_id == selected_game_id
                ),
                None,
            )
            if game is None:
                self.select_game(None)
                self._reset_polling_cursor()
                self.status_changed.emit(
                    "Multiple ongoing Lichess games found; select a game."
                )
                return None
        if game.game_id != previous_game_id:
            self.account_event_received.emit(
                AccountGameEvent(
                    "gameStart",
                    game.game_id,
                    game.full_id,
                    game.color,
                    game.current_fen,
                    game.opponent_name,
                    game.is_my_turn,
                )
            )
        snapshot = self._active_polling_snapshot(
            game,
            self._client.get_game_snapshot(game.game_id),
        )
        self.game_event_received.emit(snapshot)
        return game.game_id

    def _active_polling_snapshot(
        self,
        game: OngoingGame,
        authoritative: LichessPollingSnapshot,
    ) -> LichessPollingSnapshot:
        observed_at = monotonic()
        observed_wall_ms = int(time() * 1000)
        snapshot = self._polling_snapshot(game)
        if authoritative.game.game_id != game.game_id:
            raise LichessProtocolError(
                "The polling clock snapshot belongs to another game."
            )
        try:
            authoritative_board = chess.Board(
                authoritative.game.state.fen
            )
            authoritative_fen = authoritative_board.fen()
            current_board = chess.Board(snapshot.game.state.fen)
        except ValueError as error:
            raise LichessProtocolError(
                "The polling clock snapshot has an invalid FEN."
            ) from error
        state = authoritative.game.state
        clocks_available = authoritative.clocks_available
        if authoritative_fen != snapshot.game.state.fen:
            one_move_behind = False
            if game.last_move is not None:
                try:
                    export_board = authoritative_board.copy()
                    missing_move = export_board.parse_uci(game.last_move)
                    export_board.push(missing_move)
                except ValueError:
                    pass
                else:
                    one_move_behind = (
                        export_board.fen() == snapshot.game.state.fen
                    )
            missing_plies = current_board.ply() - authoritative_board.ply()
            previous = self._polling_clock_state
            previous_at = self._polling_clock_monotonic
            elapsed_previous = (
                _elapsed_clock_state(
                    previous,
                    int(max(0.0, observed_at - previous_at) * 1000),
                )
                if previous is not None and previous_at is not None
                else None
            )
            if (
                elapsed_previous is not None
                and elapsed_previous.fen == snapshot.game.state.fen
            ):
                state = elapsed_previous
                clocks_available = True
            else:
                previous_is_one_move_behind = False
                if elapsed_previous is not None and game.last_move is not None:
                    try:
                        previous_board = chess.Board(elapsed_previous.fen)
                        previous_move = previous_board.parse_uci(
                            game.last_move
                        )
                        previous_board.push(previous_move)
                    except ValueError:
                        pass
                    else:
                        previous_is_one_move_behind = (
                            previous_board.fen()
                            == snapshot.game.state.fen
                        )
                if (
                    previous_is_one_move_behind
                    and elapsed_previous is not None
                ):
                    state = _completed_turn_clock_state(
                        elapsed_previous,
                        snapshot.game.state,
                    )
                    clocks_available = True
                elif one_move_behind and authoritative.clocks_available:
                    state = _completed_turn_clock_state(
                        state,
                        snapshot.game.state,
                    )
                    clocks_available = True
                elif (
                    missing_plies > 1
                    and authoritative.clocks_available
                    and authoritative.clock_sample_at_ms is not None
                    and authoritative.raw_white_time_ms is not None
                    and authoritative.raw_black_time_ms is not None
                    and game.seconds_left_ms is not None
                ):
                    state = _resumed_clock_state(
                        current=snapshot.game.state,
                        export_turn=authoritative_board.turn,
                        missing_plies=missing_plies,
                        local_color=game.color,
                        local_time_ms=game.seconds_left_ms,
                        raw_white_time_ms=(
                            authoritative.raw_white_time_ms
                        ),
                        raw_black_time_ms=(
                            authoritative.raw_black_time_ms
                        ),
                        white_increment_ms=state.white_increment_ms,
                        black_increment_ms=state.black_increment_ms,
                        elapsed_since_export_ms=max(
                            0,
                            observed_wall_ms
                            - authoritative.clock_sample_at_ms,
                        ),
                    )
                    clocks_available = True
                else:
                    clocks_available = False
                    state = replace(
                        snapshot.game.state,
                        white_time_ms=0,
                        black_time_ms=0,
                        white_increment_ms=state.white_increment_ms,
                        black_increment_ms=state.black_increment_ms,
                    )
        if game.seconds_left_ms is not None and clocks_available:
            if game.color == "white":
                state = replace(
                    state,
                    white_time_ms=game.seconds_left_ms,
                )
            else:
                state = replace(
                    state,
                    black_time_ms=game.seconds_left_ms,
                )
        if (
            game.seconds_left_ms == 0
            and game.is_my_turn
            and state.result == "*"
        ):
            winner: PlayerColor = (
                "black" if game.color == "white" else "white"
            )
            state = replace(
                state,
                status="outoftime",
                winner=winner,
                result="0-1" if winner == "black" else "1-0",
            )
        final_state = LichessGameState(
            snapshot.game.state.moves,
            snapshot.game.state.fen,
            snapshot.game.state.turn,
            state.white_time_ms,
            state.black_time_ms,
            state.white_increment_ms,
            state.black_increment_ms,
            state.status,
            state.winner,
            state.result,
            state.white_draw_offer,
            state.black_draw_offer,
        )
        if clocks_available and final_state.result == "*":
            self._polling_clock_state = final_state
            self._polling_clock_monotonic = observed_at
        else:
            self._polling_clock_state = None
            self._polling_clock_monotonic = None
        return LichessPollingSnapshot(
            LichessGameFull(
                snapshot.game.game_id,
                snapshot.game.initial_fen,
                authoritative.game.white,
                authoritative.game.black,
                final_state,
            ),
            clocks_available,
        )

    def _final_polling_snapshot(
        self,
        game_id: str,
        snapshot: LichessPollingSnapshot,
    ) -> LichessPollingSnapshot:
        with self._polling_lock:
            if (
                self._polling_game_id != game_id
                or self._polling_initial_fen is None
                or self._polling_fen is None
            ):
                raise LichessProtocolError(
                    "The finished Lichess game has no polling baseline."
                )
            final_fen, official_suffix = self._official_polling_suffix(
                snapshot,
                self._polling_initial_fen,
            )
            self._polling_moves = official_suffix
            self._polling_fen = final_fen
            state = snapshot.game.state
            return LichessPollingSnapshot(
                LichessGameFull(
                    snapshot.game.game_id,
                    self._polling_initial_fen,
                    snapshot.game.white,
                    snapshot.game.black,
                    LichessGameState(
                        self._polling_moves,
                        final_fen,
                        state.turn,
                        state.white_time_ms,
                        state.black_time_ms,
                        state.white_increment_ms,
                        state.black_increment_ms,
                        state.status,
                        state.winner,
                        state.result,
                        state.white_draw_offer,
                        state.black_draw_offer,
                    ),
                ),
                snapshot.clocks_available,
            )

    @staticmethod
    def _official_polling_suffix(
        snapshot: LichessPollingSnapshot,
        polling_initial_fen: str,
    ) -> tuple[str, tuple[str, ...]]:
        try:
            board = chess.Board(snapshot.game.initial_fen)
            baseline = chess.Board(polling_initial_fen).fen()
            final_fen = chess.Board(snapshot.game.state.fen).fen()
        except ValueError as error:
            raise LichessProtocolError(
                "The finished Lichess game has an invalid FEN."
            ) from error
        baseline_index: int | None = 0 if board.fen() == baseline else None
        official_moves: list[str] = []
        for index, uci in enumerate(snapshot.game.state.moves, start=1):
            try:
                move = board.parse_uci(uci)
                board.push(move)
            except ValueError as error:
                raise LichessProtocolError(
                    "The finished Lichess game has invalid move history."
                ) from error
            official_moves.append(move.uci())
            if board.fen() == baseline:
                baseline_index = index
        if board.fen() != final_fen:
            raise LichessProtocolError(
                "The finished Lichess game history does not produce its "
                "final position."
            )
        if baseline_index is None:
            raise LichessProtocolError(
                "The finished Lichess game does not contain the polling "
                "baseline."
            )
        return final_fen, tuple(official_moves[baseline_index:])

    def _polling_snapshot(
        self,
        game: OngoingGame,
    ) -> LichessPollingSnapshot:
        with self._polling_lock:
            return self._polling_snapshot_locked(game)

    def _polling_snapshot_locked(
        self,
        game: OngoingGame,
    ) -> LichessPollingSnapshot:
        try:
            current_board = chess.Board(game.current_fen)
        except ValueError as error:
            raise LichessProtocolError(
                "The ongoing Lichess game has an invalid current FEN."
            ) from error
        current_fen = current_board.fen()
        if self._polling_game_id != game.game_id:
            self._set_polling_baseline(game.game_id, current_fen)
        elif current_fen != self._polling_fen:
            advanced = False
            retained_confirmed_position = False
            if self._polling_fen is not None and self._polling_moves:
                try:
                    stale_board = current_board.copy()
                    confirmed_move = stale_board.parse_uci(
                        self._polling_moves[-1]
                    )
                    stale_board.push(confirmed_move)
                except ValueError:
                    pass
                else:
                    retained_confirmed_position = (
                        stale_board.fen() == self._polling_fen
                    )
                    if retained_confirmed_position:
                        current_board = stale_board
                        current_fen = self._polling_fen
            if (
                not retained_confirmed_position
                and self._polling_fen is not None
                and game.last_move is not None
            ):
                try:
                    previous_board = chess.Board(self._polling_fen)
                    move = previous_board.parse_uci(game.last_move)
                    previous_board.push(move)
                except ValueError:
                    pass
                else:
                    if previous_board.fen() == current_fen:
                        self._polling_moves = (
                            *self._polling_moves,
                            move.uci(),
                        )
                        self._polling_fen = current_fen
                        advanced = True
            if not advanced and not retained_confirmed_position:
                self._set_polling_baseline(game.game_id, current_fen)
        initial_fen = self._polling_initial_fen
        if initial_fen is None:
            raise RuntimeError("Polling baseline was not initialized.")
        local_player = LichessPlayer(game.color, "Me", None)
        opponent_color = "black" if game.color == "white" else "white"
        opponent = LichessPlayer(
            opponent_color,
            game.opponent_name,
            None,
        )
        return LichessPollingSnapshot(
            LichessGameFull(
                game.game_id,
                initial_fen,
                local_player if game.color == "white" else opponent,
                opponent if game.color == "white" else local_player,
                LichessGameState(
                    self._polling_moves,
                    current_fen,
                    "white" if current_board.turn == chess.WHITE else "black",
                    0,
                    0,
                    0,
                    0,
                    "started",
                    None,
                    "*",
                ),
            ),
            False,
        )

    def _set_polling_baseline(
        self,
        game_id: str,
        fen: str,
    ) -> None:
        self._polling_game_id = game_id
        self._polling_initial_fen = fen
        self._polling_fen = fen
        self._polling_moves = ()
        self._polling_clock_state = None
        self._polling_clock_monotonic = None

    def _reset_polling_cursor(self) -> None:
        with self._polling_lock:
            self._polling_game_id = None
            self._polling_initial_fen = None
            self._polling_fen = None
            self._polling_moves = ()
            self._polling_clock_state = None
            self._polling_clock_monotonic = None

    def _wait_for_active_game(
        self,
        on_open: Callable[[], None],
    ) -> str | None:
        def stream_opened() -> None:
            on_open()
            self.status_changed.emit(
                "Connected to Lichess; waiting for an active game."
            )

        events = self._client.stream_account_events(
            on_open=stream_opened,
            should_stop=self._stop_requested.is_set,
        )
        try:
            for event in events:
                self.account_event_received.emit(event)
                if event.event_type == "gameStart":
                    return event.game_id
            return None
        finally:
            _close_iterator(events)

    def _follow_game(
        self,
        game_id: str,
        on_open: Callable[[], None],
    ) -> bool:
        game_finished = False

        def stream_opened() -> None:
            on_open()
            self.status_changed.emit(
                f"Following Lichess game {game_id}."
            )

        events = self._client.stream_game(
            game_id,
            on_open=stream_opened,
            should_stop=self._stop_requested.is_set,
        )
        try:
            for event in events:
                self.game_event_received.emit(event)
                if (
                    isinstance(event, LichessGameState)
                    and event.status not in ("created", "started")
                ):
                    game_finished = True
            return game_finished
        finally:
            _close_iterator(events)


def _close_iterator(
    iterator: Iterator[AccountGameEvent] | Iterator[GameStreamEvent],
) -> None:
    close = getattr(iterator, "close", None)
    if callable(close):
        close()
