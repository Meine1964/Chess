from collections.abc import Callable, Collection, Iterator
from threading import Event, Lock
from typing import Literal

import chess
from PySide6.QtCore import QObject, Signal, Slot

from .lichess_client import (
    AccountGameEvent,
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
    reconnect_delays,
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
    ) -> None:
        super().__init__()
        self._client = client
        self._game_id = game_id
        self._uci = uci
        self._confirmed_moves = confirmed_moves

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
            self.accepted.emit(self._game_id, self._uci)
        finally:
            self.finished.emit()

    def _resolve_uncertain_outcome(self, submission_error: str) -> None:
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
        if server_moves == (*self._confirmed_moves, self._uci):
            self.accepted.emit(self._game_id, self._uci)
            return
        if server_moves == self._confirmed_moves:
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


LichessGameAction = Literal["offer-draw", "resign"]


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
            if self._action == "offer-draw":
                self._client.offer_draw(self._game_id)
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
        self._new_game_baseline_ids: frozenset[str] | None = None
        self._polling_game_id: str | None = None
        self._polling_initial_fen: str | None = None
        self._polling_fen: str | None = None
        self._polling_moves: tuple[str, ...] = ()
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
            if selected is not None:
                self._new_game_baseline_ids = None

    def expect_new_game(
        self,
        existing_game_ids: Collection[str],
    ) -> None:
        with self._selection_lock:
            self._selected_game_id = None
            self._new_game_baseline_ids = frozenset(existing_game_ids)

    def _game_selection(
        self,
    ) -> tuple[str | None, frozenset[str] | None]:
        with self._selection_lock:
            return self._selected_game_id, self._new_game_baseline_ids

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
                        if self._wait(5):
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
        self.ongoing_games_received.emit(games)
        if not games:
            self.select_game(None)
            self._reset_polling_cursor()
            self.status_changed.emit(
                "Connected to Lichess by polling; waiting for a game."
            )
            return None
        selected_game_id, new_game_baseline_ids = self._game_selection()
        if new_game_baseline_ids is not None:
            new_games = tuple(
                game
                for game in games
                if game.game_id not in new_game_baseline_ids
            )
            if len(new_games) == 1:
                game = new_games[0]
                self.select_game(game.game_id)
            else:
                self._reset_polling_cursor()
                self.status_changed.emit(
                    "Connected to Lichess by polling; waiting for the newly "
                    "started game."
                )
                return None
        elif len(games) == 1:
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
        snapshot = self._polling_snapshot(game)
        self.game_event_received.emit(snapshot)
        return game.game_id

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
            if self._polling_fen is not None and game.last_move is not None:
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
            if not advanced:
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

    def _reset_polling_cursor(self) -> None:
        with self._polling_lock:
            self._polling_game_id = None
            self._polling_initial_fen = None
            self._polling_fen = None
            self._polling_moves = ()

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
