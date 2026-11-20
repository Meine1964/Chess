from collections.abc import Callable, Iterator
from threading import Event

from PySide6.QtCore import QObject, Signal, Slot

from .lichess_client import (
    AccountGameEvent,
    GameStreamEvent,
    LichessConnectionError,
    LichessError,
    LichessGameState,
    LichessHTTPError,
    LichessProtocolError,
    LichessReadOnlyClient,
    reconnect_delays,
)


class LichessConnectionWorker(QObject):
    status_changed = Signal(str)
    account_event_received = Signal(object)
    game_event_received = Signal(object)
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
        self._wait = (
            self._stop_requested.wait if wait is None else wait
        )

    def request_stop(self) -> None:
        self._stop_requested.set()

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
                            f"{error} Using read-only polling instead."
                        )
                        continue
                    delay = delays[min(reconnect_attempt, len(delays) - 1)]
                    reconnect_attempt += 1
                    self.status_changed.emit(
                        f"{error} Retrying polling in {delay} seconds."
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
        if previous_game_id is not None:
            snapshot = self._client.get_game_snapshot(previous_game_id)
            self.game_event_received.emit(snapshot)
            if snapshot.game.state.status not in ("created", "started"):
                return None
            return previous_game_id
        games = self._client.get_ongoing_games()
        if not games:
            self.status_changed.emit(
                "Connected to Lichess by polling; waiting for a game."
            )
            return None
        game = games[0]
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
        snapshot = self._client.get_game_snapshot(game.game_id)
        self.game_event_received.emit(snapshot)
        return game.game_id

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
