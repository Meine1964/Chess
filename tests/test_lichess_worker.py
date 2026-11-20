from collections.abc import Callable, Iterator
from typing import cast

import chess

from src.chess_camera.lichess_client import (
    AccountGameEvent,
    GameStreamEvent,
    LichessConnectionError,
    LichessGameFull,
    LichessGameState,
    LichessHTTPError,
    LichessPlayer,
    LichessPollingSnapshot,
    LichessReadOnlyClient,
    OngoingGame,
)
from src.chess_camera.lichess_worker import LichessConnectionWorker


class FakeClient:
    def stream_account_events(
        self,
        *,
        on_open: Callable[[], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> Iterator[AccountGameEvent]:
        if on_open is not None:
            on_open()
        if should_stop is None or not should_stop():
            yield AccountGameEvent(
                "gameStart",
                "qP8XaFtN",
                "qP8XaFtNabcd",
                "white",
                chess.STARTING_FEN,
                "Salma",
                True,
            )

    def stream_game(
        self,
        game_id: str,
        *,
        on_open: Callable[[], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> Iterator[GameStreamEvent]:
        assert game_id == "qP8XaFtN"
        if on_open is not None:
            on_open()
        if should_stop is None or not should_stop():
            yield LichessGameState(
                ("e2e4",),
                chess.Board().fen(),
                "black",
                299000,
                300000,
                0,
                0,
                "started",
                None,
                "*",
            )


class RateLimitedClient:
    def stream_account_events(
        self,
        *,
        on_open: Callable[[], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> Iterator[AccountGameEvent]:
        del on_open, should_stop
        if False:
            yield
        raise LichessHTTPError(
            429,
            retry_after_seconds=75,
        )


class RawTimeoutClient:
    def stream_account_events(
        self,
        *,
        on_open: Callable[[], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> Iterator[AccountGameEvent]:
        del on_open, should_stop
        if False:
            yield
        raise TimeoutError("The read operation timed out")


class OpenThenTimeoutClient:
    def stream_account_events(
        self,
        *,
        on_open: Callable[[], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> Iterator[AccountGameEvent]:
        del should_stop
        if False:
            yield
        if on_open is not None:
            on_open()
        raise TimeoutError("The read operation timed out")


class ClosedAccountStreamClient:
    def stream_account_events(
        self,
        *,
        on_open: Callable[[], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> Iterator[AccountGameEvent]:
        del should_stop
        if on_open is not None:
            on_open()
        if False:
            yield

    def get_ongoing_games(self) -> tuple[OngoingGame, ...]:
        return ()


class PollingFallbackClient:
    def stream_account_events(
        self,
        *,
        on_open: Callable[[], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> Iterator[AccountGameEvent]:
        del on_open, should_stop
        if False:
            yield
        raise LichessConnectionError("Could not connect to Lichess.")

    def get_ongoing_games(self) -> tuple[OngoingGame, ...]:
        return (
            OngoingGame(
                "oozTO3xk",
                "oozTO3xkabcd",
                "white",
                chess.STARTING_FEN,
                "Stockfish level 1",
                True,
            ),
        )

    def get_game_snapshot(
        self,
        game_id: str,
    ) -> LichessPollingSnapshot:
        assert game_id == "oozTO3xk"
        state = LichessGameState(
            (),
            chess.STARTING_FEN,
            "white",
            0,
            0,
            0,
            0,
            "started",
            None,
            "*",
        )
        return LichessPollingSnapshot(
            LichessGameFull(
                game_id,
                chess.STARTING_FEN,
                LichessPlayer("white", "Meine", 1500),
                LichessPlayer("black", "Stockfish level 1", None),
                state,
            ),
            False,
        )


def test_worker_follows_active_game_without_blocking_retry_loop() -> None:
    worker = LichessConnectionWorker(
        cast(LichessReadOnlyClient, FakeClient()),
    )
    statuses: list[str] = []
    account_events: list[AccountGameEvent] = []
    game_events: list[GameStreamEvent] = []
    finished: list[bool] = []
    worker.status_changed.connect(statuses.append)
    worker.account_event_received.connect(account_events.append)

    def receive_game_event(event: GameStreamEvent) -> None:
        game_events.append(event)
        worker.request_stop()

    worker.game_event_received.connect(receive_game_event)
    worker.finished.connect(lambda: finished.append(True))

    worker.run()

    assert statuses == [
        "Connected to Lichess; waiting for an active game.",
        "Following Lichess game qP8XaFtN.",
    ]
    assert account_events[0].opponent_name == "Salma"
    assert isinstance(game_events[0], LichessGameState)
    assert finished == [True]


def test_worker_respects_rate_limit_retry_delay() -> None:
    waits: list[float] = []
    worker = LichessConnectionWorker(
        cast(LichessReadOnlyClient, RateLimitedClient()),
        wait=lambda delay: waits.append(delay) is None,
    )
    statuses: list[str] = []
    worker.status_changed.connect(statuses.append)

    worker.run()

    assert waits == [75]
    assert statuses == [
        "Lichess HTTP 429; reconnecting in 75 seconds."
    ]


def test_worker_reconnects_after_untranslated_socket_timeout() -> None:
    waits: list[float] = []
    worker = LichessConnectionWorker(
        cast(LichessReadOnlyClient, RawTimeoutClient()),
        wait=lambda delay: waits.append(delay) is None,
    )
    statuses: list[str] = []
    worker.status_changed.connect(statuses.append)

    worker.run()

    assert waits == [1]
    assert statuses == [
        "A Lichess socket operation failed; reconnecting in 1 seconds."
    ]


def test_successful_stream_open_resets_reconnect_backoff() -> None:
    waits: list[float] = []

    def wait(delay: float) -> bool:
        waits.append(delay)
        return len(waits) == 2

    worker = LichessConnectionWorker(
        cast(LichessReadOnlyClient, OpenThenTimeoutClient()),
        wait=wait,
    )
    statuses: list[str] = []
    worker.status_changed.connect(statuses.append)

    worker.run()

    assert waits == [1, 1]
    assert statuses == [
        "Connected to Lichess; waiting for an active game.",
        "A Lichess socket operation failed; reconnecting in 1 seconds.",
        "Connected to Lichess; waiting for an active game.",
        "A Lichess socket operation failed; reconnecting in 1 seconds.",
    ]


def test_worker_reports_when_account_stream_closes() -> None:
    worker = LichessConnectionWorker(
        cast(LichessReadOnlyClient, ClosedAccountStreamClient()),
        wait=lambda _delay: True,
    )
    statuses: list[str] = []
    worker.status_changed.connect(statuses.append)

    worker.run()

    assert statuses == [
        "Connected to Lichess; waiting for an active game.",
        (
            "The Lichess account stream closed. "
            "Using read-only polling instead."
        ),
        "Connected to Lichess by polling; waiting for a game.",
    ]


def test_worker_falls_back_to_read_only_polling() -> None:
    worker = LichessConnectionWorker(
        cast(LichessReadOnlyClient, PollingFallbackClient()),
        wait=lambda delay: delay == 5,
    )
    statuses: list[str] = []
    account_events: list[AccountGameEvent] = []
    game_events: list[object] = []
    worker.status_changed.connect(statuses.append)
    worker.account_event_received.connect(account_events.append)
    worker.game_event_received.connect(game_events.append)

    worker.run()

    assert statuses == [
        "Could not connect to Lichess. Using read-only polling instead."
    ]
    assert account_events[0].game_id == "oozTO3xk"
    assert isinstance(game_events[0], LichessPollingSnapshot)
