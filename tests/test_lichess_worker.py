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
from src.chess_camera.lichess_worker import LichessMoveSubmissionWorker


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

class MultiplePollingGamesClient:
    def __init__(self) -> None:
        self.games = (
            OngoingGame(
                "oldGame1",
                "oldGame1abcd",
                "black",
                chess.STARTING_FEN,
                "Older opponent",
                True,
            ),
            OngoingGame(
                "newGame2",
                "newGame2abcd",
                "white",
                chess.STARTING_FEN,
                "Selected opponent",
                False,
            ),
        )

    def get_ongoing_games(self) -> tuple[OngoingGame, ...]:
        return self.games


class SubmissionClient:
    def __init__(
        self,
        *,
        submit_error: Exception | None = None,
        server_moves: tuple[str, ...] = (),
        refresh_error: Exception | None = None,
    ) -> None:
        self._submit_error = submit_error
        self._server_moves = server_moves
        self._refresh_error = refresh_error
        self.submissions: list[tuple[str, str]] = []

    def submit_move(self, game_id: str, uci: str) -> None:
        self.submissions.append((game_id, uci))
        if self._submit_error is not None:
            raise self._submit_error

    def get_game_snapshot(self, game_id: str) -> LichessPollingSnapshot:
        if self._refresh_error is not None:
            raise self._refresh_error
        board = chess.Board()
        for uci in self._server_moves:
            board.push_uci(uci)
        state = LichessGameState(
            self._server_moves,
            board.fen(),
            "white" if board.turn else "black",
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


def test_move_submission_worker_emits_acceptance() -> None:
    client = SubmissionClient()
    worker = LichessMoveSubmissionWorker(
        cast(LichessReadOnlyClient, client),
        "game1234",
        "e2e4",
        (),
    )
    accepted: list[tuple[str, str]] = []
    worker.accepted.connect(
        lambda game_id, uci: accepted.append((game_id, uci))
    )

    worker.run()

    assert client.submissions == [("game1234", "e2e4")]
    assert accepted == [("game1234", "e2e4")]


def test_move_submission_worker_resolves_lost_response_from_history() -> None:
    client = SubmissionClient(
        submit_error=LichessConnectionError("Response was lost."),
        server_moves=("e2e4",),
    )
    worker = LichessMoveSubmissionWorker(
        cast(LichessReadOnlyClient, client),
        "game1234",
        "e2e4",
        (),
    )
    accepted: list[tuple[str, str]] = []
    worker.accepted.connect(
        lambda game_id, uci: accepted.append((game_id, uci))
    )

    worker.run()

    assert accepted == [("game1234", "e2e4")]


def test_move_submission_worker_resolves_rejection_from_history() -> None:
    client = SubmissionClient(
        submit_error=LichessConnectionError("Response was lost."),
    )
    worker = LichessMoveSubmissionWorker(
        cast(LichessReadOnlyClient, client),
        "game1234",
        "e2e4",
        (),
    )
    rejected: list[tuple[str, str, str]] = []
    worker.rejected.connect(
        lambda game_id, uci, message: rejected.append(
            (game_id, uci, message)
        )
    )

    worker.run()

    assert rejected == [
        (
            "game1234",
            "e2e4",
            "Lichess history confirms that the move was not accepted.",
        )
    ]


def test_move_submission_worker_reports_unresolved_outcome() -> None:
    client = SubmissionClient(
        submit_error=LichessConnectionError("Response was lost."),
        refresh_error=LichessConnectionError("Refresh failed."),
    )
    worker = LichessMoveSubmissionWorker(
        cast(LichessReadOnlyClient, client),
        "game1234",
        "e2e4",
        (),
    )
    uncertain: list[tuple[str, str, str]] = []
    worker.uncertain.connect(
        lambda game_id, uci, message: uncertain.append(
            (game_id, uci, message)
        )
    )

    worker.run()

    assert len(uncertain) == 1
    assert uncertain[0][:2] == ("game1234", "e2e4")


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
            "Live Lichess stream unavailable; using current-position "
            "polling instead."
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
        "Live Lichess stream unavailable; using current-position polling "
        "instead."
    ]
    assert account_events[0].game_id == "oozTO3xk"
    assert isinstance(game_events[0], LichessPollingSnapshot)


def test_polling_waits_for_selection_when_multiple_games_are_active() -> None:
    client = MultiplePollingGamesClient()
    worker = LichessConnectionWorker(cast(LichessReadOnlyClient, client))
    statuses: list[str] = []
    ongoing: list[tuple[OngoingGame, ...]] = []
    account_events: list[AccountGameEvent] = []
    game_events: list[object] = []
    worker.status_changed.connect(statuses.append)
    worker.ongoing_games_received.connect(ongoing.append)
    worker.account_event_received.connect(account_events.append)
    worker.game_event_received.connect(game_events.append)

    assert worker._poll_ongoing_game(None) is None

    assert ongoing == [client.games]
    assert game_events == []
    assert account_events == []
    assert statuses == [
        "Multiple ongoing Lichess games found; select a game."
    ]

    worker.select_game("newGame2")

    assert worker._poll_ongoing_game(None) == "newGame2"
    assert isinstance(game_events[-1], LichessPollingSnapshot)
    assert game_events[-1].game.game_id == "newGame2"
    assert account_events[-1].game_id == "newGame2"


def test_polling_cursor_appends_authoritative_last_move() -> None:
    worker = LichessConnectionWorker(
        cast(LichessReadOnlyClient, MultiplePollingGamesClient())
    )
    board = chess.Board()
    for uci in ("e2e4", "e7e6", "g1f3", "d7d5", "b1c3"):
        board.push_uci(uci)
    before = OngoingGame(
        "jgWZLWs6",
        "jgWZLWs6abcd",
        "white",
        board.fen(),
        "Stockfish level 1",
        False,
        "b1c3",
    )

    initial = worker._polling_snapshot(before)
    board.push_uci("c7c6")
    after = OngoingGame(
        "jgWZLWs6",
        "jgWZLWs6abcd",
        "white",
        board.fen(),
        "Stockfish level 1",
        True,
        "c7c6",
    )
    current = worker._polling_snapshot(after)

    assert initial.game.state.moves == ()
    assert current.game.state.moves == ("c7c6",)
    assert current.game.state.fen == board.fen()
    assert current.game.state.turn == "white"
