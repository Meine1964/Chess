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
    PlayerColor,
)
from src.chess_camera.lichess_worker import (
    LichessAiGameCreationWorker,
    LichessConnectionWorker,
    LichessGameAction,
    LichessGameActionWorker,
    LichessMoveSubmissionWorker,
)


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
    def __init__(self) -> None:
        self.snapshot_calls: list[str] = []

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
        self.snapshot_calls.append(game_id)
        return LichessPollingSnapshot(
            LichessGameFull(
                game_id,
                chess.STARTING_FEN,
                LichessPlayer("white", "Meine", 1500),
                LichessPlayer("black", "Stockfish level 1", None),
                LichessGameState(
                    (),
                    chess.STARTING_FEN,
                    "white",
                    173000,
                    604000,
                    5000,
                    5000,
                    "started",
                    None,
                    "*",
                ),
            ),
            True,
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

    def get_game_snapshot(
        self,
        game_id: str,
    ) -> LichessPollingSnapshot:
        game = next(item for item in self.games if item.game_id == game_id)
        return LichessPollingSnapshot(
            LichessGameFull(
                game_id,
                chess.STARTING_FEN,
                LichessPlayer("white", "White", None),
                LichessPlayer("black", "Black", None),
                LichessGameState(
                    (),
                    game.current_fen,
                    "white",
                    300000,
                    300000,
                    0,
                    0,
                    "started",
                    None,
                    "*",
                ),
            ),
            True,
        )


class SubmissionClient:
    def __init__(
        self,
        *,
        submit_error: Exception | None = None,
        server_moves: tuple[str, ...] = (),
        ongoing_moves: tuple[str, ...] | None = None,
        refresh_error: Exception | None = None,
        status: str = "started",
        winner: PlayerColor | None = None,
        result: str = "*",
    ) -> None:
        self._submit_error = submit_error
        self._server_moves = server_moves
        self._ongoing_moves = (
            server_moves if ongoing_moves is None else ongoing_moves
        )
        self._refresh_error = refresh_error
        self._status = status
        self._winner = winner
        self._result = result
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
            self._status,
            self._winner,
            self._result,
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

    def get_ongoing_games(self) -> tuple[OngoingGame, ...]:
        if self._refresh_error is not None:
            raise self._refresh_error
        if self._status != "started":
            return ()
        board = chess.Board()
        for uci in self._ongoing_moves:
            board.push_uci(uci)
        return (
            OngoingGame(
                "game1234",
                "game1234abcd",
                "white",
                board.fen(),
                "Stockfish level 1",
                board.turn == chess.WHITE,
                self._ongoing_moves[-1] if self._ongoing_moves else None,
            ),
        )


class GameActionClient:
    def __init__(self) -> None:
        self.actions: list[tuple[str, str]] = []

    def offer_draw(self, game_id: str) -> None:
        self.actions.append(("offer-draw", game_id))

    def resign_game(self, game_id: str) -> None:
        self.actions.append(("resign", game_id))

    def decline_draw(self, game_id: str) -> None:
        self.actions.append(("decline-draw", game_id))

    def abort_game(self, game_id: str) -> None:
        self.actions.append(("abort", game_id))


class AiCreationClient:
    def __init__(self) -> None:
        self.requests: list[
            tuple[int, str, int | None, int | None]
        ] = []

    def create_ai_game(
        self,
        *,
        level: int,
        color: str,
        clock_limit_seconds: int | None,
        clock_increment_seconds: int | None,
    ) -> str:
        self.requests.append(
            (
                level,
                color,
                clock_limit_seconds,
                clock_increment_seconds,
            )
        )
        return "game1234"


def test_ai_creation_worker_emits_created_game_id() -> None:
    client = AiCreationClient()
    worker = LichessAiGameCreationWorker(
        cast(LichessReadOnlyClient, client),
        level=4,
        color="white",
        clock_limit_seconds=600,
        clock_increment_seconds=5,
    )
    created: list[str] = []
    worker.created.connect(created.append)

    worker.run()

    assert client.requests == [(4, "white", 600, 5)]
    assert created == ["game1234"]


def test_game_action_worker_submits_draw_and_resignation() -> None:
    client = GameActionClient()
    succeeded: list[tuple[str, str]] = []

    actions: tuple[LichessGameAction, ...] = (
        "offer-draw",
        "accept-draw",
        "decline-draw",
        "abort",
        "resign",
    )
    for action in actions:
        worker = LichessGameActionWorker(
            cast(LichessReadOnlyClient, client),
            "game1234",
            action,
        )
        worker.succeeded.connect(
            lambda game_id, completed_action: succeeded.append(
                (game_id, completed_action)
            )
        )
        worker.run()

    assert client.actions == [
        ("offer-draw", "game1234"),
        ("offer-draw", "game1234"),
        ("decline-draw", "game1234"),
        ("abort", "game1234"),
        ("resign", "game1234"),
    ]
    assert succeeded == [
        ("game1234", "offer-draw"),
        ("game1234", "accept-draw"),
        ("game1234", "decline-draw"),
        ("game1234", "abort"),
        ("game1234", "resign"),
    ]


def test_move_submission_worker_emits_acceptance() -> None:
    client = SubmissionClient(
        server_moves=(),
        ongoing_moves=("e2e4",),
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

    assert client.submissions == [("game1234", "e2e4")]
    assert accepted == [("game1234", "e2e4")]


def test_successful_submission_is_rejected_after_time_forfeit() -> None:
    client = SubmissionClient(
        status="outoftime",
        winner="black",
        result="0-1",
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

    assert len(rejected) == 1
    assert "outoftime; result 0-1" in rejected[0][2]


def test_polling_submission_confirms_by_authoritative_fen() -> None:
    confirmed_board = chess.Board()
    confirmed_board.push_uci("e2e4")
    expected_board = confirmed_board.copy()
    expected_board.push_uci("e7e5")
    client = SubmissionClient(
        server_moves=("e2e4", "e7e5"),
    )
    worker = LichessMoveSubmissionWorker(
        cast(LichessReadOnlyClient, client),
        "game1234",
        "e7e5",
        (),
        confirmed_fen=confirmed_board.fen(),
    )
    accepted: list[tuple[str, str]] = []
    worker.accepted.connect(
        lambda game_id, uci: accepted.append((game_id, uci))
    )

    worker.run()

    assert accepted == [("game1234", "e7e5")]
    assert expected_board.fen() == client.get_game_snapshot(
        "game1234"
    ).game.state.fen


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


def test_move_submission_worker_keeps_unchanged_active_position_uncertain(
) -> None:
    client = SubmissionClient(
        submit_error=LichessConnectionError("Response was lost."),
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
    assert "does not contain the move yet" in uncertain[0][2]


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
    client = PollingFallbackClient()
    worker = LichessConnectionWorker(
        cast(LichessReadOnlyClient, client),
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
    snapshot = cast(LichessPollingSnapshot, game_events[0])
    assert client.snapshot_calls == ["oozTO3xk"]
    assert snapshot.clocks_available
    assert snapshot.game.state.white_time_ms == 173000
    assert snapshot.game.state.black_time_ms == 604000


def test_active_polling_accepts_ongoing_position_one_move_ahead_of_export(
) -> None:
    before = chess.Board()
    for uci in ("e2e4", "e7e5", "g1f3", "f8d6", "f1c4"):
        before.push_uci(uci)
    after = before.copy()
    after.push_uci("b8c6")
    worker = LichessConnectionWorker(
        cast(LichessReadOnlyClient, PollingFallbackClient())
    )
    before_game = OngoingGame(
        "v0qtcftC",
        "v0qtcftCabcd",
        "white",
        before.fen(),
        "Stockfish level 1",
        False,
        "f1c4",
        343000,
    )
    after_game = OngoingGame(
        "v0qtcftC",
        "v0qtcftCabcd",
        "white",
        after.fen(),
        "Stockfish level 1",
        True,
        "b8c6",
        343000,
    )
    stale_export = LichessPollingSnapshot(
        LichessGameFull(
            "v0qtcftC",
            chess.STARTING_FEN,
            LichessPlayer("white", "Meine", 1661),
            LichessPlayer("black", "Stockfish level 1", None),
            LichessGameState(
                ("e2e4", "e7e5", "g1f3", "f8d6", "f1c4"),
                before.fen(),
                "black",
                343360,
                603390,
                5000,
                5000,
                "started",
                None,
                "*",
            ),
        ),
        True,
        1795272166192,
    )
    worker._polling_snapshot(before_game)

    snapshot = worker._active_polling_snapshot(after_game, stale_export)

    assert snapshot.game.state.moves == ("b8c6",)
    assert snapshot.game.state.fen == after.fen()
    assert snapshot.game.state.turn == "white"
    assert snapshot.game.state.white_time_ms == 343000
    assert snapshot.game.state.black_time_ms == 603390
    assert snapshot.clocks_available


def test_active_polling_accepts_position_three_moves_ahead_of_export() -> None:
    export_board = chess.Board()
    for uci in ("e2e4", "e7e5", "c2c4", "b8c6", "d2d3"):
        export_board.push_uci(uci)
    previous_board = export_board.copy()
    for uci in ("f8b4", "c1d2"):
        previous_board.push_uci(uci)
    current_board = previous_board.copy()
    current_board.push_uci("d8e7")
    worker = LichessConnectionWorker(
        cast(LichessReadOnlyClient, PollingFallbackClient())
    )
    previous_game = OngoingGame(
        "9YQEOn3r",
        "9YQEOn3rabcd",
        "white",
        previous_board.fen(),
        "Stockfish level 1",
        False,
        "c1d2",
        541000,
    )
    current_game = OngoingGame(
        "9YQEOn3r",
        "9YQEOn3rabcd",
        "white",
        current_board.fen(),
        "Stockfish level 1",
        True,
        "d8e7",
        541000,
    )
    stale_export = LichessPollingSnapshot(
        LichessGameFull(
            "9YQEOn3r",
            chess.STARTING_FEN,
            LichessPlayer("white", "Meine", 1661),
            LichessPlayer("black", "Stockfish level 1", None),
            LichessGameState(
                ("e2e4", "e7e5", "c2c4", "b8c6", "d2d3"),
                export_board.fen(),
                "black",
                541000,
                602000,
                5000,
                5000,
                "started",
                None,
                "*",
            ),
        ),
        True,
    )
    worker._polling_snapshot(previous_game)

    snapshot = worker._active_polling_snapshot(current_game, stale_export)

    assert snapshot.game.state.moves == ("d8e7",)
    assert snapshot.game.state.fen == current_board.fen()
    assert snapshot.game.state.turn == "white"
    assert snapshot.game.state.white_time_ms == 541000
    assert not snapshot.clocks_available


def test_reconnect_accepts_current_position_when_export_is_one_move_behind(
) -> None:
    before = chess.Board()
    for uci in ("e2e4", "e7e5", "g1f3", "f8d6", "f1c4"):
        before.push_uci(uci)
    after = before.copy()
    after.push_uci("b8c6")
    worker = LichessConnectionWorker(
        cast(LichessReadOnlyClient, PollingFallbackClient())
    )
    game = OngoingGame(
        "v0qtcftC",
        "v0qtcftCabcd",
        "white",
        after.fen(),
        "Stockfish level 1",
        True,
        "b8c6",
        343000,
    )
    stale_export = LichessPollingSnapshot(
        LichessGameFull(
            "v0qtcftC",
            chess.STARTING_FEN,
            LichessPlayer("white", "Meine", 1661),
            LichessPlayer("black", "Stockfish level 1", None),
            LichessGameState(
                ("e2e4", "e7e5", "g1f3", "f8d6", "f1c4"),
                before.fen(),
                "black",
                343360,
                603390,
                5000,
                5000,
                "started",
                None,
                "*",
            ),
        ),
        True,
        1795272166192,
    )

    snapshot = worker._active_polling_snapshot(game, stale_export)

    assert snapshot.game.initial_fen == after.fen()
    assert snapshot.game.state.moves == ()
    assert snapshot.game.state.fen == after.fen()
    assert snapshot.game.state.white_time_ms == 343000
    assert snapshot.clocks_available


def test_polling_fetches_terminal_move_when_game_disappears() -> None:
    moves = (
        "e2e4",
        "e7e5",
        "c2c3",
        "d8h4",
        "d2d3",
        "f8c5",
        "g1f3",
    )
    before_mate = chess.Board()
    for uci in moves:
        before_mate.push_uci(uci)
    after_mate = before_mate.copy()
    after_mate.push_uci("h4f2")

    class FinishedGameClient:
        def __init__(self) -> None:
            self.ongoing_calls = 0
            self.snapshot_calls: list[str] = []

        def get_ongoing_games(self) -> tuple[OngoingGame, ...]:
            self.ongoing_calls += 1
            if self.ongoing_calls > 1:
                return ()
            return (
                OngoingGame(
                    "Fz54fOC3",
                    "Fz54fOC3abcd",
                    "white",
                    before_mate.fen(),
                    "Stockfish level 1",
                    False,
                    "g1f3",
                ),
            )

        def get_game_snapshot(
            self,
            game_id: str,
        ) -> LichessPollingSnapshot:
            self.snapshot_calls.append(game_id)
            if self.ongoing_calls == 1:
                state = LichessGameState(
                    moves,
                    before_mate.fen(),
                    "black",
                    120000,
                    300000,
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
                        LichessPlayer("white", "Meine", None),
                        LichessPlayer(
                            "black",
                            "Stockfish level 1",
                            None,
                        ),
                        state,
                    ),
                    True,
                )
            state = LichessGameState(
                (*moves, "h4f2"),
                after_mate.fen(),
                "white",
                0,
                0,
                0,
                0,
                "mate",
                "black",
                "0-1",
            )
            return LichessPollingSnapshot(
                LichessGameFull(
                    game_id,
                    chess.STARTING_FEN,
                    LichessPlayer("white", "Meine", None),
                    LichessPlayer(
                        "black",
                        "Stockfish level 1",
                        None,
                    ),
                    state,
                ),
                False,
            )

    client = FinishedGameClient()
    worker = LichessConnectionWorker(
        cast(LichessReadOnlyClient, client)
    )
    game_events: list[object] = []
    worker.game_event_received.connect(game_events.append)

    assert worker._poll_ongoing_game(None) == "Fz54fOC3"
    assert worker._poll_ongoing_game("Fz54fOC3") is None

    final_snapshot = cast(LichessPollingSnapshot, game_events[-1])
    assert client.snapshot_calls == ["Fz54fOC3", "Fz54fOC3"]
    assert final_snapshot.game.initial_fen == before_mate.fen()
    assert final_snapshot.game.state.moves == ("h4f2",)
    assert final_snapshot.game.state.fen == after_mate.fen()
    assert final_snapshot.game.state.status == "mate"
    assert final_snapshot.game.state.result == "0-1"
    assert worker._polling_game_id is None


def test_polling_retains_cursor_while_terminal_export_is_still_active(
) -> None:
    class LaggingTerminalClient:
        def __init__(self) -> None:
            self.ongoing_calls = 0

        def get_ongoing_games(self) -> tuple[OngoingGame, ...]:
            self.ongoing_calls += 1
            if self.ongoing_calls > 1:
                return ()
            return (
                OngoingGame(
                    "lagging1",
                    "lagging1abcd",
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
            return LichessPollingSnapshot(
                LichessGameFull(
                    game_id,
                    chess.STARTING_FEN,
                    LichessPlayer("white", "Meine", None),
                    LichessPlayer("black", "Stockfish level 1", None),
                    LichessGameState(
                        (),
                        chess.STARTING_FEN,
                        "white",
                        600000,
                        600000,
                        5000,
                        5000,
                        "started",
                        None,
                        "*",
                    ),
                ),
                True,
            )

    worker = LichessConnectionWorker(
        cast(LichessReadOnlyClient, LaggingTerminalClient())
    )
    statuses: list[str] = []
    ongoing_events: list[tuple[OngoingGame, ...]] = []
    worker.status_changed.connect(statuses.append)
    worker.ongoing_games_received.connect(ongoing_events.append)

    assert worker._poll_ongoing_game(None) == "lagging1"
    assert worker._poll_ongoing_game("lagging1") == "lagging1"
    assert worker._polling_game_id == "lagging1"
    assert len(ongoing_events) == 1
    assert ongoing_events[0][0].game_id == "lagging1"
    assert statuses == [
        "Lichess game ended; waiting for its final authoritative snapshot."
    ]


def test_terminal_timeout_rolls_back_uncommitted_local_move() -> None:
    official_moves = ("e2e4", "e7e5")
    official_board = chess.Board()
    for uci in official_moves:
        official_board.push_uci(uci)
    optimistic_board = official_board.copy()
    optimistic_board.push_uci("g1f3")

    class TimedOutGameClient:
        def __init__(self) -> None:
            self.ongoing_calls = 0

        def get_ongoing_games(self) -> tuple[OngoingGame, ...]:
            self.ongoing_calls += 1
            if self.ongoing_calls > 1:
                return ()
            return (
                OngoingGame(
                    "timeout1",
                    "timeout1abcd",
                    "white",
                    official_board.fen(),
                    "Stockfish level 1",
                    True,
                    "e7e5",
                ),
            )

        def get_game_snapshot(
            self,
            game_id: str,
        ) -> LichessPollingSnapshot:
            status = "started" if self.ongoing_calls == 1 else "outoftime"
            result = "*" if status == "started" else "0-1"
            winner = None if status == "started" else "black"
            return LichessPollingSnapshot(
                LichessGameFull(
                    game_id,
                    chess.STARTING_FEN,
                    LichessPlayer("white", "Meine", None),
                    LichessPlayer(
                        "black",
                        "Stockfish level 1",
                        None,
                    ),
                    LichessGameState(
                        official_moves,
                        official_board.fen(),
                        "white",
                        0,
                        610000,
                        5000,
                        5000,
                        status,
                        winner,
                        result,
                    ),
                ),
                True,
            )

    client = TimedOutGameClient()
    worker = LichessConnectionWorker(
        cast(LichessReadOnlyClient, client)
    )
    game_events: list[object] = []
    worker.game_event_received.connect(game_events.append)

    assert worker._poll_ongoing_game(None) == "timeout1"
    assert worker.note_local_move(
        "timeout1",
        "g1f3",
        optimistic_board.fen(),
    )
    assert worker._poll_ongoing_game("timeout1") is None

    final_snapshot = cast(LichessPollingSnapshot, game_events[-1])
    assert final_snapshot.game.state.moves == ()
    assert final_snapshot.game.state.fen == official_board.fen()
    assert final_snapshot.game.state.status == "outoftime"
    assert final_snapshot.game.state.winner == "black"
    assert final_snapshot.game.state.result == "0-1"


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


def test_local_move_advances_polling_cursor_before_opponent_reply() -> None:
    worker = LichessConnectionWorker(
        cast(LichessReadOnlyClient, MultiplePollingGamesClient())
    )
    board = chess.Board()
    before = OngoingGame(
        "jgWZLWs6",
        "jgWZLWs6abcd",
        "white",
        board.fen(),
        "Stockfish level 1",
        True,
        None,
    )
    worker._polling_snapshot(before)

    board.push_uci("e2e4")
    assert worker.note_local_move(
        "jgWZLWs6",
        "e2e4",
        board.fen(),
    )
    board.push_uci("e7e5")
    after = OngoingGame(
        "jgWZLWs6",
        "jgWZLWs6abcd",
        "white",
        board.fen(),
        "Stockfish level 1",
        True,
        "e7e5",
    )
    current = worker._polling_snapshot(after)

    assert current.game.state.moves == ("e2e4", "e7e5")
    assert current.game.state.fen == board.fen()
