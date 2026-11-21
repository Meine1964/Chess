import json
import urllib.error
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from email.message import Message

import chess
import pytest

from src.chess_camera.lichess_client import (
    LichessConfigurationError,
    LichessConnectionError,
    LichessGameFull,
    LichessGameState,
    LichessHTTPError,
    LichessPollingSnapshot,
    LichessProtocolError,
    LichessReadOnlyClient,
    reconnect_delays,
)


class FakeTransport:
    def __init__(
        self,
        lines: Mapping[str, Iterable[bytes | str]],
        json_responses: Mapping[str, Mapping[str, object]] | None = None,
    ) -> None:
        self._lines = lines
        self._json_responses = (
            {} if json_responses is None else json_responses
        )
        self.requests: list[tuple[str, Mapping[str, str]]] = []
        self.post_responses: dict[str, Mapping[str, object]] = {}

    @contextmanager
    def open_stream(
        self,
        path: str,
        headers: Mapping[str, str],
    ) -> Iterator[Iterable[bytes | str]]:
        self.requests.append((path, dict(headers)))
        yield self._lines[path]

    def get_json(
        self,
        path: str,
        headers: Mapping[str, str],
    ) -> Mapping[str, object]:
        self.requests.append((path, dict(headers)))
        return self._json_responses[path]

    def post_json(
        self,
        path: str,
        headers: Mapping[str, str],
    ) -> Mapping[str, object]:
        self.requests.append((path, dict(headers)))
        return self.post_responses[path]


def _line(payload: object) -> bytes:
    return (json.dumps(payload) + "\n").encode()


def _state(moves: str, *, status: str = "started") -> dict[str, object]:
    return {
        "type": "gameState",
        "moves": moves,
        "wtime": 299000,
        "btime": 298000,
        "winc": 2000,
        "binc": 2000,
        "status": status,
    }


def test_account_stream_uses_auth_headers_and_ignores_keepalives() -> None:
    event = {
        "type": "gameStart",
        "game": {
            "gameId": "p8Jy2ayw",
            "fullId": "p8Jy2ayw91cB",
            "fen": chess.STARTING_FEN,
            "color": "white",
            "opponent": {"username": "Aaron"},
            "isMyTurn": True,
        },
    }
    transport = FakeTransport(
        {
            "/api/stream/event": (
                b"\n",
                _line({"type": "challenge"}),
                _line(event),
            )
        }
    )
    client = LichessReadOnlyClient(
        "secret-token",
        transport=transport,
        user_agent="Chess test client",
    )

    events = tuple(client.stream_account_events())

    assert len(events) == 1
    assert events[0].game_id == "p8Jy2ayw"
    assert events[0].color == "white"
    path, headers = transport.requests[0]
    assert path == "/api/stream/event"
    assert headers == {
        "Authorization": "Bearer secret-token",
        "Accept": "application/x-ndjson",
        "User-Agent": "Chess test client",
    }


def test_submit_move_uses_board_api_and_auth_headers() -> None:
    path = "/api/board/game/p8Jy2ayw/move/e2e4"
    transport = FakeTransport({})
    transport.post_responses[path] = {"ok": True}
    client = LichessReadOnlyClient(
        "secret-token",
        transport=transport,
        user_agent="Chess test client",
    )

    client.submit_move("p8Jy2ayw", "e2e4")

    assert transport.requests == [
        (
            path,
            {
                "Authorization": "Bearer secret-token",
                "Accept": "application/json",
                "User-Agent": "Chess test client",
            },
        )
    ]


def test_submit_move_requires_positive_confirmation() -> None:
    path = "/api/board/game/p8Jy2ayw/move/e2e4"
    transport = FakeTransport({})
    transport.post_responses[path] = {"ok": False}
    client = LichessReadOnlyClient(
        "secret-token",
        transport=transport,
    )

    with pytest.raises(
        LichessProtocolError,
        match="did not confirm",
    ):
        client.submit_move("p8Jy2ayw", "e2e4")


@pytest.mark.parametrize(
    ("method_name", "path"),
    (
        ("offer_draw", "/api/board/game/p8Jy2ayw/draw/yes"),
        ("resign_game", "/api/board/game/p8Jy2ayw/resign"),
    ),
)
def test_game_controls_use_board_api(
    method_name: str,
    path: str,
) -> None:
    transport = FakeTransport({})
    transport.post_responses[path] = {"ok": True}
    client = LichessReadOnlyClient(
        "secret-token",
        transport=transport,
        user_agent="Chess test client",
    )

    getattr(client, method_name)("p8Jy2ayw")

    assert transport.requests == [
        (
            path,
            {
                "Authorization": "Bearer secret-token",
                "Accept": "application/json",
                "User-Agent": "Chess test client",
            },
        )
    ]


def test_stream_open_callback_and_keepalive_cancellation() -> None:
    event = {
        "type": "gameStart",
        "game": {
            "gameId": "p8Jy2ayw",
            "fullId": "p8Jy2ayw91cB",
            "fen": chess.STARTING_FEN,
            "color": "white",
            "opponent": {"username": "Aaron"},
            "isMyTurn": True,
        },
    }
    transport = FakeTransport(
        {"/api/stream/event": (b"\n", _line(event))}
    )
    client = LichessReadOnlyClient("secret-token", transport=transport)
    opened: list[bool] = []
    stop_checks = 0

    def should_stop() -> bool:
        nonlocal stop_checks
        stop_checks += 1
        return stop_checks > 1

    events = tuple(
        client.stream_account_events(
            on_open=lambda: opened.append(True),
            should_stop=should_stop,
        )
    )

    assert opened == [True]
    assert stop_checks == 2
    assert events == ()


def test_stream_read_timeout_is_a_reconnectable_connection_error() -> None:
    def timed_out_lines() -> Iterator[bytes]:
        yield b"\n"
        raise TimeoutError("The read operation timed out")

    transport = FakeTransport(
        {"/api/stream/event": timed_out_lines()}
    )
    client = LichessReadOnlyClient("secret-token", transport=transport)

    with pytest.raises(
        LichessConnectionError,
        match="Lichess stream was interrupted",
    ):
        tuple(client.stream_account_events())


def test_game_stream_reconstructs_position_and_clocks() -> None:
    game_full = {
        "type": "gameFull",
        "id": "qP8XaFtN",
        "variant": {"key": "standard"},
        "initialFen": "startpos",
        "white": {"name": "Kenneth", "rating": 566},
        "black": {"name": "Salma", "rating": 662},
        "state": _state("e2e4"),
    }
    transport = FakeTransport(
        {
            "/api/board/game/stream/qP8XaFtN": (
                _line(game_full),
                b"\n",
                _line(_state("e2e4 e7e5")),
                _line({"type": "chatLine", "text": "hello"}),
            )
        }
    )
    client = LichessReadOnlyClient("secret-token", transport=transport)

    events = tuple(client.stream_game("qP8XaFtN"))

    assert len(events) == 2
    full = events[0]
    state = events[1]
    assert isinstance(full, LichessGameFull)
    assert full.white.name == "Kenneth"
    assert full.black.color == "black"
    assert full.state.moves == ("e2e4",)
    assert isinstance(state, LichessGameState)
    assert state.moves == ("e2e4", "e7e5")
    assert state.turn == "white"
    assert state.white_time_ms == 299000
    board = chess.Board()
    board.push_uci("e2e4")
    board.push_uci("e7e5")
    assert state.fen == board.fen()
    assert state.result == "*"


def test_polling_fallback_discovers_and_reconstructs_ai_game() -> None:
    game_full = {
        "type": "gameFull",
        "id": "oozTO3xk",
        "variant": {"key": "standard"},
        "initialFen": "startpos",
        "white": {"name": "Meine", "rating": 1500},
        "black": {"name": "Stockfish level 1"},
        "state": _state("e2e4 e7e6 g1f3 d7d5 b1c3 c7c6"),
    }
    transport = FakeTransport(
        {
            "/api/board/game/stream/oozTO3xk": (
                _line(game_full),
            )
        },
        {
            "/api/account/playing?nb=10": {
                "nowPlaying": [
                    {
                        "gameId": "oozTO3xk",
                        "fullId": "oozTO3xkabcd",
                        "color": "white",
                        "fen": chess.STARTING_FEN,
                        "lastMove": "",
                        "opponent": {
                            "username": "Stockfish level 1",
                        },
                        "isMyTurn": True,
                    }
                ],
                "nbMyTurn": 1,
            },
        },
    )
    client = LichessReadOnlyClient("secret-token", transport=transport)

    games = client.get_ongoing_games()
    snapshot = client.get_game_snapshot(games[0].game_id)

    assert games[0].opponent_name == "Stockfish level 1"
    assert isinstance(snapshot, LichessPollingSnapshot)
    assert snapshot.game.black.name == "Stockfish level 1"
    assert snapshot.game.state.moves == (
        "e2e4",
        "e7e6",
        "g1f3",
        "d7d5",
        "b1c3",
        "c7c6",
    )
    assert snapshot.game.state.turn == "white"
    assert snapshot.clocks_available
    assert transport.requests[-1][0] == (
        "/api/board/game/stream/oozTO3xk"
    )


def test_invalid_game_move_list_is_rejected() -> None:
    game_full = {
        "type": "gameFull",
        "id": "qP8XaFtN",
        "variant": {"key": "standard"},
        "initialFen": "startpos",
        "white": {"name": "Kenneth"},
        "black": {"name": "Salma"},
        "state": _state("e2e5"),
    }
    transport = FakeTransport(
        {"/api/board/game/stream/qP8XaFtN": (_line(game_full),)}
    )
    client = LichessReadOnlyClient("secret-token", transport=transport)

    with pytest.raises(
        LichessProtocolError,
        match="move list is invalid",
    ):
        tuple(client.stream_game("qP8XaFtN"))


def test_missing_environment_token_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LICHESS_TOKEN", raising=False)

    with pytest.raises(
        LichessConfigurationError,
        match="LICHESS_TOKEN is not configured",
    ):
        LichessReadOnlyClient.from_environment()


def test_rate_limit_error_has_retry_delay_without_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    headers = Message()
    headers["Retry-After"] = "75"

    def raise_rate_limit(
        _request: object,
        *,
        timeout: float,
    ) -> object:
        assert timeout == 15.0
        raise urllib.error.HTTPError(
            "https://lichess.org/api/stream/event",
            429,
            "Too Many Requests",
            headers,
            None,
        )

    monkeypatch.setattr(urllib.request, "urlopen", raise_rate_limit)
    client = LichessReadOnlyClient("secret-token")

    with pytest.raises(LichessHTTPError) as raised:
        next(client.stream_account_events())

    assert raised.value.status == 429
    assert raised.value.retry_after_seconds == 75
    assert "secret-token" not in str(raised.value)


def test_socket_timeout_while_opening_is_a_connection_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_timeout(
        _request: object,
        *,
        timeout: float,
    ) -> object:
        assert timeout == 15.0
        raise TimeoutError("The read operation timed out")

    monkeypatch.setattr(urllib.request, "urlopen", raise_timeout)
    client = LichessReadOnlyClient("secret-token")

    with pytest.raises(
        LichessConnectionError,
        match="Could not connect to Lichess",
    ):
        next(client.stream_account_events())


def test_reconnect_delays_are_bounded() -> None:
    assert reconnect_delays(7) == (1, 2, 4, 8, 16, 30, 30)
