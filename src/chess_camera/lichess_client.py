import json
import os
import urllib.error
import urllib.request
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Callable, ContextManager, Literal, Protocol, cast

import chess

PlayerColor = Literal["white", "black"]
AccountEventType = Literal["gameStart", "gameFinish"]
_USER_AGENT = (
    "ChessProjectorCamera/0.1 "
    "(https://github.com/Meine1964/Chess)"
)


class LichessError(RuntimeError):
    pass


class LichessConfigurationError(LichessError):
    pass


class LichessConnectionError(LichessError):
    pass


class LichessProtocolError(LichessError):
    pass


class LichessHTTPError(LichessError):
    def __init__(
        self,
        status: int,
        *,
        retry_after_seconds: int | None = None,
    ) -> None:
        self.status = status
        self.retry_after_seconds = retry_after_seconds
        message = f"Lichess request failed with HTTP {status}."
        if retry_after_seconds is not None:
            message += f" Retry after {retry_after_seconds} seconds."
        super().__init__(message)


@dataclass(frozen=True)
class AccountGameEvent:
    event_type: AccountEventType
    game_id: str
    full_id: str
    color: PlayerColor
    initial_fen: str
    opponent_name: str
    is_my_turn: bool


@dataclass(frozen=True)
class LichessPlayer:
    color: PlayerColor
    name: str
    rating: int | None


@dataclass(frozen=True)
class LichessGameState:
    moves: tuple[str, ...]
    fen: str
    turn: PlayerColor
    white_time_ms: int
    black_time_ms: int
    white_increment_ms: int
    black_increment_ms: int
    status: str
    winner: PlayerColor | None
    result: str


@dataclass(frozen=True)
class LichessGameFull:
    game_id: str
    initial_fen: str
    white: LichessPlayer
    black: LichessPlayer
    state: LichessGameState


GameStreamEvent = LichessGameFull | LichessGameState


@dataclass(frozen=True)
class OngoingGame:
    game_id: str
    full_id: str
    color: PlayerColor
    current_fen: str
    opponent_name: str
    is_my_turn: bool
    last_move: str | None = None


@dataclass(frozen=True)
class LichessPollingSnapshot:
    game: LichessGameFull
    clocks_available: bool


class NdjsonTransport(Protocol):
    def open_stream(
        self,
        path: str,
        headers: Mapping[str, str],
    ) -> ContextManager[Iterable[bytes | str]]: ...

    def get_json(
        self,
        path: str,
        headers: Mapping[str, str],
    ) -> Mapping[str, object]: ...

    def post_json(
        self,
        path: str,
        headers: Mapping[str, str],
    ) -> Mapping[str, object]: ...


class UrllibNdjsonTransport:
    def __init__(
        self,
        *,
        base_url: str = "https://lichess.org",
        timeout_seconds: float = 15.0,
    ) -> None:
        if not base_url.startswith(("https://", "http://")):
            raise ValueError("The Lichess base URL must use HTTP or HTTPS.")
        if timeout_seconds <= 0:
            raise ValueError("The HTTP timeout must be positive.")
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds

    @contextmanager
    def open_stream(
        self,
        path: str,
        headers: Mapping[str, str],
    ) -> Iterator[Iterable[bytes | str]]:
        if not path.startswith("/"):
            raise ValueError("Lichess request paths must start with '/'.")
        request = urllib.request.Request(
            f"{self._base_url}{path}",
            headers=dict(headers),
            method="GET",
        )
        try:
            response = urllib.request.urlopen(
                request,
                timeout=self._timeout_seconds,
            )
        except urllib.error.HTTPError as error:
            retry_after = (
                _retry_after_seconds(error.headers.get("Retry-After"))
                if error.code == 429
                else None
            )
            raise LichessHTTPError(
                error.code,
                retry_after_seconds=retry_after,
            ) from error
        except OSError as error:
            raise LichessConnectionError(
                "Could not connect to Lichess."
            ) from error
        try:
            yield cast(Iterable[bytes | str], response)
        finally:
            response.close()

    def get_json(
        self,
        path: str,
        headers: Mapping[str, str],
    ) -> Mapping[str, object]:
        if not path.startswith("/"):
            raise ValueError("Lichess request paths must start with '/'.")
        request = urllib.request.Request(
            f"{self._base_url}{path}",
            headers=dict(headers),
            method="GET",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self._timeout_seconds,
            ) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as error:
            retry_after = (
                _retry_after_seconds(error.headers.get("Retry-After"))
                if error.code == 429
                else None
            )
            raise LichessHTTPError(
                error.code,
                retry_after_seconds=retry_after,
            ) from error
        except OSError as error:
            raise LichessConnectionError(
                "Could not connect to Lichess."
            ) from error
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise LichessProtocolError(
                "Lichess returned invalid JSON."
            ) from error
        if not isinstance(payload, dict):
            raise LichessProtocolError(
                "A Lichess JSON response must be an object."
            )
        return cast(dict[str, object], payload)

    def post_json(
        self,
        path: str,
        headers: Mapping[str, str],
    ) -> Mapping[str, object]:
        if not path.startswith("/"):
            raise ValueError("Lichess request paths must start with '/'.")
        request = urllib.request.Request(
            f"{self._base_url}{path}",
            data=b"",
            headers=dict(headers),
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self._timeout_seconds,
            ) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as error:
            retry_after = (
                _retry_after_seconds(error.headers.get("Retry-After"))
                if error.code == 429
                else None
            )
            raise LichessHTTPError(
                error.code,
                retry_after_seconds=retry_after,
            ) from error
        except OSError as error:
            raise LichessConnectionError(
                "Could not confirm the Lichess move submission."
            ) from error
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise LichessProtocolError(
                "Lichess returned invalid JSON after move submission."
            ) from error
        if not isinstance(payload, dict):
            raise LichessProtocolError(
                "A Lichess JSON response must be an object."
            )
        return cast(dict[str, object], payload)


class LichessReadOnlyClient:
    def __init__(
        self,
        token: str,
        *,
        transport: NdjsonTransport | None = None,
        user_agent: str = _USER_AGENT,
    ) -> None:
        if not token or token.isspace():
            raise LichessConfigurationError(
                "LICHESS_TOKEN is not configured."
            )
        if not user_agent.strip():
            raise ValueError("A Lichess User-Agent is required.")
        self._token = token
        self._transport = (
            UrllibNdjsonTransport() if transport is None else transport
        )
        self._headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/x-ndjson",
            "User-Agent": user_agent,
        }

    @classmethod
    def from_environment(
        cls,
        *,
        transport: NdjsonTransport | None = None,
        user_agent: str = _USER_AGENT,
    ) -> "LichessReadOnlyClient":
        return cls(
            os.environ.get("LICHESS_TOKEN", ""),
            transport=transport,
            user_agent=user_agent,
        )

    def stream_account_events(
        self,
        *,
        on_open: Callable[[], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> Iterator[AccountGameEvent]:
        with self._transport.open_stream(
            "/api/stream/event",
            self._headers,
        ) as lines:
            if on_open is not None:
                on_open()
            for payload in _ndjson_objects(
                lines,
                should_stop=should_stop,
            ):
                event_type = payload.get("type")
                if event_type not in ("gameStart", "gameFinish"):
                    continue
                yield _parse_account_game_event(
                    payload,
                    cast(AccountEventType, event_type),
                )

    def stream_game(
        self,
        game_id: str,
        *,
        on_open: Callable[[], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> Iterator[GameStreamEvent]:
        normalized_game_id = _normalize_game_id(game_id)
        initial_fen: str | None = None
        with self._transport.open_stream(
            f"/api/board/game/stream/{normalized_game_id}",
            self._headers,
        ) as lines:
            if on_open is not None:
                on_open()
            for payload in _ndjson_objects(
                lines,
                should_stop=should_stop,
            ):
                event_type = payload.get("type")
                if event_type == "gameFull":
                    if initial_fen is not None:
                        raise LichessProtocolError(
                            "The game stream sent more than one gameFull event."
                        )
                    full = _parse_game_full(payload)
                    if full.game_id != normalized_game_id:
                        raise LichessProtocolError(
                            "The game stream returned a different game ID."
                        )
                    initial_fen = full.initial_fen
                    yield full
                elif event_type == "gameState":
                    if initial_fen is None:
                        raise LichessProtocolError(
                            "The game stream sent gameState before gameFull."
                        )
                    yield _parse_game_state(payload, initial_fen)

    def get_ongoing_games(self) -> tuple[OngoingGame, ...]:
        payload = self._transport.get_json(
            "/api/account/playing?nb=10",
            {**self._headers, "Accept": "application/json"},
        )
        games = payload.get("nowPlaying")
        if not isinstance(games, list):
            raise LichessProtocolError(
                "The ongoing-games response is missing nowPlaying."
            )
        return tuple(_parse_ongoing_game(game) for game in games)

    def get_game_snapshot(
        self,
        game_id: str,
    ) -> LichessPollingSnapshot:
        normalized_game_id = _normalize_game_id(game_id)
        events = self.stream_game(normalized_game_id)
        try:
            try:
                first_event = next(events)
            except StopIteration as error:
                raise LichessProtocolError(
                    "The Board API game stream closed before gameFull."
                ) from error
            if not isinstance(first_event, LichessGameFull):
                raise LichessProtocolError(
                    "The Board API game stream did not begin with gameFull."
                )
            return LichessPollingSnapshot(first_event, True)
        finally:
            close = getattr(events, "close", None)
            if callable(close):
                close()

    def submit_move(self, game_id: str, uci: str) -> None:
        normalized_game_id = _normalize_game_id(game_id)
        try:
            move = chess.Move.from_uci(uci)
        except ValueError as error:
            raise ValueError(f"Invalid UCI move '{uci}'.") from error
        normalized_uci = move.uci()
        payload = self._transport.post_json(
            f"/api/board/game/{normalized_game_id}/move/{normalized_uci}",
            {**self._headers, "Accept": "application/json"},
        )
        if payload.get("ok") is not True:
            raise LichessProtocolError(
                "Lichess did not confirm the submitted move."
            )

    def offer_draw(self, game_id: str) -> None:
        self._submit_game_action(
            game_id,
            "draw/yes",
            "draw offer",
        )

    def resign_game(self, game_id: str) -> None:
        self._submit_game_action(
            game_id,
            "resign",
            "resignation",
        )

    def _submit_game_action(
        self,
        game_id: str,
        path_suffix: str,
        description: str,
    ) -> None:
        normalized_game_id = _normalize_game_id(game_id)
        payload = self._transport.post_json(
            f"/api/board/game/{normalized_game_id}/{path_suffix}",
            {**self._headers, "Accept": "application/json"},
        )
        if payload.get("ok") is not True:
            raise LichessProtocolError(
                f"Lichess did not confirm the {description}."
            )


def reconnect_delays(
    attempts: int,
    *,
    initial_seconds: int = 1,
    maximum_seconds: int = 30,
) -> tuple[int, ...]:
    if attempts < 0:
        raise ValueError("Reconnect attempts cannot be negative.")
    if initial_seconds < 1 or maximum_seconds < initial_seconds:
        raise ValueError("Reconnect delay bounds are invalid.")
    return tuple(
        min(maximum_seconds, initial_seconds * (2**attempt))
        for attempt in range(attempts)
    )


def _ndjson_objects(
    lines: Iterable[bytes | str],
    *,
    should_stop: Callable[[], bool] | None = None,
) -> Iterator[dict[str, object]]:
    iterator = iter(lines)
    while True:
        try:
            line = next(iterator)
        except StopIteration:
            return
        except OSError as error:
            raise LichessConnectionError(
                "The Lichess stream was interrupted."
            ) from error
        if should_stop is not None and should_stop():
            return
        try:
            text = line.decode("utf-8") if isinstance(line, bytes) else line
            if not text.strip():
                continue
            payload = json.loads(text)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise LichessProtocolError(
                "Lichess returned invalid NDJSON."
            ) from error
        if not isinstance(payload, dict):
            raise LichessProtocolError(
                "A Lichess stream event must be a JSON object."
            )
        yield cast(dict[str, object], payload)


def _parse_account_game_event(
    payload: Mapping[str, object],
    event_type: AccountEventType,
) -> AccountGameEvent:
    game = _required_mapping(payload, "game")
    color = _required_color(game, "color")
    opponent = _required_mapping(game, "opponent")
    return AccountGameEvent(
        event_type,
        _required_string(game, "gameId"),
        _required_string(game, "fullId"),
        color,
        _required_string(game, "fen"),
        _required_string(opponent, "username"),
        _required_bool(game, "isMyTurn"),
    )


def _parse_ongoing_game(payload: object) -> OngoingGame:
    if not isinstance(payload, dict):
        raise LichessProtocolError(
            "An ongoing-game entry must be an object."
        )
    game = cast(Mapping[str, object], payload)
    opponent = _required_mapping(game, "opponent")
    last_move = _required_string(game, "lastMove")
    return OngoingGame(
        _required_string(game, "gameId"),
        _required_string(game, "fullId"),
        _required_color(game, "color"),
        _required_string(game, "fen"),
        _required_string(opponent, "username"),
        _required_bool(game, "isMyTurn"),
        last_move or None,
    )


def _parse_game_full(payload: Mapping[str, object]) -> LichessGameFull:
    variant = _required_mapping(payload, "variant")
    if _required_string(variant, "key") != "standard":
        raise LichessProtocolError(
            "Only standard chess games are supported."
        )
    initial_fen = _normalize_initial_fen(
        _required_string(payload, "initialFen")
    )
    return LichessGameFull(
        _required_string(payload, "id"),
        initial_fen,
        _parse_player(_required_mapping(payload, "white"), "white"),
        _parse_player(_required_mapping(payload, "black"), "black"),
        _parse_game_state(
            _required_mapping(payload, "state"),
            initial_fen,
        ),
    )


def _parse_player(
    payload: Mapping[str, object],
    color: PlayerColor,
) -> LichessPlayer:
    rating = payload.get("rating")
    if rating is not None and (
        not isinstance(rating, int) or isinstance(rating, bool)
    ):
        raise LichessProtocolError("A player rating must be an integer.")
    return LichessPlayer(
        color,
        _required_string(payload, "name"),
        rating,
    )


def _parse_game_state(
    payload: Mapping[str, object],
    initial_fen: str,
) -> LichessGameState:
    moves_text = _required_string(payload, "moves")
    moves = tuple(moves_text.split()) if moves_text else ()
    try:
        board = chess.Board(initial_fen)
        for uci in moves:
            board.push(board.parse_uci(uci))
    except ValueError as error:
        raise LichessProtocolError(
            "The Lichess move list is invalid for its initial FEN."
        ) from error
    winner_value = payload.get("winner")
    winner: PlayerColor | None
    if winner_value is None:
        winner = None
    elif winner_value in ("white", "black"):
        winner = cast(PlayerColor, winner_value)
    else:
        raise LichessProtocolError("The Lichess winner value is invalid.")
    status = _required_string(payload, "status")
    return LichessGameState(
        moves,
        board.fen(),
        "white" if board.turn == chess.WHITE else "black",
        _required_int(payload, "wtime"),
        _required_int(payload, "btime"),
        _required_int(payload, "winc"),
        _required_int(payload, "binc"),
        status,
        winner,
        _game_result(status, winner),
    )


def _game_result(
    status: str,
    winner: PlayerColor | None,
) -> str:
    if winner == "white":
        return "1-0"
    if winner == "black":
        return "0-1"
    if status in {
        "draw",
        "stalemate",
        "repetition",
        "insufficient",
        "fiftymoves",
    }:
        return "1/2-1/2"
    return "*"


def _normalize_game_id(game_id: str) -> str:
    normalized = game_id.strip()
    if not normalized or not normalized.isalnum():
        raise ValueError("A Lichess game ID must be alphanumeric.")
    return normalized


def _normalize_initial_fen(value: str) -> str:
    if value == "startpos":
        return chess.STARTING_FEN
    try:
        return chess.Board(value).fen()
    except ValueError as error:
        raise LichessProtocolError(
            "The Lichess initial FEN is invalid."
        ) from error


def _required_mapping(
    payload: Mapping[str, object],
    key: str,
) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise LichessProtocolError(f"Lichess event field '{key}' is invalid.")
    return cast(Mapping[str, object], value)


def _required_string(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise LichessProtocolError(f"Lichess event field '{key}' is invalid.")
    return value


def _required_bool(payload: Mapping[str, object], key: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise LichessProtocolError(f"Lichess event field '{key}' is invalid.")
    return value


def _required_int(payload: Mapping[str, object], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise LichessProtocolError(f"Lichess event field '{key}' is invalid.")
    if value < 0:
        raise LichessProtocolError(
            f"Lichess event field '{key}' cannot be negative."
        )
    return value


def _required_color(
    payload: Mapping[str, object],
    key: str,
) -> PlayerColor:
    value = payload.get(key)
    if value not in ("white", "black"):
        raise LichessProtocolError(f"Lichess event field '{key}' is invalid.")
    return cast(PlayerColor, value)


def _retry_after_seconds(value: str | None) -> int:
    if value is None:
        return 60
    try:
        seconds = int(value)
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return 60
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        seconds = round(
            (retry_at - datetime.now(timezone.utc)).total_seconds()
        )
    return max(1, seconds)
