from collections.abc import Callable
from dataclasses import dataclass
from time import monotonic
from typing import Literal

from .lichess_client import PlayerColor

OfflineClockResult = Literal["1-0", "0-1"]


@dataclass(frozen=True)
class OfflineClockSnapshot:
    white_time_ms: int
    black_time_ms: int
    active_player: PlayerColor
    running: bool
    has_started: bool
    expired_player: PlayerColor | None
    result: OfflineClockResult | None


class OfflineClock:
    def __init__(
        self,
        initial_time_ms: int,
        increment_ms: int,
        *,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        if initial_time_ms <= 0:
            raise ValueError("Initial clock time must be positive.")
        if increment_ms < 0:
            raise ValueError("Clock increment cannot be negative.")
        self._initial_time_ms = initial_time_ms
        self._increment_ms = increment_ms
        self._clock = clock
        self._white_time_ms = initial_time_ms
        self._black_time_ms = initial_time_ms
        self._active_player: PlayerColor = "white"
        self._running_since: float | None = None
        self._has_started = False
        self._expired_player: PlayerColor | None = None

    @property
    def initial_time_ms(self) -> int:
        return self._initial_time_ms

    @property
    def increment_ms(self) -> int:
        return self._increment_ms

    def configure(self, initial_time_ms: int, increment_ms: int) -> None:
        if initial_time_ms <= 0:
            raise ValueError("Initial clock time must be positive.")
        if increment_ms < 0:
            raise ValueError("Clock increment cannot be negative.")
        self._initial_time_ms = initial_time_ms
        self._increment_ms = increment_ms
        self.reset(self._active_player)

    def restore(
        self,
        white_time_ms: int,
        black_time_ms: int,
        active_player: PlayerColor,
        *,
        has_started: bool,
    ) -> None:
        if white_time_ms < 0 or black_time_ms < 0:
            raise ValueError("Restored clock times cannot be negative.")
        self._white_time_ms = white_time_ms
        self._black_time_ms = black_time_ms
        self._active_player = active_player
        self._running_since = None
        self._has_started = has_started
        self._expired_player = (
            "white"
            if white_time_ms == 0
            else "black"
            if black_time_ms == 0
            else None
        )

    def reset(self, active_player: PlayerColor) -> OfflineClockSnapshot:
        self._white_time_ms = self._initial_time_ms
        self._black_time_ms = self._initial_time_ms
        self._active_player = active_player
        self._running_since = None
        self._has_started = False
        self._expired_player = None
        return self.snapshot()

    def start(self, active_player: PlayerColor) -> OfflineClockSnapshot:
        snapshot = self.snapshot()
        if snapshot.expired_player is not None:
            return snapshot
        self._active_player = active_player
        if self._running_since is None:
            self._running_since = self._clock()
        self._has_started = True
        return self.snapshot()

    def pause(self) -> OfflineClockSnapshot:
        snapshot = self.snapshot()
        self._white_time_ms = snapshot.white_time_ms
        self._black_time_ms = snapshot.black_time_ms
        self._running_since = None
        return OfflineClockSnapshot(
            snapshot.white_time_ms,
            snapshot.black_time_ms,
            snapshot.active_player,
            False,
            snapshot.has_started,
            snapshot.expired_player,
            snapshot.result,
        )

    def accept_move(
        self,
        next_player: PlayerColor,
    ) -> OfflineClockSnapshot:
        was_running = self._running_since is not None
        snapshot = self.snapshot()
        if snapshot.expired_player is not None:
            return snapshot
        self._white_time_ms = snapshot.white_time_ms
        self._black_time_ms = snapshot.black_time_ms
        if was_running:
            if self._active_player == "white":
                self._white_time_ms += self._increment_ms
            else:
                self._black_time_ms += self._increment_ms
        self._active_player = next_player
        should_run = was_running or not self._has_started
        self._running_since = self._clock() if should_run else None
        self._has_started = True
        return self.snapshot()

    def snapshot(self) -> OfflineClockSnapshot:
        if self._running_since is not None and self._expired_player is None:
            elapsed_ms = int(
                max(0.0, self._clock() - self._running_since) * 1000
            )
            if self._active_player == "white":
                remaining = max(0, self._white_time_ms - elapsed_ms)
                if remaining == 0:
                    self._white_time_ms = 0
                    self._running_since = None
                    self._expired_player = "white"
            else:
                remaining = max(0, self._black_time_ms - elapsed_ms)
                if remaining == 0:
                    self._black_time_ms = 0
                    self._running_since = None
                    self._expired_player = "black"
        white_time_ms = self._display_time("white")
        black_time_ms = self._display_time("black")
        expired_player = self._expired_player
        result: OfflineClockResult | None = (
            "0-1"
            if expired_player == "white"
            else "1-0"
            if expired_player == "black"
            else None
        )
        return OfflineClockSnapshot(
            white_time_ms,
            black_time_ms,
            self._active_player,
            self._running_since is not None,
            self._has_started,
            expired_player,
            result,
        )

    def _display_time(self, player: PlayerColor) -> int:
        stored = (
            self._white_time_ms
            if player == "white"
            else self._black_time_ms
        )
        if (
            self._running_since is None
            or self._expired_player is not None
            or player != self._active_player
        ):
            return stored
        elapsed_ms = int(
            max(0.0, self._clock() - self._running_since) * 1000
        )
        return max(0, stored - elapsed_ms)
