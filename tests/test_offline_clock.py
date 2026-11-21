from src.chess_camera.offline_clock import OfflineClock


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_offline_clock_switches_and_adds_increment() -> None:
    clock = FakeClock()
    game_clock = OfflineClock(60_000, 2_000, clock=clock)

    game_clock.start("white")
    clock.now = 5.0
    snapshot = game_clock.accept_move("black")

    assert snapshot.white_time_ms == 57_000
    assert snapshot.black_time_ms == 60_000
    assert snapshot.active_player == "black"
    assert snapshot.running is True

    clock.now = 8.5
    snapshot = game_clock.pause()

    assert snapshot.white_time_ms == 57_000
    assert snapshot.black_time_ms == 56_500
    assert snapshot.running is False


def test_first_accepted_move_starts_next_players_clock() -> None:
    game_clock = OfflineClock(60_000, 2_000)

    snapshot = game_clock.accept_move("black")

    assert snapshot.white_time_ms == 60_000
    assert snapshot.black_time_ms == 60_000
    assert snapshot.active_player == "black"
    assert snapshot.running is True
    assert snapshot.has_started is True


def test_move_after_pause_switches_but_remains_paused() -> None:
    game_clock = OfflineClock(60_000, 2_000)
    game_clock.start("white")
    game_clock.pause()

    snapshot = game_clock.accept_move("black")

    assert snapshot.white_time_ms == 60_000
    assert snapshot.black_time_ms == 60_000
    assert snapshot.active_player == "black"
    assert snapshot.running is False


def test_offline_clock_timeout_stops_and_sets_result() -> None:
    clock = FakeClock()
    game_clock = OfflineClock(1_000, 0, clock=clock)

    game_clock.start("white")
    clock.now = 1.2
    snapshot = game_clock.snapshot()

    assert snapshot.white_time_ms == 0
    assert snapshot.black_time_ms == 1_000
    assert snapshot.expired_player == "white"
    assert snapshot.result == "0-1"
    assert snapshot.running is False


def test_offline_clock_restores_paused() -> None:
    game_clock = OfflineClock(60_000, 0)

    game_clock.restore(42_000, 31_000, "black", has_started=True)
    snapshot = game_clock.snapshot()

    assert snapshot.white_time_ms == 42_000
    assert snapshot.black_time_ms == 31_000
    assert snapshot.active_player == "black"
    assert snapshot.running is False
    assert snapshot.has_started is True
