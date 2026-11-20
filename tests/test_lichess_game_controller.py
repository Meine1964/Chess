import pytest

from src.chess_camera.lichess_game_controller import LichessGameController


def _controller() -> LichessGameController:
    return LichessGameController(
        "game1234",
        "startpos",
        "white",
        "Stockfish level 1",
    )


def test_controller_queues_and_confirms_remote_ordinary_move() -> None:
    controller = _controller()

    attached = controller.accept_server_history(())
    pending = controller.accept_server_history(("e2e4",))

    assert attached.state == "synchronized"
    assert pending.state == "guidance"
    assert pending.pending_moves == ("e2e4",)
    assert controller.server_position[(4, 4)] == "P"
    assert controller.physical_position[(6, 4)] == "P"
    assert controller.expected_signature is not None
    assert controller.expected_signature.uci == "e2e4"

    confirmed = controller.confirm_expected_move(((6, 4), (4, 4)))

    assert confirmed.state == "synchronized"
    assert confirmed.pending_moves == ()
    assert controller.physical_position == controller.server_position


def test_controller_queues_multiple_moves_in_server_order() -> None:
    controller = _controller()
    controller.accept_server_history(())
    controller.accept_server_history(("e2e4",))
    controller.confirm_expected_move(((6, 4), (4, 4)))

    update = controller.accept_server_history(
        ("e2e4", "e7e5", "g1f3")
    )

    assert update.pending_moves == ("e7e5", "g1f3")
    assert controller.expected_signature is not None
    assert controller.expected_signature.uci == "e7e5"

    controller.confirm_expected_move(((1, 4), (3, 4)))

    assert controller.expected_signature is not None
    assert controller.expected_signature.uci == "g1f3"


def test_controller_prepares_and_commits_local_ordinary_move() -> None:
    controller = _controller()
    controller.accept_server_history(())

    uci, preview = controller.prepare_local_move(((6, 4), (4, 4)))

    assert uci == "e2e4"
    assert preview.position[(4, 4)] == "P"
    committed = controller.commit_local_move(((6, 4), (4, 4)))
    assert committed.fen == preview.fen
    assert controller.server_moves == ("e2e4",)
    assert controller.server_fen == preview.fen
    assert controller.physical_fen == preview.fen
    assert controller.state == "synchronized"


def test_controller_does_not_prepare_opponent_or_special_local_move() -> None:
    controller = _controller()
    controller.accept_server_history(())
    controller.commit_local_move(((6, 4), (4, 4)))

    assert controller.local_move_signatures == ()
    with pytest.raises(ValueError, match="not a supported"):
        controller.prepare_local_move(((1, 4), (3, 4)))


def test_controller_rejects_wrong_physical_move_without_advancing() -> None:
    controller = _controller()
    controller.accept_server_history(())
    controller.accept_server_history(("e2e4",))

    rejected = controller.confirm_expected_move(((6, 3), (4, 3)))

    assert rejected.state == "correction"
    assert rejected.pending_moves == ("e2e4",)
    assert controller.physical_position[(6, 4)] == "P"

    restored = controller.note_correction_restored()

    assert restored.state == "guidance"
    assert restored.pending_moves == ("e2e4",)


def test_controller_requires_reconciliation_for_rewritten_history() -> None:
    controller = _controller()
    controller.accept_server_history(())
    controller.accept_server_history(("e2e4",))

    update = controller.accept_server_history(("d2d4",))

    assert update.state == "reconciliation-required"
    assert controller.pending_moves == ("e2e4",)


def test_controller_does_not_replay_long_existing_game_on_attach() -> None:
    controller = _controller()

    update = controller.accept_server_history(
        ("e2e4", "e7e5", "g1f3")
    )

    assert update.state == "reconciliation-required"
    assert update.pending_moves == ()


def test_controller_stages_capture_support() -> None:
    controller = _controller()
    controller.accept_server_history(())
    controller.accept_server_history(("e2e4", "d7d5"))
    controller.confirm_expected_move(((6, 4), (4, 4)))
    controller.confirm_expected_move(((1, 3), (3, 3)))

    update = controller.accept_server_history(
        ("e2e4", "d7d5", "e4d5")
    )

    assert update.state == "reconciliation-required"
    assert update.pending_moves == ("e4d5",)
