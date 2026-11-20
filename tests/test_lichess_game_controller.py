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


def test_controller_queues_and_confirms_remote_capture() -> None:
    controller = _controller()
    controller.accept_server_history(())
    controller.accept_server_history(("e2e4", "d7d5"))
    controller.confirm_expected_move(((6, 4), (4, 4)))
    controller.confirm_expected_move(((1, 3), (3, 3)))

    update = controller.accept_server_history(
        ("e2e4", "d7d5", "e4d5")
    )

    assert update.state == "guidance"
    assert update.pending_moves == ("e4d5",)
    assert controller.expected_signature is not None
    assert controller.expected_signature.kind == "capture"

    confirmed = controller.confirm_expected_move(((4, 4), (3, 3)))

    assert confirmed.state == "synchronized"
    assert controller.physical_position == controller.server_position


def test_controller_prepares_and_commits_local_capture() -> None:
    controller = _controller()
    controller.accept_server_history(())
    controller.commit_local_move(((6, 4), (4, 4)))
    controller.accept_server_history(("e2e4", "d7d5"))
    controller.confirm_expected_move(((1, 3), (3, 3)))

    signature = next(
        signature
        for signature in controller.local_move_signatures
        if signature.uci == "e4d5"
    )
    uci, preview = controller.prepare_local_move(
        (signature.source, signature.destination)
    )

    assert signature.kind == "capture"
    assert uci == "e4d5"
    assert preview.position[(3, 3)] == "P"
    committed = controller.commit_local_move(
        (signature.source, signature.destination)
    )
    assert committed.fen == preview.fen
    assert controller.server_moves == ("e2e4", "d7d5", "e4d5")
    assert controller.physical_fen == controller.server_fen


def test_controller_keeps_promotion_in_reconciliation() -> None:
    controller = LichessGameController(
        "game1234",
        "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
        "white",
        "Stockfish level 1",
    )
    controller.accept_server_history(())

    update = controller.accept_server_history(("a7a8q",))

    assert update.state == "reconciliation-required"
    assert controller.expected_signature is None


def test_controller_queues_and_confirms_incoming_castling() -> None:
    initial_fen = "4k3/8/8/8/8/8/8/4K2R w K - 0 1"
    controller = LichessGameController(
        "game1234",
        initial_fen,
        "black",
        "Remote",
    )

    update = controller.accept_server_history(("e1g1",))

    assert update.state == "guidance"
    signature = controller.expected_signature
    assert signature is not None
    assert signature.kind == "castling"
    assert controller.confirm_expected_move(
        (signature.source, signature.destination)
    ).state == "synchronized"
    assert controller.physical_fen == controller.server_fen


def test_controller_prepares_and_commits_local_castling() -> None:
    initial_fen = "4k3/8/8/8/8/8/8/4K2R w K - 0 1"
    controller = LichessGameController(
        "game1234",
        initial_fen,
        "white",
        "Remote",
    )
    controller.accept_server_history(())

    uci, preview = controller.prepare_local_move(((7, 4), (7, 6)))

    assert uci == "e1g1"
    assert preview.position[(7, 6)] == "K"
    assert preview.position[(7, 5)] == "R"
    committed = controller.commit_local_move(((7, 4), (7, 6)))
    assert committed.fen == preview.fen
    assert controller.server_moves == ("e1g1",)
    assert controller.physical_fen == controller.server_fen
