import json
from pathlib import Path

import numpy as np
import pytest

from src.chess_camera.diagnostics import RuntimeDiagnosticLog


def test_runtime_diagnostics_write_current_state_and_event_history(
    tmp_path: Path,
) -> None:
    diagnostics = RuntimeDiagnosticLog(tmp_path)

    diagnostics.record(
        "lichess-snapshot",
        "Game state updated.",
        {
            "game_id": "game1234",
            "pending_moves": ["e2e4"],
            "physical_position": {"e2": "P"},
        },
        details={"transport": "polling"},
        images={
            "camera": np.zeros((20, 30, 3), dtype=np.uint8),
            "accepted-board": np.zeros((16, 16, 3), dtype=np.uint8),
        },
    )

    current = json.loads(
        diagnostics.state_path.read_text(encoding="utf-8")
    )
    events = diagnostics.events_path.read_text(
        encoding="utf-8"
    ).splitlines()
    assert current["event"] == {
        "details": {"transport": "polling"},
        "message": "Game state updated.",
        "type": "lichess-snapshot",
    }
    assert current["state"]["pending_moves"] == ["e2e4"]
    assert len(events) == 1
    assert json.loads(events[0])["state"]["game_id"] == "game1234"
    assert (tmp_path / "runtime-camera.png").is_file()
    assert (tmp_path / "runtime-accepted-board.png").is_file()


def test_runtime_diagnostics_reject_sensitive_keys(
    tmp_path: Path,
) -> None:
    diagnostics = RuntimeDiagnosticLog(tmp_path)

    with pytest.raises(ValueError, match="Sensitive"):
        diagnostics.record(
            "connection",
            "Unsafe state.",
            {"authorization_header": "must-not-be-written"},
        )

    with pytest.raises(ValueError, match="Sensitive"):
        diagnostics.record(
            "connection",
            "Unsafe details.",
            {},
            details={"access_token": "must-not-be-written"},
        )

    assert not diagnostics.state_path.exists()
    assert not diagnostics.events_path.exists()
