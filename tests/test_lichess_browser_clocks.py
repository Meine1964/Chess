import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Literal

import pytest

from src.chess_camera.lichess_browser_clocks import BrowserClockReader


@pytest.mark.skipif(sys.platform != "win32", reason="Windows PowerShell reader")
@pytest.mark.parametrize(
    ("scenario", "expected"),
    [
        ("normal", {"top": "05:00", "bottom": "00:09.2"}),
        ("hours", {"top": "1:23:45", "bottom": "00:09.2"}),
        ("extra", "Could not uniquely identify both browser clocks."),
        ("misaligned", "Browser clock positions are ambiguous."),
    ],
)
def test_flattened_browser_text_is_parsed_or_rejected_if_ambiguous(
    scenario: str, expected: dict[str, str] | str
) -> None:
    def clock(y: int, names: list[str], x: int = 100) -> list[dict[str, object]]:
        return [
            {"x": x + index * 30, "y": y, "width": 30, "height": 50, "name": name}
            for index, name in enumerate(names)
        ]

    top = ["1", ":", "23", ":", "45"] if scenario == "hours" else ["05", ":", "00"]
    pieces = clock(100, top) + clock(
        500, ["00", ":", "09", ".", "2"], 300 if scenario == "misaligned" else 100
    )
    pieces += [pieces[0]]
    if scenario == "extra":
        pieces += clock(700, ["01", ":", "00"])
    script = Path("src").joinpath(
        "chess_camera", "lichess_clock_reader.ps1"
    ).read_text(encoding="utf-8")
    script += (
        "\n$pieces = ConvertFrom-Json '" + json.dumps(pieces) + "'\n"
        "try { $result = Find-Clocks @($pieces) } "
        "catch { $result = @{ error = $_.Exception.Message } }\n"
        "[Console]::WriteLine(($result | ConvertTo-Json -Compress))\n"
    )
    powershell = (
        Path(os.environ["SystemRoot"]) / "System32" / "WindowsPowerShell"
        / "v1.0" / "powershell.exe"
    )
    result = subprocess.run(
        [str(powershell), "-NoProfile", "-NonInteractive", "-Command", script],
        input="",
        capture_output=True,
        encoding="utf-8",
        check=True,
        timeout=10,
    )
    parsed = json.loads(result.stdout)
    if isinstance(expected, str):
        assert parsed["error"] == expected
    else:
        assert parsed == expected


@pytest.mark.skipif(sys.platform != "win32", reason="Windows PowerShell reader")
@pytest.mark.parametrize(
    ("color", "white", "black"),
    [("white", "04:59", "05:00"), ("black", "05:00", "04:59")],
)
def test_reader_maps_opponent_clock_and_rejects_stale_samples(
    monkeypatch: pytest.MonkeyPatch,
    color: Literal["white", "black"],
    white: str,
    black: str,
) -> None:
    class Process:
        stdin = io.StringIO()
        stdout = io.StringIO('{"top":"05:00","bottom":"04:59"}\n')
        returncode = None

        def poll(self) -> int | None:
            return self.returncode

        def terminate(self) -> None:
            self.returncode = 0

        def communicate(self) -> tuple[str, str]:
            return "", ""

    process = Process()
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: process)
    reader = BrowserClockReader(10, "abc12345", color)
    deadline = time.monotonic() + 2
    try:
        while time.monotonic() < deadline:
            sample, _ = reader.snapshot()
            if sample is not None:
                break
            time.sleep(0.01)
        assert sample is not None
        assert (sample.white, sample.black) == (white, black)
        monkeypatch.setattr(
            "src.chess_camera.lichess_browser_clocks.monotonic",
            lambda: sample.observed_at + 2.1,
        )
        assert reader.snapshot() == (None, "Browser clocks are stale.")
        request = json.loads(process.stdin.getvalue().splitlines()[0])
        assert request == {
            "hwnd": 10, "game_id": "abc12345", "player_color": color
        }
    finally:
        reader.stop()
        reader._thread.join(timeout=2)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows PowerShell reader")
def test_reader_reports_missing_browser_without_installation() -> None:
    reader = BrowserClockReader(0, "abc12345", "white")
    deadline = time.monotonic() + 5
    try:
        while time.monotonic() < deadline:
            sample, reason = reader.snapshot()
            if reason != "Waiting for browser clocks.":
                break
            time.sleep(0.02)
        assert sample is None
        assert "hwnd" in reason.lower()
    finally:
        reader.stop()
        reader._thread.join(timeout=2)
    assert not reader._thread.is_alive()
