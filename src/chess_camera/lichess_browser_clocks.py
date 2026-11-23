import json
import os
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Literal


@dataclass(frozen=True)
class BrowserClocks:
    white: str
    black: str
    observed_at: float


class BrowserClockReader:
    def __init__(
        self,
        hwnd: int,
        game_id: str,
        player_color: Literal["white", "black"],
    ) -> None:
        self.game_id = game_id
        self._request = json.dumps(
            {"hwnd": hwnd, "game_id": game_id, "player_color": player_color}
        )
        self._player_color = player_color
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._process: subprocess.Popen[str] | None = None
        self._sample: BrowserClocks | None = None
        self._error = "Waiting for browser clocks."
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def snapshot(self) -> tuple[BrowserClocks | None, str]:
        with self._lock:
            sample = self._sample
            error = self._error
        if sample is not None and monotonic() - sample.observed_at <= 2.0:
            return sample, ""
        return None, error if sample is None else "Browser clocks are stale."

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            process = self._process
        if process is not None and process.poll() is None:
            process.terminate()

    def _run(self) -> None:
        if sys.platform != "win32":
            self._set_error("Browser clocks are supported on Windows only.")
            return
        try:
            script = Path(__file__).with_name("lichess_clock_reader.ps1").read_text(
                encoding="utf-8"
            )
            executable = (
                Path(os.environ["SystemRoot"])
                / "System32"
                / "WindowsPowerShell"
                / "v1.0"
                / "powershell.exe"
            )
            process = subprocess.Popen(
                [str(executable), "-NoProfile", "-NonInteractive", "-Command", script],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                encoding="utf-8",
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            with self._lock:
                self._process = process
            if self._stop.is_set():
                process.terminate()
                return
            if process.stdin is None or process.stdout is None:
                raise RuntimeError("Browser clock reader pipes are unavailable.")
            while not self._stop.is_set():
                started = monotonic()
                process.stdin.write(self._request + "\n")
                process.stdin.flush()
                line = process.stdout.readline()
                if not line:
                    raise RuntimeError("Browser clock reader stopped unexpectedly.")
                response = json.loads(line)
                if not isinstance(response, dict):
                    raise ValueError("Invalid browser clock response.")
                error = response.get("error")
                if isinstance(error, str):
                    self._set_error(error)
                else:
                    top = response.get("top")
                    bottom = response.get("bottom")
                    if not isinstance(top, str) or not isinstance(bottom, str):
                        raise ValueError("Invalid browser clock values.")
                    white, black = (
                        (bottom, top)
                        if self._player_color == "white"
                        else (top, bottom)
                    )
                    with self._lock:
                        self._sample = BrowserClocks(white, black, started)
                        self._error = ""
                self._stop.wait(0.25)
        except (OSError, KeyError, ValueError, RuntimeError) as error:
            if not self._stop.is_set():
                self._set_error(f"Browser clock reader failed: {error}")
        finally:
            with self._lock:
                process = self._process
                self._process = None
            if process is not None:
                if process.poll() is None:
                    process.terminate()
                process.communicate()

    def _set_error(self, message: str) -> None:
        with self._lock:
            self._sample = None
            self._error = message
