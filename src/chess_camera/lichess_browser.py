import json
import os
import re
import sys
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from urllib.parse import quote

from PySide6.QtCore import QProcess, QTimer
from PySide6.QtWidgets import QWidget

from .browser_window_windows import BrowserWindowError, place_identified_browser


class LichessBrowserError(RuntimeError):
    pass


def _default_browser() -> tuple[Path, Path]:
    if sys.platform != "win32":
        raise LichessBrowserError(
            "Automatic browser placement is supported on Windows with "
            "Google Chrome or Microsoft Edge."
        )
    import winreg

    try:
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\Shell\Associations\UrlAssociations\https\UserChoice",
        ) as key:
            prog_id = winreg.QueryValueEx(key, "ProgId")[0]
        if not isinstance(prog_id, str) or not prog_id:
            raise LichessBrowserError("The default browser is not configured.")
        with winreg.OpenKey(
            winreg.HKEY_CLASSES_ROOT,
            prog_id + r"\shell\open\command",
        ) as key:
            command = winreg.QueryValueEx(key, "")[0]
    except OSError as error:
        raise LichessBrowserError(
            "Cannot determine the Windows default browser."
        ) from error
    if not isinstance(command, str):
        raise LichessBrowserError("The default browser command is invalid.")
    match = re.match(r'\s*"([^"]+)"', command)
    if match is None:
        raise LichessBrowserError(
            "Cannot determine the default browser executable."
        )
    executable = Path(match.group(1))
    local_app_data = os.environ.get("LOCALAPPDATA")
    if not local_app_data:
        raise LichessBrowserError("The browser profile location is unavailable.")
    if executable.name.casefold() == "chrome.exe":
        user_data = Path(local_app_data) / "Google" / "Chrome" / "User Data"
    elif executable.name.casefold() == "msedge.exe":
        user_data = Path(local_app_data) / "Microsoft" / "Edge" / "User Data"
    else:
        raise LichessBrowserError(
            "Automatic placement requires Chrome or Edge as the default browser."
        )
    if not executable.is_file():
        raise LichessBrowserError("The default browser executable was not found.")
    return executable, user_data


def _last_used_profile(user_data: Path) -> str:
    try:
        with (user_data / "Local State").open(encoding="utf-8") as file:
            state = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise LichessBrowserError(
            "Cannot read the browser profile. Open the browser and select "
            "your profile, then retry."
        ) from error
    if not isinstance(state, dict):
        raise LichessBrowserError("The browser profile information is invalid.")
    profile = state.get("profile")
    if not isinstance(profile, dict):
        raise LichessBrowserError("The browser profile information is missing.")
    selected = profile.get("last_used")
    profiles = profile.get("info_cache")
    if (
        not isinstance(selected, str)
        or not selected
        or Path(selected).name != selected
        or not isinstance(profiles, dict)
        or selected not in profiles
    ):
        raise LichessBrowserError(
            "No browser profile is selected. Open the browser and select "
            "your profile, then retry."
        )
    return selected


def open_lichess_game(
    game_id: str,
    owner: QWidget,
    completed: Callable[[str | None], None],
) -> None:
    if re.fullmatch(r"[A-Za-z0-9]{8}", game_id) is None:
        raise LichessBrowserError("The selected Lichess game ID is invalid.")
    executable, user_data = _default_browser()
    profile = _last_used_profile(user_data)
    marker = f"Chess-Lichess-{uuid.uuid4().hex}"
    page = (
        f"<title>{marker}</title>"
        f'<meta http-equiv="refresh" content="6;url=https://lichess.org/{game_id}">'
        "<p>Opening your Lichess game...</p>"
    )
    arguments = [
        "--new-window",
        f"--profile-directory={profile}",
        "data:text/html," + quote(page),
    ]
    started, _ = QProcess.startDetached(str(executable), arguments)
    if not started:
        raise LichessBrowserError("Could not start the default browser.")
    timer = QTimer(owner)
    timer.setInterval(100)
    deadline = time.monotonic() + 5.0
    owner_window = int(owner.winId())

    def try_placement() -> None:
        try:
            placed = place_identified_browser(marker, owner_window)
        except BrowserWindowError as error:
            timer.stop()
            timer.deleteLater()
            completed(str(error))
            return
        if placed:
            timer.stop()
            timer.deleteLater()
            completed(None)
        elif time.monotonic() >= deadline:
            timer.stop()
            timer.deleteLater()
            completed("Could not identify the new browser window in time.")

    timer.timeout.connect(try_placement)
    timer.start()
