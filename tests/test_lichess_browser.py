import ctypes
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
from urllib.parse import unquote

import pytest
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QWidget

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from src.chess_camera import browser_window_windows
from src.chess_camera import lichess_browser


@pytest.fixture(scope="module")
def application() -> QApplication:
    instance = QApplication.instance()
    if isinstance(instance, QApplication):
        return instance
    return QApplication([])


def test_open_game_identifies_and_positions_new_window(
    application: QApplication,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "Local State").write_text(
        json.dumps(
            {
                "profile": {
                    "last_used": "Profile 1",
                    "info_cache": {"Profile 1": {}},
                }
            }
        ),
        encoding="utf-8",
    )
    executable = tmp_path / "chrome.exe"
    monkeypatch.setattr(
        lichess_browser, "_default_browser", lambda: (executable, tmp_path)
    )
    launch = MagicMock(return_value=(True, 123))
    monkeypatch.setattr(lichess_browser.QProcess, "startDetached", launch)
    place = MagicMock(return_value=12345)
    monkeypatch.setattr(lichess_browser, "place_identified_browser", place)
    owner = QWidget()
    completed = MagicMock()

    lichess_browser.open_lichess_game("abc12345", owner, completed)

    arguments = launch.call_args.args[1]
    assert launch.call_args.args[0] == str(executable)
    assert arguments[:2] == ["--new-window", "--profile-directory=Profile 1"]
    page = unquote(arguments[2])
    assert "https://lichess.org/abc12345" in page
    assert "abc12345secret" not in page
    QTest.qWait(150)
    marker = place.call_args.args[0]
    assert f"<title>{marker}</title>" in page
    assert place.call_args.args[1] == int(owner.winId())
    completed.assert_called_once_with(None, 12345)


@pytest.mark.parametrize("game_id", ["", "abc", "abc12345secret", "abc/2345"])
def test_invalid_game_ids_are_rejected(game_id: str) -> None:
    with pytest.raises(lichess_browser.LichessBrowserError, match="game ID"):
        lichess_browser.open_lichess_game(game_id, MagicMock(), MagicMock())


def test_missing_profile_and_failed_launch_are_reported(
    application: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    owner = QWidget()
    monkeypatch.setattr(
        lichess_browser, "_default_browser", lambda: (tmp_path / "chrome.exe", tmp_path)
    )
    with pytest.raises(lichess_browser.LichessBrowserError, match="profile"):
        lichess_browser.open_lichess_game("abc12345", owner, MagicMock())

    (tmp_path / "Local State").write_text(
        json.dumps({"profile": {"last_used": "Default", "info_cache": {"Default": {}}}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        lichess_browser.QProcess, "startDetached", MagicMock(return_value=(False, 0))
    )
    with pytest.raises(lichess_browser.LichessBrowserError, match="start"):
        lichess_browser.open_lichess_game("abc12345", owner, MagicMock())


@pytest.mark.parametrize("failure", ["timeout", "windows"])
def test_placement_failure_is_reported_without_relaunching(
    application: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    (tmp_path / "Local State").write_text(
        json.dumps({"profile": {"last_used": "Default", "info_cache": {"Default": {}}}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        lichess_browser, "_default_browser", lambda: (tmp_path / "chrome.exe", tmp_path)
    )
    launch = MagicMock(return_value=(True, 123))
    monkeypatch.setattr(lichess_browser.QProcess, "startDetached", launch)
    if failure == "windows":
        place = MagicMock(side_effect=lichess_browser.BrowserWindowError("OS error"))
    else:
        place = MagicMock(return_value=False)
        monkeypatch.setattr(
            lichess_browser, "time", SimpleNamespace(monotonic=MagicMock(side_effect=[0, 6]))
        )
    monkeypatch.setattr(lichess_browser, "place_identified_browser", place)
    owner = QWidget()
    completed = MagicMock()

    lichess_browser.open_lichess_game("abc12345", owner, completed)
    QTest.qWait(150)

    launch.assert_called_once()
    assert completed.call_count == 1
    assert completed.call_args.args[0]


def test_unsupported_platform_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lichess_browser.sys, "platform", "unsupported")
    with pytest.raises(lichess_browser.LichessBrowserError, match="Windows"):
        lichess_browser._default_browser()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows window API")
def test_only_identified_window_is_placed_on_owner_monitor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, tuple[int, int, int, int, int]] = {}
    api = SimpleNamespace(
        **{
            name: MagicMock()
            for name in (
                "EnumWindows", "GetWindowTextW", "GetClassNameW",
                "IsWindowVisible", "IsWindow", "MonitorFromWindow",
                "GetMonitorInfoW", "IsZoomed", "ShowWindow",
                "SetWindowPos", "GetWindowRect",
            )
        }
    )
    api.EnumWindows.side_effect = lambda callback, _: (
        callback(101, 0), callback(102, 0), True
    )[-1]
    api.IsWindowVisible.return_value = True
    api.IsWindow.return_value = True
    api.GetWindowTextW.side_effect = lambda hwnd, buffer, _: setattr(
        buffer, "value", "Chess-Lichess-marker - Google Chrome" if hwnd == 102 else "Other"
    )
    api.GetClassNameW.side_effect = lambda _hwnd, buffer, _: setattr(
        buffer, "value", "Chrome_WidgetWin_1"
    )
    api.MonitorFromWindow.return_value = 55

    def monitor_info(_monitor: int, pointer: ctypes.c_void_p) -> bool:
        rect = ctypes.cast(
            pointer, ctypes.POINTER(browser_window_windows._MonitorInfo)
        ).contents.rcWork
        rect.left, rect.top, rect.right, rect.bottom = (-1920, 40, 0, 1040)
        return True

    def set_position(
        hwnd: int, _after: object, x: int, y: int, w: int, h: int, _flags: int
    ) -> bool:
        calls["position"] = (hwnd, x, y, w, h)
        return True

    def window_rect(_hwnd: int, pointer: ctypes.c_void_p) -> bool:
        rect = ctypes.cast(
            pointer, ctypes.POINTER(browser_window_windows.wintypes.RECT)
        ).contents
        rect.left, rect.top, rect.right, rect.bottom = (-960, 40, 0, 1040)
        return True

    api.GetMonitorInfoW.side_effect = monitor_info
    api.IsZoomed.return_value = False
    api.SetWindowPos.side_effect = set_position
    api.GetWindowRect.side_effect = window_rect
    monkeypatch.setattr(browser_window_windows.ctypes, "WinDLL", lambda *args, **kwargs: api)

    assert browser_window_windows.place_identified_browser("Chess-Lichess-marker", 77)
    api.MonitorFromWindow.assert_called_once_with(77, 2)
    assert calls["position"] == (102, -960, 40, 960, 1000)


@pytest.mark.parametrize(
    ("browser", "message"),
    [("firefox.exe", "Chrome or Edge"), ("chrome.exe", "not found")],
)
def test_unsupported_or_missing_default_browser_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, browser: str, message: str
) -> None:
    registry = SimpleNamespace(
        HKEY_CURRENT_USER=object(),
        HKEY_CLASSES_ROOT=object(),
        OpenKey=MagicMock(),
        QueryValueEx=MagicMock(
            side_effect=[("Browser.ProgId", 1), (f'"{tmp_path / browser}" "%1"', 1)]
        ),
    )
    monkeypatch.setitem(sys.modules, "winreg", registry)
    monkeypatch.setattr(lichess_browser.sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))

    with pytest.raises(lichess_browser.LichessBrowserError, match=message):
        lichess_browser._default_browser()
