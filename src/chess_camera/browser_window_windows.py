import ctypes
from ctypes import wintypes


class BrowserWindowError(RuntimeError):
    pass


class _MonitorInfo(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("rcMonitor", wintypes.RECT),
        ("rcWork", wintypes.RECT),
        ("dwFlags", wintypes.DWORD),
    ]


def place_identified_browser(marker: str, owner_window: int) -> bool:
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    user32.EnumWindows.argtypes = [callback_type, wintypes.LPARAM]
    user32.EnumWindows.restype = wintypes.BOOL
    user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.GetWindowTextW.restype = ctypes.c_int
    user32.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.GetClassNameW.restype = ctypes.c_int
    user32.IsWindowVisible.argtypes = [wintypes.HWND]
    user32.IsWindowVisible.restype = wintypes.BOOL
    user32.IsWindow.argtypes = [wintypes.HWND]
    user32.IsWindow.restype = wintypes.BOOL
    user32.MonitorFromWindow.argtypes = [wintypes.HWND, wintypes.DWORD]
    user32.MonitorFromWindow.restype = wintypes.HANDLE
    user32.GetMonitorInfoW.argtypes = [wintypes.HANDLE, ctypes.POINTER(_MonitorInfo)]
    user32.GetMonitorInfoW.restype = wintypes.BOOL
    user32.IsZoomed.argtypes = [wintypes.HWND]
    user32.IsZoomed.restype = wintypes.BOOL
    user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
    user32.ShowWindow.restype = wintypes.BOOL
    user32.SetWindowPos.argtypes = [
        wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, wintypes.UINT,
    ]
    user32.SetWindowPos.restype = wintypes.BOOL
    user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    user32.GetWindowRect.restype = wintypes.BOOL

    matches: list[int] = []

    @callback_type
    def collect(hwnd: int, _parameter: int) -> bool:
        if not user32.IsWindowVisible(hwnd):
            return True
        title = ctypes.create_unicode_buffer(512)
        name = ctypes.create_unicode_buffer(128)
        user32.GetWindowTextW(hwnd, title, len(title))
        user32.GetClassNameW(hwnd, name, len(name))
        if title.value.startswith(marker) and name.value == "Chrome_WidgetWin_1":
            matches.append(hwnd)
        return True

    if not user32.EnumWindows(collect, 0):
        raise BrowserWindowError("Could not enumerate browser windows.")
    if not matches:
        return False
    if len(matches) != 1:
        raise BrowserWindowError("Could not uniquely identify the new browser window.")
    if not user32.IsWindow(owner_window):
        raise BrowserWindowError("The application's window is no longer available.")
    monitor = user32.MonitorFromWindow(owner_window, 2)
    info = _MonitorInfo()
    info.cbSize = ctypes.sizeof(info)
    if not monitor or not user32.GetMonitorInfoW(monitor, ctypes.byref(info)):
        raise BrowserWindowError("Could not determine the application's screen.")
    work = info.rcWork
    width = work.right - work.left
    height = work.bottom - work.top
    x = work.left + width // 2
    target_width = width - width // 2
    hwnd = matches[0]
    if user32.IsZoomed(hwnd):
        user32.ShowWindow(hwnd, 9)
    if not user32.SetWindowPos(hwnd, None, x, work.top, target_width, height, 0x0014):
        raise BrowserWindowError("Windows could not position the browser window.")
    actual = wintypes.RECT()
    if not user32.GetWindowRect(hwnd, ctypes.byref(actual)):
        raise BrowserWindowError("Could not verify the browser window position.")
    if (
        actual.left != x
        or actual.top != work.top
        or actual.right != work.right
        or actual.bottom != work.bottom
    ):
        raise BrowserWindowError("Windows did not apply the requested browser position.")
    return True
