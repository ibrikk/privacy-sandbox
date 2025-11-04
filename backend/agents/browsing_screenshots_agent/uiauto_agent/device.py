# uiauto_agent/device.py
import time
import uiautomator2 as u2

COMMON_DISMISS_SELECTORS = [
    {"text": "Skip"},
    {"text": "SKIP"},
    {"text": "Close"},
    {"text": "CLOSE"},
    {"text": "Not now"},
    {"text": "No thanks"},
    {"text": "OK"},
    {"text": "Allow"},
    {"text": "Deny"},
    {"description": "Close"},
    {"description": "Dismiss"},
    {"textContains": "Continue"},
    {"textContains": "Agree"},
    {"text": "×"},
    {"description": "×"},
]


import uiautomator2 as u2
import threading, time


def connect(
    serial: str | None = None,
    implicit_wait: float = 6.0,
    heartbeat_sec: int | None = 120,
):
    d = u2.connect(serial) if serial else u2.connect()

    # Ensure the UIA service is up
    try:
        d.healthcheck()  # starts uiautomator if needed
    except Exception:
        pass

    # Fast input IME (auto-installs if missing on recent u2)
    try:
        d.set_fastinput_ime(True)
    except Exception:
        pass  # not fatal on some ROMs

    # Global implicit wait for selectors
    try:
        d.implicitly_wait(implicit_wait)
    except Exception:
        pass
    # Older u2 versions expose wait_timeout as a property
    if hasattr(d, "wait_timeout"):
        try:
            d.wait_timeout = implicit_wait
        except Exception:
            pass

    # Optional heartbeat to prevent idle disconnects (u2 has no newCommandTimeout)
    if heartbeat_sec and heartbeat_sec > 0:

        def _hb(dev):
            while True:
                try:
                    _ = dev.info  # cheap ping
                except Exception:
                    pass
                time.sleep(heartbeat_sec)

        threading.Thread(target=_hb, args=(d,), daemon=True).start()

    return d


def exists(d, **sel) -> bool:
    try:
        return d(**sel).exists
    except Exception:
        return False


def wait_and_click(d, timeout=10, **sel) -> bool:
    obj = d(**sel)
    if obj.wait(timeout=timeout):
        obj.click()
        return True
    return False


def set_text(d, text: str, timeout=10, **sel) -> bool:
    obj = d(**sel)
    if obj.wait(timeout=timeout):
        obj.set_text(text)
        return True
    return False


def dismiss_overlays(d) -> bool:
    hit = False
    for sel in COMMON_DISMISS_SELECTORS:
        if exists(d, **sel):
            try:
                d(**sel).click()
                hit = True
            except Exception:
                pass
    return hit


def robust_click(d, tries=3, **sel) -> bool:
    for _ in range(tries):
        if wait_and_click(d, **sel):
            return True
        if dismiss_overlays(d):
            time.sleep(0.4)
    return False


def press(d, key: str = "back"):
    # "back" | "home" | "recent"
    getattr(d, key)()


def start_app(d, pkg: str):
    d.app_start(pkg)
    d.wait_activity(timeout=10)


def stop_app(d, pkg: str):
    d.app_stop(pkg)


def get_installed_apps(d) -> list[str]:
    """Returns a list of installed third-party packages."""
    return d.app_list(third_only=True)
