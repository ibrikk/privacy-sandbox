# uiauto_agent/actions_camera.py
from .device import start_app, robust_click, dismiss_overlays
from .selectors import PKG_CAMERA


def take_selfie(d):
    """
    Opens the camera app and takes a photo.
    This is a placeholder and selectors are highly device-dependent.
    """
    start_app(d, PKG_CAMERA)
    dismiss_overlays(d)

    # Placeholder selectors for taking a photo
    shutter_button_desc = {"descriptionContains": "Shutter"}
    shutter_button_id = {"resourceIdMatches": ".*shutter.*"}  # Regex match

    if not robust_click(d, **shutter_button_desc):
        robust_click(d, **shutter_button_id)

    print("✅ Took a photo/selfie")
