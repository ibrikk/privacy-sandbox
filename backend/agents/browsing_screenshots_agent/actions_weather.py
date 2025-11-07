# uiauto_agent/actions_weather.py
import time
from .device import start_app, robust_click, set_text, dismiss_overlays, press
from .selectors import PKG_WEATHER


def check_weather(d, persona):
    start_app(d, PKG_WEATHER)
    dismiss_overlays(d)

    city = persona.city or "San Francisco"
    set_text(d, city, className="android.widget.EditText")
    d.press("enter")
    time.sleep(3)
    # optional: screenshot
    path = f"weather_{city.replace(' ', '_')}.png"
    d.screenshot(path)
    print(f"📸 Weather captured for {city} -> {path}")
    press(d, "back")
