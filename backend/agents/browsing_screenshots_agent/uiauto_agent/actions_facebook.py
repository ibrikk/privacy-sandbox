# uiauto_agent/actions_facebook.py
import random
import time
from .device import start_app, robust_click, set_text, dismiss_overlays, press
from .selectors import PKG_FB, FACEBOOK


def _status_for_persona(persona) -> str:
    act = (persona.activity_description or "browsing").lower()
    city = persona.city or ""
    mood = "😊"
    msg = f"Morning {act} in {city} {mood}".strip()
    return msg


def open_and_browse(d, scrolls: int = 3):
    start_app(d, PKG_FB)
    dismiss_overlays(d)
    robust_click(d, **FACEBOOK["home_tab"])
    time.sleep(2)  # Wait for feed to load
    for _ in range(scrolls):
        d.swipe_ext("up", scale=random.uniform(0.6, 0.9))
        time.sleep(random.uniform(1, 3))


def search_topic(d, topic: str):
    # Go to search and query
    if not robust_click(d, **FACEBOOK["search_tab"]):
        robust_click(d, text="Search")
    set_text(d, topic, **FACEBOOK["search_edit"])
    d.press("enter")


def maybe_post_status(d, persona):
    """
    Post only if a 'What's on your mind?' box is visible (already logged in).
    Otherwise silently skip.
    """
    start_app(d, PKG_FB)
    dismiss_overlays(d)
    time.sleep(2)

    box_variants = [
        {"textContains": "What’s on your mind"},
        {"textContains": "What's on your mind"},
    ]
    can_post = False
    for sel in box_variants:
        if robust_click(d, **sel):
            can_post = True
            break

    if not can_post:
        print("Could not find 'What's on your mind?' box. Skipping post.")
        return

    time.sleep(random.uniform(1, 2))
    content = _status_for_persona(persona)
    # Compose text
    set_text(d, content, className="android.widget.EditText")
    time.sleep(random.uniform(1, 2))
    # Try to send/post
    if not robust_click(d, text="Post"):
        robust_click(d, descriptionContains="Post")
    
    time.sleep(2) # Wait for post to upload
    # go back to timeline
    press(d, "back")
