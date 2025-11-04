# uiauto_agent/actions_tiktok.py
from .device import start_app, robust_click, dismiss_overlays
from .selectors import PKG_TIKTOK

def watch_and_scroll(d):
    """
    Opens TikTok, dismisses any overlays, and scrolls the feed.
    This is a placeholder for more complex behavior.
    """
    start_app(d, PKG_TIKTOK)
    dismiss_overlays(d)
    
    # Placeholder: Just scroll down a few times
    for _ in range(5):
        d.swipe_ext("up", scale=0.8)
        d.sleep(1.5) # Simulate watching
    
    print("✅ Scrolled TikTok feed")
