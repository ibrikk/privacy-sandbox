# uiauto_agent/actions_spotify.py
import time
from .device import start_app, robust_click, set_text, dismiss_overlays
from .selectors import PKG_SPOTIFY, SPOTIFY


# Simple mapping from persona traits → query
def _playlist_query(persona) -> str:
    act = (persona.activity_description or "").lower()
    city = (persona.city or "").lower()
    age = persona.age

    if "commut" in act:
        return "Spotify Commute Mix"
    if "driv" in act:
        return "Driving Focus"
    if "run" in act or "jog" in act:
        return "Running playlist"
    if "study" in act or "work" in act:
        return "Lo-fi beats"
    if city in {"san francisco", "new york", "seattle"}:
        return f"{city.title()} vibes"
    if age.isdigit() and int(age) < 25:
        return "Top 50 Global"
    return "Chill mix"


def play_for_persona(d, persona):
    start_app(d, PKG_SPOTIFY)
    dismiss_overlays(d)
    time.sleep(2)

    # 1. Go to search tab
    if not robust_click(d, **SPOTIFY["tab_search_text"]):
        # fallback: sometimes search is only a magnifier
        robust_click(d, description="Search")
    
    time.sleep(1)
    
    # 2. Click the search bar to activate it
    robust_click(d, text="Search") # Or another selector for the search bar itself

    query = _playlist_query(persona)
    
    # 3. Type the search query
    if not set_text(d, query, **SPOTIFY["search_field_id"]):
        # try generic edit
        set_text(d, query, className="android.widget.EditText")
    
    time.sleep(2) # Wait for results to load

    # 4. Tap first relevant result
    # This is a bit fragile, a better selector would be ideal
    robust_click(d, resourceIdMatches=".*row_view_text_title.*", instance=0)

    time.sleep(1)

    # 5. Press play (if needed)
    robust_click(d, **SPOTIFY["play_desc"]) or robust_click(d, text="Play")
    
    time.sleep(5) # Play for a bit

