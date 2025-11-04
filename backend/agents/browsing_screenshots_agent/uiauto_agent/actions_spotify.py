# uiauto_agent/actions_spotify.py
from .device import start_app, robust_click, set_text, dismiss_overlays
from .selectors import PKG_SPOTIFY, SPOTIFY

# Simple mapping from persona traits → query
def _playlist_query(persona) -> str:
    act = (persona.activity_description or "").lower()
    city = (persona.city or "").lower()
    age  = str(persona.age)

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

    if not robust_click(d, **SPOTIFY["tab_search_text"]):
        # fallback: sometimes search is only a magnifier
        robust_click(d, description="Search")

    query = _playlist_query(persona)
    if not set_text(d, query, **SPOTIFY["search_field_id"]):
        # try generic edit
        set_text(d, query, className="android.widget.EditText")

    # tap first relevant result
    # (best to add a stronger selector after inspecting your UI dump)
    robust_click(d, textContains=query.split()[0].capitalize())

    # press play (if needed)
    robust_click(d, **SPOTIFY["play_desc"]) or robust_click(d, text="Play")