# uiauto_agent/selectors.py

PKG_SPOTIFY = "com.spotify.music"
PKG_FB = "com.facebook.katana"
PKG_TIKTOK = "com.zhiliaoapp.musically"
PKG_CAMERA = "com.google.android.GoogleCamera"  # Example for Pixel devices
PKG_TIKTOK = "com.zhiliaoapp.musically"
PKG_IG = "com.instagram.android"
PKG_WEATHER = "com.google.android.weather"


INSTAGRAM = {
    "search_tab": {"descriptionContains": "Search"},
    "search_edit": {"className": "android.widget.EditText"},
}


SPOTIFY = {
    "tab_search_text": {"text": "Search"},
    "search_field_id": {"resourceId": "com.spotify.music:id/find_search_field"},
    "play_desc": {"descriptionContains": "Play"},
}

FACEBOOK = {
    "home_tab": {"descriptionContains": "Home"},
    "search_tab": {"descriptionContains": "Search"},
    "search_edit": {"className": "android.widget.EditText"},
    "notifications": {"descriptionContains": "Notifications"},
    # Example resource-id to post something (you’ll likely need to inspect):
    # "create_post":   {"resourceId": "com.facebook.katana:id/(…)"},
}
