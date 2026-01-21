# # uiauto_agent/actions_instagram.py
# import random, time
# from .device import start_app, robust_click, dismiss_overlays, press, set_text
# from .selectors import PKG_IG, INSTAGRAM


# def browse_feed(d, persona, scrolls: int = 4):
#     start_app(d, PKG_IG)
#     dismiss_overlays(d)
#     time.sleep(2)
#     for _ in range(scrolls):
#         d.swipe_ext("up", scale=random.uniform(0.6, 0.9))
#         time.sleep(random.uniform(2, 4))
#         dismiss_overlays(d)


# def view_stories(d, persona, stories: int = 3):
#     start_app(d, PKG_IG)
#     dismiss_overlays(d)
#     # Tap top-left for stories
#     width, height = d.window_size()
#     d.click(width * 0.15, height * 0.15)
#     for _ in range(stories):
#         time.sleep(random.uniform(3, 5))
#         d.swipe_ext("left", scale=0.9)
#     press(d, "back")


# def search_interest(d, persona):
#     interest = persona.industry or "music"
#     start_app(d, PKG_IG)
#     dismiss_overlays(d)
#     robust_click(d, **INSTAGRAM["search_tab"])
#     set_text(d, interest, **INSTAGRAM["search_edit"])
#     d.press("enter")
#     time.sleep(3)
    
# def view_reels(d, persona, reels: int = 5):
#     """
#     Open Instagram Reels, watch and scroll through videos.
#     Mimics natural reels consumption.
#     """
#     start_app(d, PKG_IG)
#     dismiss_overlays(d)
#     time.sleep(2)

#     # 1️⃣ Try clicking Reels tab via selector
#     clicked = False
#     try:
#         robust_click(d, **INSTAGRAM["reels_tab"])
#         clicked = True
#     except Exception:
#         pass

#     # 2️⃣ Fallback: bottom navigation index (second tab)
#     if not clicked:
#         width, height = d.window_size()
#         # Bottom nav y-position
#         y = height * 0.94
#         # Second tab from left
#         x = width * 0.30
#         d.click(x, y)

#     time.sleep(2)

#     # 3️⃣ Watch & scroll reels
#     for _ in range(reels):
#         watch_time = random.uniform(4.5, 8.0)
#         time.sleep(watch_time)

#         # Scroll to next reel
#         d.swipe_ext("up", scale=random.uniform(0.7, 0.9))
#         time.sleep(random.uniform(1.0, 2.0))

#         dismiss_overlays(d)

#     # Optional exit
#     press(d, "back")
