# # uiauto_agent/agent.py
# from .graph_agent import run_persona_session

# __all__ = ["run_persona_session"]


# # uiauto_agent/actions_tiktok.py
# import random, time
# from .device import start_app, robust_click, dismiss_overlays, press
# from .selectors import PKG_TIKTOK


# def watch_and_scroll(d, persona, scrolls: int = 5):
#     """
#     Open TikTok and scroll through a few videos based on persona.
#     If persona.online_behavior mentions social/browsing or young age,
#     this simulates a realistic short session.
#     """
#     start_app(d, PKG_TIKTOK)
#     dismiss_overlays(d)
#     time.sleep(3)

#     for i in range(scrolls):
#         d.swipe_ext("up", scale=random.uniform(0.6, 0.9))
#         time.sleep(random.uniform(2, 5))

#         # Occasionally "like" a video
#         if random.random() < 0.3:
#             width, height = d.window_size()
#             x = int(width * 0.85)
#             y = int(height * 0.45)
#             d.click(x, y)
#         dismiss_overlays(d)
#     press(d, "back")
