import asyncio
import random
from typing import Dict, Any


# ============================================================
# Dispatcher
# ============================================================

async def execute_action(mcp, action: Dict[str, Any]):
    app = action["app"]
    act = action["action"]
    args = action.get("args", {})

    key = f"{app}.{act}"
    print(f"▶️ EXECUTOR: {key}")

    EXECUTORS = {
        "com.spotify.music.play_for_persona": spotify_play_for_persona,

        "com.facebook.katana.open_and_browse": facebook_open_and_browse,
        "com.facebook.katana.search_topic": facebook_search_topic,
        "com.facebook.katana.maybe_post_status": facebook_maybe_post_status,

        "com.instagram.android.view_stories": instagram_view_stories,
        "com.instagram.android.view_reels": instagram_view_reels,
        "com.instagram.android.browse_feed": instagram_browse_feed,

        "com.zhiliaoapp.musically.watch_and_scroll": tiktok_watch_and_scroll,

        "com.linkedin.android.browse_feed": linkedin_open_and_browse,

        "com.google.android.youtube.watch_recommended": youtube_watch_recommended,
    }

    fn = EXECUTORS.get(key)
    if not fn:
        print(f"⚠️ No executor mapped for {key}")
        return

    await fn(mcp, **args)


# ============================================================
# Helpers
# ============================================================

async def human_pause(min_s=0.6, max_s=1.8):
    await asyncio.sleep(random.uniform(min_s, max_s))


async def vertical_scroll(mcp):

    await mcp.call_tool(
       "swipe",
        {
            "start_x": 100,
            "start_y": 2200,
            "end_x": 100,
            "end_y": 1,
        }
    )

async def swipe_down(mcp):
    await mcp.call_tool("swipe", 
            {
                "start_x": 100,
                "start_y": 700,
                "end_x": 100,
                "end_y": 1500,
            })

async def ensure_device_ready(mcp):
    await mcp.call_tool("screen_on", {})
    await mcp.call_tool("unlock_screen", {})
    await asyncio.sleep(1)

# ============================================================
# Spotify
# ============================================================

async def spotify_play_for_persona(mcp, **_):
    await ensure_device_ready(mcp)
    await mcp.call_tool("start_app", {"package_name": "com.spotify.music"})
    await asyncio.sleep(3)

    # Tap first recommendation
    PLAY_SELECTORS = [
                ("com.spotify.music:id/image", "resourceId"),
                ("com.spotify.music:id/button_play_and_pause", "resourceId"),
                # ("microcore", "text"),
            ]
    for selector, selector_type in PLAY_SELECTORS:
        await mcp.call_tool("click",
            {
                "selector": selector,
                "selector_type": selector_type,
            },)
        await human_pause(2, 4)

    # Let music play for a bit
    for _ in range(random.randint(2, 4)):
        await human_pause(3, 6)

        # Occasionally skip
        if random.random() < 0.50:
            await mcp.call_tool("click", 
                                {"selector": "com.spotify.music:id/now_playing_bar_layout",
                                 "selector_type": "resourceId"
                                })
            await mcp.call_tool("click", 
                                {"selector": "Next",
                                "selector_type": "description"})
            # swipe down
            await swipe_down(mcp)
            await human_pause(3, 6)


# ============================================================
# Facebook
# ============================================================

async def facebook_open_and_browse(mcp, **_):
    await ensure_device_ready(mcp)
    await mcp.call_tool("start_app", {"package_name": "com.facebook.katana"})
    await asyncio.sleep(3)

    scrolls = random.randint(3, 6)
    for _ in range(scrolls):
        await vertical_scroll(mcp)
        await human_pause()

        # Occasionally pause longer on a post
        if random.random() < 0.3:
            await human_pause(2.5, 4)


async def facebook_search_topic(mcp, topic: str = "fitness"):
    await ensure_device_ready(mcp)
    await mcp.call_tool("start_app", {"package_name": "com.facebook.katana"})
    await asyncio.sleep(2)
    await mcp.call_tool("scroll_to", 
                                {"selector": "Home, tab 1 of 6",
                                "selector_type": "description"})

    await human_pause(0.5, 1)
    await mcp.call_tool("send_text", {"text": topic})
    await mcp.call_tool("press_key", {"key": "enter"})

    await asyncio.sleep(2)

    for _ in range(random.randint(2, 4)):
        await vertical_scroll(mcp)
        await human_pause()


async def facebook_maybe_post_status(mcp, **_):
    await ensure_device_ready(mcp)
    await mcp.call_tool("start_app", {"package_name": "com.facebook.katana"})
    await asyncio.sleep(2)
    await mcp.call_tool("scroll_to", 
                                {"selector": "Home, tab 1 of 6",
                                "selector_type": "description"})
    try:
        await mcp.call_tool(
            "click",
            {"selector": "What's on your mind?",
             "selector_type": "description"},
        )
    except Exception:
        return

    await human_pause(1, 2)

    text = random.choice(
        [
            "Good run today 💪",
            "Trying to stay consistent.",
            "Nice weather out today ☀️",
            "Long day, but productive.",
        ]
    )
    #TODO: Need to fix this
    await mcp.call_tool("send_text", {"text": text})
    await mcp.call_tool("click", {"selector": "Done", "selector_type": "text"})
    await human_pause(1, 2)


# ============================================================
# Instagram
# ============================================================

async def instagram_browse_feed(mcp, **_):
    await ensure_device_ready(mcp)
    await mcp.call_tool("start_app", {"package_name": "com.instagram.android"})
    await asyncio.sleep(3)
    await mcp.call_tool("click", {"selector": "Home", 
                                  "selector_type": "description"})

    for _ in range(random.randint(4, 8)):
        await vertical_scroll(mcp)
        await human_pause()

        # Occasionally like (tap center-ish)
        if random.random() < 0.2:
            await mcp.call_tool("click", {"x": 540, "y": 900})
            await human_pause(0.5, 1)


async def instagram_view_stories(mcp, **_):
    await ensure_device_ready(mcp)
    await mcp.call_tool("start_app", {"package_name": "com.instagram.android"})
    await asyncio.sleep(3)

    await mcp.call_tool("click", {"selector": "Reels", "selector_type": "description"})
    await asyncio.sleep(1.5)

    stories = random.randint(2, 5)
    for _ in range(stories):
        await vertical_scroll(mcp)
        await human_pause(1.2, 2.5)

        if random.random() < 0.15:
            break


async def instagram_view_reels(mcp, **_):
    await ensure_device_ready(mcp)
    await mcp.call_tool("start_app", {"package_name": "com.instagram.android"})
    await asyncio.sleep(3)

    await mcp.call_tool("click", {"selector": "Reels", 
                                  "selector_type": "description"})
    await asyncio.sleep(1.5)

    for _ in range(random.randint(3, 6)):
        await vertical_scroll(mcp)
        await human_pause(1.5, 3)

        if random.random() < 0.25:
            break


# ============================================================
# TikTok
# ============================================================

async def tiktok_watch_and_scroll(mcp, **_):
    await ensure_device_ready(mcp)
    await mcp.call_tool("start_app", {"package_name": "com.zhiliaoapp.musically"})
    await asyncio.sleep(4)

    for _ in range(random.randint(4, 7)):
        await human_pause(2, 4)
        await vertical_scroll(mcp)

        if random.random() < 0.2:
            break


# ============================================================
# LinkedIn
# ============================================================

async def linkedin_open_and_browse(mcp, **_):
    await ensure_device_ready(mcp)
    await mcp.call_tool("start_app", {"package_name": "com.linkedin.android"})
    await asyncio.sleep(3)

    for _ in range(random.randint(3, 5)):
        await vertical_scroll(mcp)
        await human_pause()

        if random.random() < 0.25:
            await human_pause(2, 3)


# ============================================================
# YouTube
# ============================================================

async def youtube_watch_recommended(mcp, **_):
    await ensure_device_ready(mcp)
    await mcp.call_tool("start_app", {"package_name": "com.google.android.youtube"})
    await asyncio.sleep(3)

    await mcp.call_tool("click", {"selector": "Shorts", "selector_type": "text"})
    await asyncio.sleep(random.uniform(6, 12))

    # Light interaction while watching
    if random.random() < 0.3:
        await vertical_scroll(mcp)
        await human_pause(1, 2)
