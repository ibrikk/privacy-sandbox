import asyncio
import random
from typing import Dict, Any

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
        "com.instagram.android.browse_feed": instagram_view_reels,
        "com.zhiliaoapp.musically.watch_and_scroll": tiktok_watch_and_scroll,
        "com.google.android.cameraextensions.take_selfie": camera_take_selfie,
        "com.linkedin.android.browse_feed": linkedin_browse_feed,
        "com.google.android.youtube.watch_recommended": youtube_watch_recommended,
    }

    fn = EXECUTORS.get(key)
    if not fn:
        print(f"⚠️ No executor mapped for {key}")
        return

    await fn(mcp, **args)

async def spotify_play_for_persona(mcp, **_):
    await mcp.call_tool("open_app", {"package": "com.spotify.music"})
    await mcp.call_tool("wait", {"ms": 3000})

    # Tap center-ish (Play / first recommendation)
    await mcp.call_tool("tap", {"x": 540, "y": 1200})
    await mcp.call_tool("wait", {"ms": 2000})

async def facebook_open_and_browse(mcp, **_):
    await mcp.call_tool("open_app", {"package": "com.facebook.katana"})
    await mcp.call_tool("wait", {"ms": 3000})

    # Scroll feed
    await mcp.call_tool("swipe", {
        "x1": 540, "y1": 1400,
        "x2": 540, "y2": 400,
        "duration": 400
    })

async def facebook_search_topic(mcp, topic: str = "fitness"):
    await mcp.call_tool("open_app", {"package": "com.facebook.katana"})
    await mcp.call_tool("wait", {"ms": 2000})

    # Tap search bar (top)
    await mcp.call_tool("tap", {"x": 540, "y": 180})
    await mcp.call_tool("wait", {"ms": 500})

    await mcp.call_tool("type", {"text": topic})
    await mcp.call_tool("press", {"key": "enter"})

async def instagram_view_stories(mcp, **_):
    await mcp.call_tool("open_app", {"package": "com.instagram.android"})
    await mcp.call_tool("wait", {"ms": 3000})

    # Tap first story bubble
    await mcp.call_tool("tap", {"x": 180, "y": 260})
    await mcp.call_tool("wait", {"ms": 1500})

    # Advance stories
    await mcp.call_tool("tap", {"x": 900, "y": 800})

async def instagram_view_reels(mcp, **_):
    await mcp.call_tool("open_app", {"package": "com.instagram.android"})
    await mcp.call_tool("wait", {"ms": 3000})

    # Reels tab
    await mcp.call_tool("tap", {"x": 800, "y": 2200})
    await mcp.call_tool("wait", {"ms": 1500})

    await mcp.call_tool("swipe", {
        "x1": 540, "y1": 1600,
        "x2": 540, "y2": 400,
        "duration": 300
    })

async def tiktok_watch_and_scroll(mcp, **_):
    await mcp.call_tool("open_app", {"package": "com.zhiliaoapp.musically"})
    await mcp.call_tool("wait", {"ms": 4000})

    await mcp.call_tool("swipe", {
        "x1": 540, "y1": 1600,
        "x2": 540, "y2": 400,
        "duration": 300
    })

async def camera_take_selfie(mcp, **_):
    await mcp.call_tool("open_app", {"package": "com.google.android.GoogleCamera"})
    await mcp.call_tool("wait", {"ms": 3000})

    # Flip camera
    await mcp.call_tool("tap", {"x": 900, "y": 200})
    await mcp.call_tool("wait", {"ms": 1000})

    # Shutter
    await mcp.call_tool("tap", {"x": 540, "y": 1900})


async def linkedin_browse_feed(mcp, **_):
    await mcp.call_tool("open_app", {"package": "com.linkedin.android"})
    await mcp.call_tool("wait", {"ms": 3000})

    await mcp.call_tool("swipe", {
        "x1": 540, "y1": 1500,
        "x2": 540, "y2": 500,
        "duration": 400
    })


async def youtube_watch_recommended(mcp, **_):
    await mcp.call_tool("open_app", {"package": "com.google.android.youtube"})
    await mcp.call_tool("wait", {"ms": 3000})

    # Tap first recommended video
    await mcp.call_tool("tap", {"x": 540, "y": 600})
    await mcp.call_tool("wait", {"ms": 5000})
    
async def facebook_maybe_post_status(self):
        """
        Possibly post a short Facebook status.
        Human-like: often opens composer but does not always post.
        """

        print("🟦 Facebook: maybe_post_status")

        # 1️⃣ Ensure Facebook is open
        await self.session.call_tool(
            "start_app",
            {"package": "com.facebook.katana"}
        )
        await asyncio.sleep(2)

        # 2️⃣ Try to open status composer
        opened_composer = False

        try:
            # Common Facebook UI text
            await self.session.call_tool(
                "click",
                {"description": "What's on your mind", "timeout": 4000}
            )
            opened_composer = True
        except Exception:
            pass

        if not opened_composer:
            try:
                await self.session.call_tool(
                    "click",
                    {"description": "Write something here", "timeout": 4000}
                )
                opened_composer = True
            except Exception:
                pass

        if not opened_composer:
            print("⚠️ Could not open status composer")
            return

        await asyncio.sleep(1.5)

        # 3️⃣ Decide whether to actually post
        should_post = random.random() < 0.35  # 35% chance

        if not should_post:
            print("🤷 Decided not to post — backing out")
            await self.session.call_tool("press", {"key": "back"})
            await asyncio.sleep(1)
            return

        # 4️⃣ Type a short, realistic status
        status_templates = [
            "Good run today 💪",
            "Feeling productive today.",
            "Long day, but a good one.",
            "Trying to stay consistent.",
            "Nice weather out today ☀️",
        ]

        status_text = random.choice(status_templates)

        await self.session.call_tool(
            "type_text",
            {"text": status_text}
        )

        await asyncio.sleep(1.5)

        # 5️⃣ Post
        try:
            await self.session.call_tool(
                "click",
                {"description": "Post", "timeout": 3000}
            )
            print("✅ Status posted:", status_text)
        except Exception:
            print("⚠️ Failed to click Post, backing out")
            await self.session.call_tool("press", {"key": "back"})



