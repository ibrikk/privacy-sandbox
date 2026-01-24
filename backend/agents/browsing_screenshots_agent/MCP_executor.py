import asyncio
import random


class MCPExecutor:
    def __init__(self, session):
        self.session = session

    async def execute(self, step: dict):
        app = step["app"]
        action = step["action"]
        args = step.get("args", {})

        # Normalize app → package
        if app == "spotify":
            pkg = "com.spotify.music"
        elif app == "facebook":
            pkg = "com.facebook.katana"
        elif app == "instagram":
            pkg = "com.instagram.android"
        elif app == "tiktok":
            pkg = "com.zhiliaoapp.musically"
        # elif app == "camera":
        #     pkg = "com.google.android.GoogleCamera"
        else:
            raise ValueError(f"Unknown app {app}")

        # Always start app first
        await self.session.call_tool("start_app", {"package": pkg})

        # Action-specific behavior
        if action == "play_for_persona":
            pass  # Spotify auto-plays

        elif action == "open_and_browse":
            await self.session.call_tool("swipe", {"direction": "up"})

        elif action == "view_reels":
            await self.session.call_tool("click", {"description": "Reels"})
            await self.session.call_tool("swipe", {"direction": "up"})

        elif action == "take_selfie":
            await self.session.call_tool("press_key", {"key": "camera"})

        else:
            raise ValueError(f"Unhandled action {action}")
    
    async def _instagram(self, action: str):
        pkg = "com.instagram.android"

        await self.session.call_tool("start_app", {"package": pkg})
        await self.session.call_tool("wait_activity", {"package": pkg})

        if action == "view_reels":
            await self._instagram_view_reels()
        elif action == "browse_feed":
            await self._instagram_browse_feed()
        else:
            raise ValueError(f"Unknown Instagram action {action}")

    async def _instagram_view_reels(self):
        try:
            await self.session.call_tool(
                "click",
                {"description": "Reels", "timeout": 3000}
            )
        except Exception:
            pass

        # Watch + scroll like a human
        for _ in range(3):
            await self.session.call_tool("wait", {"seconds": 3})
            await self.session.call_tool("swipe", {"direction": "up"})
    
    async def _instagram_browse_feed(self):
        """
        Browse the Instagram home feed in a realistic way:
        - Ensure we're on Home
        - Scroll the feed
        - Occasionally open a post
        """

        # 1️⃣ Go to Home / Feed tab
        try:
            await self.session.call_tool(
                "click",
                {"description": "Home", "timeout": 3000}
            )
        except Exception:
            # Fallback: try icon-based or content-desc-based
            try:
                await self.session.call_tool(
                    "click",
                    {"description": "Feed", "timeout": 3000}
                )
            except Exception:
                pass  # If already on feed, this is fine

        await asyncio.sleep(random.uniform(1.0, 2.0))

        # 2️⃣ Scroll feed naturally
        scroll_count = random.randint(3, 6)

        for _ in range(scroll_count):
            await self.session.call_tool(
                "swipe",
                {
                    "direction": "up",
                    "distance": random.uniform(0.6, 0.8),
                    "duration": random.randint(300, 600),
                },
            )
            await asyncio.sleep(random.uniform(1.2, 2.5))

            # 3️⃣ Occasionally tap a post
            if random.random() < 0.3:
                try:
                    await self.session.call_tool(
                        "click",
                        {
                            "description": "Like",
                            "timeout": 2000
                        }
                    )
                    await asyncio.sleep(random.uniform(0.8, 1.5))
                except Exception:
                    pass

        # 4️⃣ Optional: pause as if reading
        await asyncio.sleep(random.uniform(2.0, 4.0))

    