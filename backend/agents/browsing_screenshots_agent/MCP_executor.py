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
        elif app == "camera":
            pkg = "com.google.android.GoogleCamera"
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
        elif action == "view_stories":
            await self._instagram_view_stories()
        else:
            raise ValueError(f"Unknown Instagram action {action}")

    async def _instagram_view_reels(self):
        # Try semantic click first
        clicked = False

        try:
            await self.session.call_tool(
                "click",
                {"description": "Reels", "timeout": 3000}
            )
            clicked = True
        except Exception:
            pass

        # Fallback: bottom-nav second tab (very common layout)
        if not clicked:
            width, height = 1080, 2400  # conservative default
            await self.session.call_tool(
                "click",
                {
                    "x": int(width * 0.5),
                    "y": int(height * 0.95),
                }
            )

        # Watch + scroll like a human
        for _ in range(3):
            await self.session.call_tool("wait", {"seconds": 3})
            await self.session.call_tool("swipe", {"direction": "up"})
