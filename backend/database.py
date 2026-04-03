# database.py

"""
SQLite persistence for personas.
Supports both local SQLite and Turso (libSQL) for serverless deployment.
"""

import json
import os
import secrets
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Any

import aiosqlite

from models import (
    ComprehensiveSurveyInput,
    BehavioralDimensions,
    BehavioralParameters,
    DailySchedule,
)

# ============================================================
# CONFIGURATION
# ============================================================

# Use environment variable for deployment flexibility
# Local: DATABASE_URL not set -> uses personas.db
# Turso: DATABASE_URL=libsql://your-db.turso.io?authToken=xxx
DATABASE_URL = os.getenv("DATABASE_URL", "personas.db")

# For local SQLite, extract just the path
if DATABASE_URL.startswith("libsql://"):
    # Turso URL - would need libsql-client for full support
    # For now, fall back to local SQLite
    DATABASE_PATH = Path("personas.db")
    print(
        f"⚠️ Turso URL detected but using local SQLite. Set up libsql-client for production."
    )
else:
    DATABASE_PATH = Path(DATABASE_URL)


# ============================================================
# SCHEMA
# ============================================================

SCHEMA_VERSION = 2

CREATE_TABLES_SQL = """
-- Main personas table
CREATE TABLE IF NOT EXISTS personas (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    
    -- Denormalized fields for quick queries/filtering
    city TEXT,
    occupation TEXT,
    age_range TEXT,
    
    -- Full JSON data
    survey_input TEXT NOT NULL,
    dimensions TEXT NOT NULL,
    parameters TEXT NOT NULL,
    weekday_schedule TEXT,
    weekend_schedule TEXT
);

-- Index for listing/filtering
CREATE INDEX IF NOT EXISTS idx_personas_created_at ON personas(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_personas_city ON personas(city);

-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
);
"""

MIGRATION_V2_SQL = """
-- Add denormalized columns if they don't exist
ALTER TABLE personas ADD COLUMN city TEXT;
ALTER TABLE personas ADD COLUMN occupation TEXT;
ALTER TABLE personas ADD COLUMN age_range TEXT;
"""


# ============================================================
# INITIALIZATION & MIGRATIONS
# ============================================================


async def init_db():
    """Initialize database schema with migrations."""
    async with aiosqlite.connect(DATABASE_PATH) as db:
        # Check current schema version
        try:
            async with db.execute("SELECT MAX(version) FROM schema_version") as cursor:
                row = await cursor.fetchone()
                current_version = row[0] if row and row[0] else 0
        except aiosqlite.OperationalError:
            # Table doesn't exist yet
            current_version = 0

        # Apply base schema
        if current_version == 0:
            await db.executescript(CREATE_TABLES_SQL)
            await db.execute(
                "INSERT OR REPLACE INTO schema_version (version) VALUES (?)",
                (SCHEMA_VERSION,),
            )
            await db.commit()
            print(f"✅ Database initialized at {DATABASE_PATH}")
            return

        # Apply migrations
        if current_version < 2:
            try:
                # Add new columns (SQLite ADD COLUMN is idempotent-ish)
                await db.execute("ALTER TABLE personas ADD COLUMN city TEXT")
            except aiosqlite.OperationalError:
                pass  # Column already exists

            try:
                await db.execute("ALTER TABLE personas ADD COLUMN occupation TEXT")
            except aiosqlite.OperationalError:
                pass

            try:
                await db.execute("ALTER TABLE personas ADD COLUMN age_range TEXT")
            except aiosqlite.OperationalError:
                pass

            # Backfill from JSON for existing rows
            await _backfill_denormalized_fields(db)

            await db.execute(
                "INSERT OR REPLACE INTO schema_version (version) VALUES (?)", (2,)
            )
            await db.commit()
            print(f"✅ Database migrated to version 2")


async def _backfill_denormalized_fields(db: aiosqlite.Connection):
    """Backfill city/occupation/age_range from survey_input JSON."""
    async with db.execute(
        "SELECT id, survey_input FROM personas WHERE city IS NULL"
    ) as cursor:
        rows = await cursor.fetchall()

    for row in rows:
        persona_id, survey_json = row
        try:
            survey = json.loads(survey_json)
            await db.execute(
                """
                UPDATE personas 
                SET city = ?, occupation = ?, age_range = ?
                WHERE id = ?
                """,
                (
                    survey.get("city"),
                    survey.get("occupation"),
                    survey.get("age_range"),
                    persona_id,
                ),
            )
        except (json.JSONDecodeError, TypeError):
            pass


# ============================================================
# CRUD OPERATIONS
# ============================================================


async def save_persona(
    survey: ComprehensiveSurveyInput,
    dimensions: BehavioralDimensions,
    parameters: BehavioralParameters,
    weekday_schedule: Optional[DailySchedule] = None,
    weekend_schedule: Optional[DailySchedule] = None,
) -> str:
    """
    Save a generated persona to the database.

    Returns:
        Short persona ID for easy sharing (8 characters).
    """
    persona_id = secrets.token_urlsafe(24)

    async with aiosqlite.connect(DATABASE_PATH) as db:
        await db.execute(
            """
            INSERT INTO personas (
                id, created_at, 
                city, occupation, age_range,
                survey_input, dimensions, parameters, 
                weekday_schedule, weekend_schedule
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                persona_id,
                datetime.utcnow().isoformat(),
                survey.city,
                survey.occupation,
                survey.age_range,
                survey.model_dump_json(),
                dimensions.model_dump_json(),
                parameters.model_dump_json(),
                weekday_schedule.model_dump_json() if weekday_schedule else None,
                weekend_schedule.model_dump_json() if weekend_schedule else None,
            ),
        )
        await db.commit()

    return persona_id


async def get_persona(persona_id: str) -> Optional[dict]:
    """
    Retrieve a persona by ID.

    Returns dict with structure matching what frontend expects:
    {
        "persona_id": str,
        "created_at": str,
        "survey": dict,
        "dimensions": dict,
        "parameters": dict,
        "schedule": dict | None  (weekday schedule)
    }
    """
    async with aiosqlite.connect(DATABASE_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM personas WHERE id = ?", (persona_id,)
        ) as cursor:
            row = await cursor.fetchone()

            if not row:
                return None

            # Parse JSON fields
            survey = json.loads(row["survey_input"])
            dimensions = json.loads(row["dimensions"])
            parameters = json.loads(row["parameters"])
            weekday_schedule = (
                json.loads(row["weekday_schedule"]) if row["weekday_schedule"] else None
            )
            weekend_schedule = (
                json.loads(row["weekend_schedule"]) if row["weekend_schedule"] else None
            )

            return {
                "persona_id": row["id"],
                "created_at": row["created_at"],
                "survey": survey,
                "dimensions": dimensions,
                "parameters": parameters,
                "schedule": weekday_schedule,  # Primary schedule for display
                "weekday_schedule": weekday_schedule,
                "weekend_schedule": weekend_schedule,
            }


async def list_personas(
    limit: int = 20,
    city: Optional[str] = None,
    occupation: Optional[str] = None,
) -> List[dict]:
    """
    List recent personas with summary info.

    Args:
        limit: Maximum number of personas to return
        city: Optional filter by city (partial match)
        occupation: Optional filter by occupation (partial match)

    Returns list of dicts:
    [
        {
            "persona_id": str,
            "created_at": str,
            "city": str,
            "occupation": str,
            "age_range": str,
        },
        ...
    ]
    """
    async with aiosqlite.connect(DATABASE_PATH) as db:
        db.row_factory = aiosqlite.Row

        # Build query with optional filters
        query = "SELECT id, created_at, city, occupation, age_range FROM personas"
        params: List[Any] = []
        conditions = []

        if city:
            conditions.append("city LIKE ?")
            params.append(f"%{city}%")

        if occupation:
            conditions.append("occupation LIKE ?")
            params.append(f"%{occupation}%")

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        async with db.execute(query, params) as cursor:
            rows = await cursor.fetchall()

            return [
                {
                    "persona_id": row["id"],
                    "created_at": row["created_at"],
                    "city": row["city"] or "Unknown",
                    "occupation": row["occupation"] or "Unknown",
                    "age_range": row["age_range"] or "Unknown",
                }
                for row in rows
            ]


async def delete_persona(persona_id: str) -> bool:
    """
    Delete a persona by ID.

    Returns True if deleted, False if not found.
    """
    async with aiosqlite.connect(DATABASE_PATH) as db:
        cursor = await db.execute("DELETE FROM personas WHERE id = ?", (persona_id,))
        await db.commit()
        return cursor.rowcount > 0


async def get_stats() -> dict:
    """Get database statistics."""
    async with aiosqlite.connect(DATABASE_PATH) as db:
        # Total count
        async with db.execute("SELECT COUNT(*) FROM personas") as cursor:
            row = await cursor.fetchone()
            total = row[0] if row else 0

        # Count by city (top 10)
        async with db.execute(
            """
            SELECT city, COUNT(*) as count 
            FROM personas 
            WHERE city IS NOT NULL 
            GROUP BY city 
            ORDER BY count DESC 
            LIMIT 10
            """
        ) as cursor:
            cities = await cursor.fetchall()

        # Recent activity (last 7 days)
        async with db.execute(
            """
            SELECT DATE(created_at) as date, COUNT(*) as count
            FROM personas
            WHERE created_at >= datetime('now', '-7 days')
            GROUP BY DATE(created_at)
            ORDER BY date DESC
            """
        ) as cursor:
            recent = await cursor.fetchall()

        return {
            "total_personas": total,
            "top_cities": [{"city": c[0], "count": c[1]} for c in cities],
            "recent_activity": [{"date": r[0], "count": r[1]} for r in recent],
        }


# ============================================================
# EXPORT FUNCTIONS
# ============================================================


async def export_all_personas() -> List[dict]:
    """Export all personas as a list of dicts (for backup/migration)."""
    async with aiosqlite.connect(DATABASE_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM personas ORDER BY created_at") as cursor:
            rows = await cursor.fetchall()

            return [
                {
                    "persona_id": row["id"],
                    "created_at": row["created_at"],
                    "city": row["city"],
                    "occupation": row["occupation"],
                    "age_range": row["age_range"],
                    "survey": json.loads(row["survey_input"]),
                    "dimensions": json.loads(row["dimensions"]),
                    "parameters": json.loads(row["parameters"]),
                    "weekday_schedule": (
                        json.loads(row["weekday_schedule"])
                        if row["weekday_schedule"]
                        else None
                    ),
                    "weekend_schedule": (
                        json.loads(row["weekend_schedule"])
                        if row["weekend_schedule"]
                        else None
                    ),
                }
                for row in rows
            ]


async def import_personas(personas: List[dict]) -> int:
    """
    Import personas from a list of dicts.

    Returns number of personas imported.
    """
    count = 0
    async with aiosqlite.connect(DATABASE_PATH) as db:
        for p in personas:
            try:
                await db.execute(
                    """
                    INSERT OR IGNORE INTO personas (
                        id, created_at, city, occupation, age_range,
                        survey_input, dimensions, parameters,
                        weekday_schedule, weekend_schedule
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        p["persona_id"],
                        p.get("created_at", datetime.utcnow().isoformat()),
                        p.get("city"),
                        p.get("occupation"),
                        p.get("age_range"),
                        json.dumps(p["survey"]),
                        json.dumps(p["dimensions"]),
                        json.dumps(p["parameters"]),
                        (
                            json.dumps(p.get("weekday_schedule"))
                            if p.get("weekday_schedule")
                            else None
                        ),
                        (
                            json.dumps(p.get("weekend_schedule"))
                            if p.get("weekend_schedule")
                            else None
                        ),
                    ),
                )
                count += 1
            except Exception as e:
                print(f"Failed to import persona {p.get('persona_id')}: {e}")

        await db.commit()

    return count


async def delete_all_personas() -> int:
    """
    Delete all personas from the database.

    Returns:
        Number of rows deleted.
    """
    async with aiosqlite.connect(DATABASE_PATH) as db:
        cursor = await db.execute("DELETE FROM personas")
        await db.commit()
        return cursor.rowcount if cursor.rowcount is not None else 0
