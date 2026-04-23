# database.py

"""
PostgreSQL persistence for personas.
Uses asyncpg for async PostgreSQL access.
Compatible with Neon, Supabase, or any PostgreSQL provider.
"""

import json
import os
import secrets
from datetime import datetime
from typing import Optional, List, Any
from contextlib import asynccontextmanager

import asyncpg

from models import (
    ComprehensiveSurveyInput,
    BehavioralDimensions,
    BehavioralParameters,
    DailySchedule,
)

# ============================================================
# CONFIGURATION
# ============================================================

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL environment variable is required. "
        "Example: postgresql://user:pass@host/dbname?sslmode=require"
    )

# Connection pool (initialized on startup)
_pool: Optional[asyncpg.Pool] = None


async def get_pool() -> asyncpg.Pool:
    """Get or create the connection pool."""
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(
            DATABASE_URL,
            min_size=1,
            max_size=10,
            command_timeout=60,
        )
    return _pool


async def close_pool():
    """Close the connection pool."""
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


# ============================================================
# SCHEMA
# ============================================================

CREATE_TABLES_SQL = """
-- Main personas table
CREATE TABLE IF NOT EXISTS personas (
    id TEXT PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Denormalized fields for quick queries/filtering
    city TEXT,
    occupation TEXT,
    age_range TEXT,
    
    -- Full JSON data
    survey_input JSONB NOT NULL,
    dimensions JSONB NOT NULL,
    parameters JSONB NOT NULL,
    weekday_schedule JSONB,
    weekend_schedule JSONB
);

-- Indexes for listing/filtering
CREATE INDEX IF NOT EXISTS idx_personas_created_at ON personas(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_personas_city ON personas(city);
CREATE INDEX IF NOT EXISTS idx_personas_occupation ON personas(occupation);
"""


# ============================================================
# INITIALIZATION
# ============================================================


async def init_db():
    """Initialize database schema."""
    pool = await get_pool()

    async with pool.acquire() as conn:
        await conn.execute(CREATE_TABLES_SQL)

    print("✅ PostgreSQL database initialized")


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
        Persona ID (URL-safe token).
    """
    persona_id = secrets.token_urlsafe(24)
    pool = await get_pool()

    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO personas (
                id, city, occupation, age_range,
                survey_input, dimensions, parameters,
                weekday_schedule, weekend_schedule
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """,
            persona_id,
            survey.city,
            survey.occupation,
            (
                survey.age_range.value
                if hasattr(survey.age_range, "value")
                else str(survey.age_range)
            ),
            survey.model_dump_json(),
            dimensions.model_dump_json(),
            parameters.model_dump_json(),
            weekday_schedule.model_dump_json() if weekday_schedule else None,
            weekend_schedule.model_dump_json() if weekend_schedule else None,
        )

    return persona_id


async def get_persona(persona_id: str) -> Optional[dict]:
    """Retrieve a persona by ID."""
    pool = await get_pool()

    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM personas WHERE id = $1", persona_id)

        if not row:
            return None

        # Parse JSON fields (asyncpg returns dicts for JSONB)
        survey = (
            row["survey_input"]
            if isinstance(row["survey_input"], dict)
            else json.loads(row["survey_input"])
        )
        dimensions = (
            row["dimensions"]
            if isinstance(row["dimensions"], dict)
            else json.loads(row["dimensions"])
        )
        parameters = (
            row["parameters"]
            if isinstance(row["parameters"], dict)
            else json.loads(row["parameters"])
        )
        weekday_schedule = row["weekday_schedule"]
        weekend_schedule = row["weekend_schedule"]

        if weekday_schedule and isinstance(weekday_schedule, str):
            weekday_schedule = json.loads(weekday_schedule)
        if weekend_schedule and isinstance(weekend_schedule, str):
            weekend_schedule = json.loads(weekend_schedule)

        return {
            "persona_id": row["id"],
            "created_at": row["created_at"].isoformat() if row["created_at"] else None,
            "survey": survey,
            "dimensions": dimensions,
            "parameters": parameters,
            "schedule": weekday_schedule,
            "weekday_schedule": weekday_schedule,
            "weekend_schedule": weekend_schedule,
        }


async def list_personas(
    limit: int = 20,
    city: Optional[str] = None,
    occupation: Optional[str] = None,
) -> List[dict]:
    """List recent personas with optional filtering."""
    pool = await get_pool()

    # Build query with optional filters
    query = """
        SELECT id, created_at, city, occupation, age_range 
        FROM personas
        WHERE 1=1
    """
    params: List[Any] = []
    param_count = 0

    if city:
        param_count += 1
        query += f" AND city ILIKE ${param_count}"
        params.append(f"%{city}%")

    if occupation:
        param_count += 1
        query += f" AND occupation ILIKE ${param_count}"
        params.append(f"%{occupation}%")

    param_count += 1
    query += f" ORDER BY created_at DESC LIMIT ${param_count}"
    params.append(limit)

    async with pool.acquire() as conn:
        rows = await conn.fetch(query, *params)

        return [
            {
                "persona_id": row["id"],
                "created_at": (
                    row["created_at"].isoformat() if row["created_at"] else None
                ),
                "city": row["city"] or "Unknown",
                "occupation": row["occupation"] or "Unknown",
                "age_range": row["age_range"] or "Unknown",
            }
            for row in rows
        ]


async def delete_persona(persona_id: str) -> bool:
    """Delete a persona by ID."""
    pool = await get_pool()

    async with pool.acquire() as conn:
        result = await conn.execute("DELETE FROM personas WHERE id = $1", persona_id)
        # Result is like "DELETE 1" or "DELETE 0"
        return result.split()[-1] != "0"


async def get_stats() -> dict:
    """Get database statistics."""
    pool = await get_pool()

    async with pool.acquire() as conn:
        # Total count
        total = await conn.fetchval("SELECT COUNT(*) FROM personas")

        # Top cities
        cities = await conn.fetch(
            """
            SELECT city, COUNT(*) as count 
            FROM personas 
            WHERE city IS NOT NULL 
            GROUP BY city 
            ORDER BY count DESC 
            LIMIT 10
        """
        )

        # Recent activity
        recent = await conn.fetch(
            """
            SELECT DATE(created_at) as date, COUNT(*) as count
            FROM personas
            WHERE created_at >= NOW() - INTERVAL '7 days'
            GROUP BY DATE(created_at)
            ORDER BY date DESC
        """
        )

        return {
            "total_personas": total or 0,
            "top_cities": [{"city": r["city"], "count": r["count"]} for r in cities],
            "recent_activity": [
                {
                    "date": r["date"].isoformat() if r["date"] else None,
                    "count": r["count"],
                }
                for r in recent
            ],
        }


async def export_all_personas() -> List[dict]:
    """Export all personas."""
    pool = await get_pool()

    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT * FROM personas ORDER BY created_at")

        results = []
        for row in rows:
            survey = (
                row["survey_input"]
                if isinstance(row["survey_input"], dict)
                else json.loads(row["survey_input"])
            )
            dimensions = (
                row["dimensions"]
                if isinstance(row["dimensions"], dict)
                else json.loads(row["dimensions"])
            )
            parameters = (
                row["parameters"]
                if isinstance(row["parameters"], dict)
                else json.loads(row["parameters"])
            )
            weekday = row["weekday_schedule"]
            weekend = row["weekend_schedule"]

            if weekday and isinstance(weekday, str):
                weekday = json.loads(weekday)
            if weekend and isinstance(weekend, str):
                weekend = json.loads(weekend)

            results.append(
                {
                    "persona_id": row["id"],
                    "created_at": (
                        row["created_at"].isoformat() if row["created_at"] else None
                    ),
                    "city": row["city"],
                    "occupation": row["occupation"],
                    "age_range": row["age_range"],
                    "survey": survey,
                    "dimensions": dimensions,
                    "parameters": parameters,
                    "weekday_schedule": weekday,
                    "weekend_schedule": weekend,
                }
            )

        return results


async def delete_all_personas() -> int:
    """Delete all personas."""
    pool = await get_pool()

    async with pool.acquire() as conn:
        result = await conn.execute("DELETE FROM personas")
        # Result is like "DELETE 42"
        try:
            return int(result.split()[-1])
        except (ValueError, IndexError):
            return 0
