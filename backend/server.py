# server.py

"""
FastAPI server for the Synthetic Smartphone Persona Generator.

Endpoints:
    POST /generate - Generate a persona from survey responses
    GET /persona/{id} - Retrieve a saved persona
    GET /personas - List recent personas
    GET /health - Health check
"""

import uuid
import os
from datetime import datetime, date
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, status, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import uvicorn

# Import database functions
from database import (
    init_db,
    save_persona,
    get_persona,
    list_personas,
    get_stats,
    export_all_personas,
    delete_all_personas,
)

# Import our models
from models import (
    # Enums
    AgeRange,
    AreaType,
    WakeTime,
    SleepTime,
    ChronotypeLabel,
    PeakUsageTime,
    RoutineStructure,
    CommuteDays,
    CommuteMode,
    PhysicalActivityDays,
    CommuteTime,
    ScreenTime,
    CheckingFrequency,
    SessionType,
    GlanceFrequency,
    WorkPhoneRestriction,
    EveningSessionChange,
    AppCategory,
    UsageReason,
    ExplorationStyle,
    # Main models
    ComprehensiveSurveyInput,
    BehavioralDimensions,
    BehavioralParameters,
    DailySchedule,
)

# Import engine
from engine import PersonaEngine


# ============================================================
# API MODELS
# ============================================================


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    timestamp: str
    version: str
    engine_ready: bool


class SimpleGenerateResponse(BaseModel):
    """Response for persona generation."""

    persona_id: str
    created_at: str
    survey_summary: Dict[str, Any]
    dimensions: Dict[str, Any]
    parameters: Dict[str, Any]
    schedule: Optional[Dict[str, Any]] = None


# ============================================================
# APPLICATION SETUP
# ============================================================

engine: Optional[PersonaEngine] = None
OUTPUT_DIR = Path("output/personas")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    global engine

    print("🚀 Starting Persona Generator API...")

    # Initialize database
    await init_db()
    print("✅ Database initialized")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize engine
    try:
        engine = PersonaEngine()
        print("✅ PersonaEngine initialized")
    except Exception as e:
        print(f"⚠️ Engine initialization warning: {e}")
        engine = PersonaEngine()

    yield

    print("👋 Shutting down...")


app = FastAPI(
    title="Synthetic Smartphone Persona Generator",
    description="Generate realistic smartphone user personas from survey responses.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================


def parse_survey_from_frontend(data: Dict[str, Any]) -> ComprehensiveSurveyInput:
    """
    Parse survey data from frontend format to ComprehensiveSurveyInput.

    Handles field name differences between frontend and backend.
    """
    # Map frontend field names to backend field names
    parsed = {
        # Demographics
        "age_range": data.get("age_range", "25-34"),
        "city": data.get("city", "Unknown"),
        "occupation": data.get("occupation", "Unknown"),
        "area_type": data.get("area_type", "urban"),
        # Sleep & Chronotype
        # Frontend sends: wake_time, sleep_time, chronotype, peak_usage_time
        "wake_time": data.get("wake_time", "6am-8am"),
        "sleep_time": data.get("sleep_time", "10pm-12am"),
        "chronotype_self_report": data.get(
            "chronotype", data.get("chronotype_self_report", "neither")
        ),
        "peak_usage_time": data.get("peak_usage_time", "evening"),
        # Daily Structure
        # Frontend sends: routine_level, commute_days, commute_mode, commute_time, physical_activity_days
        "routine_structure": data.get(
            "routine_level", data.get("routine_structure", "mixed")
        ),
        "places_visited_daily": int(data.get("places_visited_daily", 3)),
        "commute_days": data.get("commute_days", "3-4"),
        "commute_mode": data.get("commute_mode", "mixed"),
        "commute_time": data.get("commute_time", "30-60min"),
        "physical_activity_days": data.get("physical_activity_days", "1-2"),
        # Phone Usage
        # Frontend sends: screen_time, checking_frequency, session_type, glance_frequency,
        #                 work_phone_restriction, evening_usage_change
        "daily_screen_time": data.get(
            "screen_time", data.get("daily_screen_time", "2-4hours")
        ),
        "checking_frequency": data.get("checking_frequency", "3-4_per_hour"),
        "session_type": data.get("session_type", "mixed"),
        "glance_frequency": data.get("glance_frequency", "sometimes"),
        "work_phone_restriction": data.get("work_phone_restriction", "occasionally"),
        "evening_session_change": data.get(
            "evening_usage_change", data.get("evening_session_change", "about_same")
        ),
        # App Preferences
        # Frontend sends: top_app_categories, usage_reasons, exploration_preference
        "most_used_categories": data.get(
            "top_app_categories", data.get("most_used_categories", ["social_media"])
        ),
        "usage_reasons": data.get("usage_reasons", []),
        "exploration_style": data.get(
            "exploration_preference", data.get("exploration_style", "sometimes_explore")
        ),
        # Optional fields
        "evening_activities_increase": data.get("evening_activities_increase", []),
        "commute_activities": data.get("commute_activities", []),
        "top_apps": data.get("top_apps"),
        "important_habit": data.get("important_habit"),
    }

    # Ensure list fields are actually lists
    for field in [
        "most_used_categories",
        "usage_reasons",
        "evening_activities_increase",
        "commute_activities",
    ]:
        if not isinstance(parsed.get(field), list):
            parsed[field] = [parsed[field]] if parsed.get(field) else []

    return ComprehensiveSurveyInput(**parsed)


def serialize_model(model: BaseModel) -> Dict[str, Any]:
    """Serialize a Pydantic model to dict, handling enums."""
    data = model.model_dump()

    def convert_value(v):
        if hasattr(v, "value"):
            return v.value
        elif isinstance(v, (datetime, date)):
            return v.isoformat()
        elif isinstance(v, dict):
            return {k: convert_value(val) for k, val in v.items()}
        elif isinstance(v, list):
            return [convert_value(item) for item in v]
        return v

    return {k: convert_value(v) for k, v in data.items()}


def create_survey_summary(survey: ComprehensiveSurveyInput) -> Dict[str, Any]:
    """Create a summary dict from survey input."""
    return {
        "age_range": (
            survey.age_range.value
            if hasattr(survey.age_range, "value")
            else str(survey.age_range)
        ),
        "city": survey.city,
        "occupation": survey.occupation,
        "area_type": (
            survey.area_type.value
            if hasattr(survey.area_type, "value")
            else str(survey.area_type)
        ),
    }


# ============================================================
# ENDPOINTS
# ============================================================


@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint."""
    return {
        "message": "Synthetic Smartphone Persona Generator API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        version="1.0.0",
        engine_ready=engine is not None,
    )


@app.post("/generate", response_model=SimpleGenerateResponse, tags=["Generation"])
async def generate_persona(request: Dict[str, Any]):
    """
    Generate a synthetic persona from survey responses.

    Accepts survey data directly as JSON body (flat structure from frontend).
    """
    global engine

    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Engine not initialized",
        )

    try:
        # Parse the survey data
        # Frontend sends flat dict, not nested under "survey"
        survey_data = request.get("survey", request)  # Support both formats
        survey = parse_survey_from_frontend(survey_data)

        # Generate dimensions and parameters
        dimensions, parameters = engine.generate_persona(survey)

        # Generate schedule
        day_type = request.get("day_type", "weekday")
        generate_schedule = request.get("generate_schedule", True)

        schedule = None
        schedule_obj = None
        if generate_schedule:
            schedule_obj = engine.generate_schedule(
                dimensions=dimensions,
                parameters=parameters,
                survey=survey,
                day_type=day_type,
            )
            if schedule_obj:
                schedule = serialize_model(schedule_obj)

        # Save to database
        persona_id = await save_persona(
            survey=survey,
            dimensions=dimensions,
            parameters=parameters,
            weekday_schedule=schedule_obj if day_type == "weekday" else None,
            weekend_schedule=schedule_obj if day_type == "weekend" else None,
        )

        # Build response
        return SimpleGenerateResponse(
            persona_id=persona_id,
            created_at=datetime.utcnow().isoformat(),
            survey_summary=create_survey_summary(survey),
            dimensions=serialize_model(dimensions),
            parameters=serialize_model(parameters),
            schedule=schedule,
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid survey data: {str(e)}",
        )
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generation failed: {str(e)}",
        )


@app.get("/persona/{persona_id}", tags=["Personas"])
async def get_persona_endpoint(persona_id: str):
    """Retrieve a persona by ID."""
    persona = await get_persona(persona_id)
    if not persona:
        raise HTTPException(status_code=404, detail="Persona not found")
    return persona


ADMIN_API_KEY = os.getenv("ADMIN_API_KEY")


async def require_admin(x_api_key: str | None = Header(default=None)):
    if not ADMIN_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="ADMIN_API_KEY is not configured on the server",
        )

    if x_api_key != ADMIN_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
        )

    return True


@app.get("/personas", tags=["Personas"])
async def list_personas_endpoint(
    limit: int = 20,
    city: Optional[str] = None,
    occupation: Optional[str] = None,
    _: bool = Depends(require_admin),
):
    personas = await list_personas(limit=limit, city=city, occupation=occupation)
    return {"personas": personas}


@app.get("/stats", tags=["System"])
async def get_stats_endpoint(_: bool = Depends(require_admin)):
    return await get_stats()


@app.get("/export", tags=["System"])
async def export_endpoint(_: bool = Depends(require_admin)):
    personas = await export_all_personas()
    return {"count": len(personas), "personas": personas}


@app.delete("/personas", tags=["Personas"])
async def delete_all_personas_endpoint(_: bool = Depends(require_admin)):
    deleted = await delete_all_personas()
    return {
        "message": "All personas deleted",
        "deleted_count": deleted,
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
