# server.py

"""
FastAPI server for the Synthetic Smartphone Persona Generator.

Endpoints:
    POST /generate - Generate a persona from survey responses
    GET /health - Health check
    GET /schema/survey - Get survey field schema
    POST /batch/generate - Batch generate multiple personas
"""

import uuid
import os
from datetime import datetime, date
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import uvicorn

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
    LiteratureReference,
    GenerationRequest,
    GenerationResponse,
)

# Import engine
from engine import PersonaEngine


# ============================================================
# ADDITIONAL API MODELS
# ============================================================


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    timestamp: str
    version: str
    engine_ready: bool


class SurveySchemaResponse(BaseModel):
    """Survey schema for frontend reference."""

    fields: Dict[str, Any]
    enums: Dict[str, List[str]]


class BatchGenerateRequest(BaseModel):
    """Request for batch persona generation."""

    surveys: List[ComprehensiveSurveyInput] = Field(..., max_length=100)
    generate_schedule: bool = True
    day_type: Optional[str] = "weekday"


class BatchGenerateResponse(BaseModel):
    """Response for batch generation."""

    total: int
    successful: int
    failed: int
    results: List[Dict[str, Any]]


class SimpleGenerateRequest(BaseModel):
    """
    Simplified request that matches what the Streamlit frontend sends.
    This wraps the survey data for easier frontend integration.
    """

    survey: Dict[str, Any] = Field(..., description="Survey responses as dict")
    generate_schedule: bool = True
    day_type: str = "weekday"


class SimpleGenerateResponse(BaseModel):
    """Simplified response for frontend consumption."""

    persona_id: str
    created_at: str
    survey_summary: Dict[str, Any]
    dimensions: Dict[str, Any]
    parameters: Dict[str, Any]
    schedule: Optional[Dict[str, Any]] = None
    generation_metadata: Dict[str, Any] = Field(default_factory=dict)


# ============================================================
# APPLICATION SETUP
# ============================================================

# Engine instance (initialized on startup)
engine: Optional[PersonaEngine] = None

# Output directory for saved personas
OUTPUT_DIR = Path("output/personas")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    global engine

    # Startup
    print("🚀 Starting Persona Generator API...")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize engine
    try:
        engine = PersonaEngine()
        print("✅ PersonaEngine initialized")
    except Exception as e:
        print(f"⚠️ Engine initialization warning: {e}")
        engine = PersonaEngine()  # Try basic init

    yield

    # Shutdown
    print("👋 Shutting down...")


app = FastAPI(
    title="Synthetic Smartphone Persona Generator",
    description="""
    Generate realistic smartphone user personas from survey responses.
    
    This API takes comprehensive survey data about a user's demographics, 
    sleep patterns, daily structure, phone usage habits, and app preferences,
    then generates:
    
    - **Behavioral Dimensions**: 8 normalized scores characterizing the user
    - **Behavioral Parameters**: Concrete simulation parameters with literature grounding
    - **Daily Schedule**: Optional 24-hour schedule with expected phone sessions
    """,
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================


def parse_survey_dict(data: Dict[str, Any]) -> ComprehensiveSurveyInput:
    """
    Parse a survey dictionary into ComprehensiveSurveyInput.
    Handles both enum values and string values.
    """
    # Convert string values to enums where needed
    parsed = {}

    # Direct mappings
    parsed["age_range"] = data.get("age_range", "25-34")
    parsed["city"] = data.get("city", "Unknown")
    parsed["occupation"] = data.get("occupation", "Unknown")
    parsed["area_type"] = data.get("area_type", "urban")

    parsed["wake_time"] = data.get("wake_time", "6am-8am")
    parsed["sleep_time"] = data.get("sleep_time", "10pm-12am")
    parsed["chronotype_self_report"] = data.get("chronotype_self_report", "neither")
    parsed["peak_usage_time"] = data.get("peak_usage_time", "evening")

    parsed["routine_structure"] = data.get("routine_structure", "mixed")
    parsed["places_visited_daily"] = int(data.get("places_visited_daily", 3))
    parsed["commute_days"] = data.get("commute_days", "3-4")
    parsed["commute_mode"] = data.get("commute_mode", "mixed")
    parsed["physical_activity_days"] = data.get("physical_activity_days", "1-2")
    parsed["commute_time"] = data.get("commute_time", "30-60min")

    parsed["daily_screen_time"] = data.get("daily_screen_time", "2-4hours")
    parsed["checking_frequency"] = data.get("checking_frequency", "3-4_per_hour")
    parsed["session_type"] = data.get("session_type", "mixed")
    parsed["glance_frequency"] = data.get("glance_frequency", "sometimes")
    parsed["work_phone_restriction"] = data.get(
        "work_phone_restriction", "occasionally"
    )
    parsed["evening_session_change"] = data.get("evening_session_change", "about_same")

    # Lists - ensure they're lists
    parsed["evening_activities_increase"] = data.get("evening_activities_increase", [])
    if not isinstance(parsed["evening_activities_increase"], list):
        parsed["evening_activities_increase"] = []

    parsed["usage_reasons"] = data.get("usage_reasons", [])
    if not isinstance(parsed["usage_reasons"], list):
        parsed["usage_reasons"] = []

    parsed["commute_activities"] = data.get("commute_activities", [])
    if not isinstance(parsed["commute_activities"], list):
        parsed["commute_activities"] = []

    parsed["most_used_categories"] = data.get("most_used_categories", ["social_media"])
    if not isinstance(parsed["most_used_categories"], list):
        parsed["most_used_categories"] = ["social_media"]

    parsed["exploration_style"] = data.get("exploration_style", "sometimes_explore")

    # Optional fields
    parsed["top_apps"] = data.get("top_apps")
    parsed["important_habit"] = data.get("important_habit")

    return ComprehensiveSurveyInput(**parsed)


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
        "chronotype": (
            survey.chronotype_self_report.value
            if hasattr(survey.chronotype_self_report, "value")
            else str(survey.chronotype_self_report)
        ),
        "screen_time": (
            survey.daily_screen_time.value
            if hasattr(survey.daily_screen_time, "value")
            else str(survey.daily_screen_time)
        ),
        "top_categories": [
            c.value if hasattr(c, "value") else str(c)
            for c in (
                survey.most_used_categories[:3] if survey.most_used_categories else []
            )
        ],
    }


def serialize_model(model: BaseModel) -> Dict[str, Any]:
    """Serialize a Pydantic model to dict, handling enums and special types."""
    data = model.model_dump()

    def convert_value(v):
        if hasattr(v, "value"):  # Enum
            return v.value
        elif isinstance(v, (datetime, date)):
            return v.isoformat()
        elif isinstance(v, dict):
            return {k: convert_value(val) for k, val in v.items()}
        elif isinstance(v, list):
            return [convert_value(item) for item in v]
        return v

    return {k: convert_value(v) for k, v in data.items()}


# ============================================================
# ENDPOINTS
# ============================================================


@app.get("/", include_in_schema=False)
async def root():
    """Root redirect to docs."""
    return {
        "message": "Synthetic Smartphone Persona Generator API",
        "version": "1.0.0",
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
async def generate_persona(request: SimpleGenerateRequest):
    """
    Generate a synthetic persona from survey responses.

    This endpoint takes survey responses and generates:
    - **8 behavioral dimensions** (chronotype, usage intensity, etc.)
    - **Simulation parameters** (screen time, session counts, app weights, etc.)
    - **24-hour schedule** (optional) with expected phone usage

    The generation uses literature-grounded heuristics and mappings.
    """
    global engine

    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Engine not initialized. Please restart the server.",
        )

    try:
        # Parse survey dict into model
        survey = parse_survey_dict(request.survey)

        # Generate persona ID
        persona_id = str(uuid.uuid4())

        # Generate dimensions and parameters using engine
        dimensions, parameters = engine.generate_persona(survey)

        # Generate schedule if requested
        schedule = None
        if request.generate_schedule:
            schedule_obj = engine.generate_schedule(
                dimensions=dimensions,
                parameters=parameters,
                survey=survey,
                day_type=request.day_type,
            )
            if schedule_obj:
                schedule = serialize_model(schedule_obj)

        # Build response
        return SimpleGenerateResponse(
            persona_id=persona_id,
            created_at=datetime.utcnow().isoformat(),
            survey_summary=create_survey_summary(survey),
            dimensions=serialize_model(dimensions),
            parameters=serialize_model(parameters),
            schedule=schedule,
            generation_metadata={
                "day_type": request.day_type,
                "schedule_generated": request.generate_schedule,
                "engine_version": "1.0.0",
            },
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


@app.post("/generate/full", response_model=GenerationResponse, tags=["Generation"])
async def generate_persona_full(request: GenerationRequest):
    """
    Generate a persona using the full GenerationRequest/Response models.

    This endpoint provides more detailed output including:
    - Literature citations for parameters
    - File paths for saved artifacts
    - Execution metrics (if run on device)
    """
    global engine

    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Engine not initialized",
        )

    try:
        persona_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()

        # Generate dimensions and parameters
        dimensions, parameters = engine.generate_persona(request.survey)

        # Generate schedule if requested
        schedule = None
        if request.generate_schedule:
            # Determine day type
            day_type = request.day_type
            if day_type is None and request.schedule_date:
                # Auto-detect from date
                try:
                    d = datetime.strptime(request.schedule_date, "%Y-%m-%d").date()
                    day_type = "weekend" if d.weekday() >= 5 else "weekday"
                except:
                    day_type = "weekday"
            elif day_type is None:
                day_type = "weekday"

            schedule = engine.generate_schedule(
                dimensions=dimensions,
                parameters=parameters,
                survey=request.survey,
                day_type=day_type,
                date_str=request.schedule_date,
            )

        # Create save directory
        save_dir = OUTPUT_DIR / persona_id
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save persona JSON
        import json

        persona_path = save_dir / "persona.json"
        with open(persona_path, "w") as f:
            json.dump(
                {
                    "persona_id": persona_id,
                    "timestamp": timestamp,
                    "survey": serialize_model(request.survey),
                    "dimensions": serialize_model(dimensions),
                    "parameters": serialize_model(parameters),
                    "schedule": serialize_model(schedule) if schedule else None,
                },
                f,
                indent=2,
                default=str,
            )

        # Build citations (from engine if available)
        citations = []
        if hasattr(engine, "get_citations"):
            citations = engine.get_citations()

        return GenerationResponse(
            persona_id=persona_id,
            generation_timestamp=timestamp,
            dimensions=dimensions,
            parameters=parameters,
            schedule=schedule,
            save_directory=str(save_dir),
            persona_json_path=str(persona_path),
            traces_tar_path=None,
            parameter_citations=citations,
            execution_metrics=None,
        )

    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generation failed: {str(e)}",
        )


@app.post("/batch/generate", response_model=BatchGenerateResponse, tags=["Generation"])
async def batch_generate(request: BatchGenerateRequest):
    """
    Generate multiple personas from a list of survey responses.

    Useful for generating a cohort of synthetic users.
    Maximum 100 surveys per batch.
    """
    global engine

    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Engine not initialized",
        )

    results = []

    for i, survey in enumerate(request.surveys):
        try:
            persona_id = str(uuid.uuid4())
            dimensions, parameters = engine.generate_persona(survey)

            schedule = None
            if request.generate_schedule:
                schedule_obj = engine.generate_schedule(
                    dimensions=dimensions,
                    parameters=parameters,
                    survey=survey,
                    day_type=request.day_type or "weekday",
                )
                schedule = serialize_model(schedule_obj) if schedule_obj else None

            results.append(
                {
                    "index": i,
                    "persona_id": persona_id,
                    "success": True,
                    "dimensions": serialize_model(dimensions),
                    "parameters": serialize_model(parameters),
                    "schedule": schedule,
                }
            )

        except Exception as e:
            results.append(
                {
                    "index": i,
                    "persona_id": None,
                    "success": False,
                    "error": str(e),
                }
            )

    return BatchGenerateResponse(
        total=len(request.surveys),
        successful=sum(1 for r in results if r.get("success")),
        failed=sum(1 for r in results if not r.get("success")),
        results=results,
    )


@app.get("/schema/survey", response_model=SurveySchemaResponse, tags=["Schema"])
async def get_survey_schema():
    """
    Get the survey field schema and enum values.

    Useful for frontends to dynamically build survey forms.
    """

    # Field descriptions matching the survey
    fields = {
        "age_range": {
            "type": "select",
            "required": True,
            "question": "Q1. What is your age?",
            "section": "Demographics",
        },
        "city": {
            "type": "text",
            "required": True,
            "question": "Q2. What city do you currently live in?",
            "section": "Demographics",
        },
        "occupation": {
            "type": "text",
            "required": True,
            "question": "Q3. What is your current occupation or primary role?",
            "section": "Demographics",
        },
        "area_type": {
            "type": "select",
            "required": True,
            "question": "Q4. How would you describe the area where you live?",
            "section": "Demographics",
        },
        "wake_time": {
            "type": "select",
            "required": True,
            "question": "Q5. On a typical weekday, what time do you usually wake up?",
            "section": "Sleep & Chronotype",
        },
        "sleep_time": {
            "type": "select",
            "required": True,
            "question": "Q6. On a typical weekday, what time do you usually go to sleep?",
            "section": "Sleep & Chronotype",
        },
        "chronotype_self_report": {
            "type": "select",
            "required": True,
            "question": "Q7. Would you describe yourself as a morning person or evening person?",
            "section": "Sleep & Chronotype",
        },
        "peak_usage_time": {
            "type": "select",
            "required": True,
            "question": "Q8. When do you typically use your phone the most?",
            "section": "Sleep & Chronotype",
        },
        "routine_structure": {
            "type": "select",
            "required": True,
            "question": "Q9. How would you describe your typical daily routine?",
            "section": "Daily Structure",
        },
        "places_visited_daily": {
            "type": "slider",
            "required": True,
            "min": 1,
            "max": 6,
            "question": "Q10. On a typical day, how many different places do you visit?",
            "section": "Daily Structure",
        },
        "commute_days": {
            "type": "select",
            "required": True,
            "question": "Q11. How many days per week do you commute to a workplace or school?",
            "section": "Daily Structure",
        },
        "commute_mode": {
            "type": "select",
            "required": True,
            "question": "Q12. What is your primary mode of transportation for commuting?",
            "section": "Daily Structure",
        },
        "physical_activity_days": {
            "type": "select",
            "required": True,
            "question": "Q13. How many days per week do you exercise or do physical activity?",
            "section": "Daily Structure",
        },
        "commute_time": {
            "type": "select",
            "required": True,
            "question": "Q14. How much total time do you typically spend traveling each day?",
            "section": "Daily Structure",
        },
        "daily_screen_time": {
            "type": "select",
            "required": True,
            "question": "Q15. On a typical day, how much total time do you spend on your smartphone?",
            "section": "Phone Usage Patterns",
        },
        "checking_frequency": {
            "type": "select",
            "required": True,
            "question": "Q16. How often do you check your phone?",
            "section": "Phone Usage Patterns",
        },
        "session_type": {
            "type": "select",
            "required": True,
            "question": "Q17. Which best describes your typical phone sessions?",
            "section": "Phone Usage Patterns",
        },
        "glance_frequency": {
            "type": "select",
            "required": True,
            "question": "Q18. How often do you quickly check your phone without unlocking?",
            "section": "Phone Usage Patterns",
        },
        "work_phone_restriction": {
            "type": "select",
            "required": True,
            "question": "Q19. When you're at work or school, how do you typically use your phone?",
            "section": "Phone Usage Patterns",
        },
        "evening_session_change": {
            "type": "select",
            "required": True,
            "question": "Q20. In the evening at home, are your phone sessions typically...",
            "section": "Phone Usage Patterns",
        },
        "evening_activities_increase": {
            "type": "multiselect",
            "required": False,
            "max_selections": 3,
            "question": "Q21. Which activities do you do MORE in the evening compared to daytime?",
            "section": "App & Content Preferences",
        },
        "usage_reasons": {
            "type": "multiselect",
            "required": False,
            "max_selections": 3,
            "question": "Q22. What are your most common reasons for using your phone?",
            "section": "App & Content Preferences",
        },
        "commute_activities": {
            "type": "multiselect",
            "required": False,
            "max_selections": 3,
            "question": "Q23. What do you typically do on your phone during commute or travel?",
            "section": "App & Content Preferences",
        },
        "most_used_categories": {
            "type": "multiselect",
            "required": True,
            "max_selections": 5,
            "question": "Q24. Which app categories do you use most frequently?",
            "section": "App & Content Preferences",
        },
        "exploration_style": {
            "type": "select",
            "required": True,
            "question": "Q25. When it comes to apps and content, which describes you best?",
            "section": "App & Content Preferences",
        },
        "top_apps": {
            "type": "text",
            "required": False,
            "question": "Q26. What are your top 3 most-used apps?",
            "section": "Additional Details",
        },
        "important_habit": {
            "type": "textarea",
            "required": False,
            "question": "Q27. Is there any specific phone habit you think is important to capture?",
            "section": "Additional Details",
        },
    }

    # Enum values from the models
    enums = {
        "age_range": [e.value for e in AgeRange],
        "area_type": [e.value for e in AreaType],
        "wake_time": [e.value for e in WakeTime],
        "sleep_time": [e.value for e in SleepTime],
        "chronotype_self_report": [e.value for e in ChronotypeLabel],
        "peak_usage_time": [e.value for e in PeakUsageTime],
        "routine_structure": [e.value for e in RoutineStructure],
        "commute_days": [e.value for e in CommuteDays],
        "commute_mode": [e.value for e in CommuteMode],
        "physical_activity_days": [e.value for e in PhysicalActivityDays],
        "commute_time": [e.value for e in CommuteTime],
        "daily_screen_time": [e.value for e in ScreenTime],
        "checking_frequency": [e.value for e in CheckingFrequency],
        "session_type": [e.value for e in SessionType],
        "glance_frequency": [e.value for e in GlanceFrequency],
        "work_phone_restriction": [e.value for e in WorkPhoneRestriction],
        "evening_session_change": [e.value for e in EveningSessionChange],
        "app_categories": [e.value for e in AppCategory],
        "usage_reasons": [e.value for e in UsageReason],
        "exploration_style": [e.value for e in ExplorationStyle],
    }

    return SurveySchemaResponse(fields=fields, enums=enums)


@app.get("/persona/{persona_id}", tags=["Personas"])
async def get_persona(persona_id: str):
    """
    Retrieve a previously generated persona by ID.
    """
    persona_path = OUTPUT_DIR / persona_id / "persona.json"

    if not persona_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Persona {persona_id} not found",
        )

    import json

    with open(persona_path) as f:
        return json.load(f)


@app.get("/persona/{persona_id}/download", tags=["Personas"])
async def download_persona(persona_id: str):
    """
    Download persona JSON file.
    """
    persona_path = OUTPUT_DIR / persona_id / "persona.json"

    if not persona_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Persona {persona_id} not found",
        )

    return FileResponse(
        path=persona_path,
        filename=f"persona_{persona_id}.json",
        media_type="application/json",
    )


@app.get("/personas", tags=["Personas"])
async def list_personas(limit: int = 50, offset: int = 0):
    """
    List all generated personas.
    """
    if not OUTPUT_DIR.exists():
        return {"total": 0, "personas": []}

    persona_dirs = sorted(OUTPUT_DIR.iterdir(), reverse=True)
    total = len(persona_dirs)

    personas = []
    for d in persona_dirs[offset : offset + limit]:
        if d.is_dir():
            persona_json = d / "persona.json"
            if persona_json.exists():
                import json

                with open(persona_json) as f:
                    data = json.load(f)
                    personas.append(
                        {
                            "persona_id": data.get("persona_id", d.name),
                            "timestamp": data.get("timestamp"),
                            "city": data.get("survey", {}).get("city"),
                            "occupation": data.get("survey", {}).get("occupation"),
                        }
                    )

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "personas": personas,
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
