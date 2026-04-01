# models.py - Comprehensive survey-aligned models

from datetime import time, datetime
from typing import List, Optional, Literal, Dict, Any, Tuple
from pydantic import BaseModel, Field
from enum import Enum


# ============================================================
# ENUMS (Match Survey Options Exactly)
# ============================================================


class AgeRange(str, Enum):
    AGE_18_24 = "18-24"
    AGE_25_34 = "25-34"
    AGE_35_44 = "35-44"
    AGE_45_54 = "45-54"
    AGE_55_64 = "55-64"
    AGE_65_PLUS = "65+"


class AreaType(str, Enum):
    URBAN = "urban"
    SUBURBAN = "suburban"
    RURAL = "rural"


class WakeTime(str, Enum):
    BEFORE_6 = "before_6am"
    SIX_TO_8 = "6am-8am"
    EIGHT_TO_10 = "8am-10am"
    TEN_TO_12 = "10am-12pm"
    AFTER_12 = "after_12pm"


class SleepTime(str, Enum):
    BEFORE_10PM = "before_10pm"
    TEN_TO_12 = "10pm-12am"
    TWELVE_TO_2 = "12am-2am"
    TWO_TO_4 = "2am-4am"
    AFTER_4AM = "after_4am"


class ChronotypeLabel(str, Enum):
    DEFINITELY_MORNING = "definitely_morning"
    MORE_MORNING = "more_morning"
    NEITHER = "neither"
    MORE_EVENING = "more_evening"
    DEFINITELY_EVENING = "definitely_evening"


class PeakUsageTime(str, Enum):
    MORNING = "morning"
    AFTERNOON = "afternoon"
    EVENING = "evening"
    LATE_NIGHT = "late_night"
    EVENLY_DISTRIBUTED = "evenly_distributed"


class RoutineStructure(str, Enum):
    VERY_STRUCTURED = "very_structured"
    SOMEWHAT_STRUCTURED = "somewhat_structured"
    MIXED = "mixed"
    SOMEWHAT_UNSTRUCTURED = "somewhat_unstructured"
    VERY_UNSTRUCTURED = "very_unstructured"


class CommuteDays(str, Enum):
    ZERO = "0"
    ONE_TWO = "1-2"
    THREE_FOUR = "3-4"
    FIVE_PLUS = "5+"


class CommuteMode(str, Enum):
    DRIVING = "driving"
    PUBLIC_TRANSIT = "public_transit"
    WALKING_BIKING = "walking_biking"
    MIXED = "mixed"
    STAY_HOME = "stay_home"
    WORK_FROM_HOME = "work_from_home"


class PhysicalActivityDays(str, Enum):
    ZERO = "0"
    ONE_TWO = "1-2"
    THREE_FOUR = "3-4"
    FIVE_SIX = "5-6"
    EVERY_DAY = "every_day"


class CommuteTime(str, Enum):
    ALMOST_NONE = "almost_none"
    LESS_THAN_30 = "less_than_30min"
    THIRTY_TO_60 = "30-60min"
    ONE_TO_TWO_HOURS = "1-2hours"
    MORE_THAN_2_HOURS = "more_than_2hours"


class ScreenTime(str, Enum):
    LESS_THAN_1_HOUR = "less_than_1hour"
    ONE_TO_2_HOURS = "1-2hours"
    TWO_TO_4_HOURS = "2-4hours"
    FOUR_TO_6_HOURS = "4-6hours"
    MORE_THAN_6_HOURS = "more_than_6hours"


class CheckingFrequency(str, Enum):
    LESS_THAN_ONCE_PER_HOUR = "less_than_1_per_hour"
    ONE_TO_2_PER_HOUR = "1-2_per_hour"
    THREE_TO_4_PER_HOUR = "3-4_per_hour"
    FIVE_TO_6_PER_HOUR = "5-6_per_hour"
    MORE_THAN_6_PER_HOUR = "more_than_6_per_hour"


class SessionType(str, Enum):
    VERY_SHORT_CHECKS = "very_short_checks"  # few seconds
    SHORT_CHECKS = "short_checks"  # under 1 minute
    MEDIUM_SESSIONS = "medium_sessions"  # 1-5 minutes
    LONG_SESSIONS = "long_sessions"  # more than 5 minutes
    MIXED = "mixed"


class GlanceFrequency(str, Enum):
    RARELY = "rarely"
    SOMETIMES = "sometimes"
    OFTEN = "often"
    VERY_OFTEN = "very_often"


class WorkPhoneRestriction(str, Enum):
    USE_FREELY = "use_freely"
    OCCASIONALLY = "occasionally"
    BRIEFLY_WHEN_NECESSARY = "briefly_when_necessary"
    NOT_APPLICABLE = "not_applicable"


class EveningSessionChange(str, Enum):
    MUCH_SHORTER = "much_shorter"
    SOMEWHAT_SHORTER = "somewhat_shorter"
    ABOUT_SAME = "about_same"
    SOMEWHAT_LONGER = "somewhat_longer"
    MUCH_LONGER = "much_longer"


class AppCategory(str, Enum):
    MESSAGING = "messaging"
    SOCIAL_MEDIA = "social_media"
    MUSIC_AUDIO = "music_audio"
    VIDEO_STREAMING = "video_streaming"
    MAPS_NAVIGATION = "maps_navigation"
    SHOPPING = "shopping"
    NEWS_READING = "news_reading"
    PRODUCTIVITY_WORK = "productivity_work"
    FITNESS_HEALTH = "fitness_health"
    GAMES = "games"
    OTHER = "other"


class UsageReason(str, Enum):
    MESSAGING_TALKING = "messaging_talking"
    SOCIAL_MEDIA = "social_media"
    WATCHING_VIDEOS = "watching_videos"
    LISTENING_MUSIC_PODCASTS = "listening_music_podcasts"
    READING_NEWS = "reading_news"
    NAVIGATION_MAPS = "navigation_maps"
    SHOPPING = "shopping"
    WORK_PRODUCTIVITY = "work_productivity"
    GAMING = "gaming"
    OTHER = "other"


class ExplorationStyle(str, Enum):
    MOSTLY_FAMILIAR = "mostly_familiar"
    SOMETIMES_EXPLORE = "sometimes_explore"
    BOTH_EQUALLY = "both_equally"


# ============================================================
# SURVEY INPUT MODEL (Comprehensive)
# ============================================================


class ComprehensiveSurveyInput(BaseModel):
    """
    Complete survey input matching the Prolific questionnaire.
    All fields map directly to survey questions Q1-Q27.
    """

    # --- Section 1: Demographics (Q1-Q4) ---
    age_range: AgeRange = Field(description="Q1: Age range")
    city: str = Field(description="Q2: City of residence")
    occupation: str = Field(description="Q3: Current occupation or primary role")
    area_type: AreaType = Field(description="Q4: Urban/suburban/rural")

    # --- Section 2: Sleep & Chronotype (Q5-Q8) ---
    wake_time: WakeTime = Field(description="Q5: Typical weekday wake time")
    sleep_time: SleepTime = Field(description="Q6: Typical weekday sleep time")
    chronotype_self_report: ChronotypeLabel = Field(
        description="Q7: Self-reported chronotype"
    )
    peak_usage_time: PeakUsageTime = Field(description="Q8: When phone is used most")

    # --- Section 3: Daily Structure (Q9-Q14) ---
    routine_structure: RoutineStructure = Field(
        description="Q9: How structured is daily routine"
    )
    places_visited_daily: int = Field(
        ge=1, le=6, description="Q10: Number of places visited daily (1-6+)"
    )
    commute_days: CommuteDays = Field(description="Q11: Days per week commuting")
    commute_mode: CommuteMode = Field(description="Q12: Primary commute mode")
    physical_activity_days: PhysicalActivityDays = Field(
        description="Q13: Days per week of exercise"
    )
    commute_time: CommuteTime = Field(
        description="Q14: Daily time spent moving between places"
    )

    # --- Section 4: Phone Usage Patterns (Q15-Q20) ---
    daily_screen_time: ScreenTime = Field(description="Q15: Total daily screen time")
    checking_frequency: CheckingFrequency = Field(
        description="Q16: How often phone is checked"
    )
    session_type: SessionType = Field(description="Q17: Typical session length pattern")
    glance_frequency: GlanceFrequency = Field(description="Q18: Quick check frequency")
    work_phone_restriction: WorkPhoneRestriction = Field(
        description="Q19: Phone restriction at work"
    )
    evening_session_change: EveningSessionChange = Field(
        description="Q20: Evening vs daytime sessions"
    )

    # --- Section 5: App & Content Preferences (Q21-Q25) ---
    evening_activities_increase: List[AppCategory] = Field(
        max_length=3, description="Q21: Activities that increase in evening (up to 3)"
    )
    usage_reasons: List[UsageReason] = Field(
        max_length=3, description="Q22: Most common reasons for phone use (up to 3)"
    )
    commute_activities: List[AppCategory] = Field(
        max_length=3, description="Q23: Phone activities during commute (up to 3)"
    )
    most_used_categories: List[AppCategory] = Field(
        max_length=5, description="Q24: Most used app categories (up to 5)"
    )
    exploration_style: ExplorationStyle = Field(
        description="Q25: Familiar vs exploring new content"
    )

    # --- Section 6: Optional Details (Q26-Q27) ---
    top_apps: Optional[str] = Field(
        default=None, description="Q26: Top 3 apps (free text)"
    )
    important_habit: Optional[str] = Field(
        default=None, description="Q27: Important habit to capture"
    )


# ============================================================
# BEHAVIORAL DIMENSIONS (Intermediate Layer)
# ============================================================


class BehavioralDimensions(BaseModel):
    """
    Extracted behavioral dimensions from survey input.
    Each dimension is normalized to 0-1 scale.
    These serve as inputs to parameter derivation.
    """

    # Temporal
    chronotype_score: float = Field(
        ge=0,
        le=1,
        description="0=extreme morning, 1=extreme evening. Derived from wake/sleep times and self-report.",
    )

    # Usage intensity
    usage_intensity: float = Field(
        ge=0,
        le=1,
        description="0=very light, 1=very heavy. Derived from screen time and checking frequency.",
    )

    # Session structure
    attentional_granularity: float = Field(
        ge=0,
        le=1,
        description="0=long focused sessions, 1=highly fragmented quick checks. Derived from session type and glance frequency.",
    )

    # Context sensitivity
    contextual_sensitivity: float = Field(
        ge=0,
        le=1,
        description="0=same usage everywhere, 1=highly context-dependent. Derived from work restriction and evening change.",
    )

    # Content preferences
    social_orientation: float = Field(
        ge=0,
        le=1,
        description="0=consumption-focused, 1=social/communication-focused. Derived from app categories and usage reasons.",
    )

    # Mobility
    mobility_diversity: float = Field(
        ge=0,
        le=1,
        description="0=mostly stationary, 1=highly mobile. Derived from places visited, commute frequency, commute time.",
    )

    # Routine
    routine_stability: float = Field(
        ge=0,
        le=1,
        description="0=chaotic/variable, 1=highly predictable. Derived from routine structure.",
    )

    # Exploration
    novelty_seeking: float = Field(
        ge=0,
        le=1,
        description="0=always familiar content, 1=always exploring. Derived from exploration style.",
    )


# ============================================================
# BEHAVIORAL PARAMETERS (Literature-Grounded)
# ============================================================


class LiteratureReference(BaseModel):
    """Citation for a parameter derivation."""

    parameter: str
    formula: str
    sources: List[str]
    notes: Optional[str] = None


class BehavioralParameters(BaseModel):
    """
    Concrete behavioral parameters derived from dimensions.
    Each parameter has literature grounding.
    """

    # --- Temporal Distribution ---
    waking_hour_start: int = Field(ge=0, le=23, description="Hour of typical wake time")
    waking_hour_end: int = Field(ge=0, le=23, description="Hour of typical sleep time")
    temporal_peak_hour: int = Field(ge=0, le=23, description="Hour of peak phone usage")
    late_night_probability: float = Field(
        ge=0, le=1, description="P(usage after midnight)"
    )

    # --- Usage Volume ---
    total_daily_minutes: float = Field(
        ge=0, description="Expected total screen time (minutes)"
    )
    pickups_per_day: float = Field(ge=0, description="Expected number of phone pickups")
    sessions_per_day: float = Field(
        ge=0, description="Expected number of unlock sessions"
    )

    # --- Session Structure ---
    mean_session_duration_seconds: float = Field(
        ge=0, description="Mean session length (seconds)"
    )
    session_duration_std: float = Field(
        ge=0, description="Session duration standard deviation"
    )
    glance_probability: float = Field(
        ge=0, le=1, description="P(session is a glance < 15s)"
    )

    # --- Checking Behavior ---
    mean_inter_session_interval_minutes: float = Field(
        ge=0, description="Mean time between sessions"
    )
    checking_burst_probability: float = Field(
        ge=0, le=1, description="P(multiple checks within 5 min)"
    )

    # --- Context Multipliers ---
    work_duration_multiplier: float = Field(
        ge=0, description="Session duration multiplier at work"
    )
    work_frequency_multiplier: float = Field(
        ge=0, description="Checking frequency multiplier at work"
    )
    home_evening_duration_multiplier: float = Field(
        ge=0, description="Session duration multiplier at home evening"
    )
    commute_frequency_multiplier: float = Field(
        ge=0, description="Checking frequency during commute"
    )
    active_suppression_factor: float = Field(
        ge=0, le=1, description="Usage suppression during physical activity"
    )

    # --- App Category Weights ---
    weight_social: float = Field(ge=0, le=1)
    weight_messaging: float = Field(ge=0, le=1)
    weight_video: float = Field(ge=0, le=1)
    weight_music: float = Field(ge=0, le=1)
    weight_navigation: float = Field(ge=0, le=1)
    weight_productivity: float = Field(ge=0, le=1)
    weight_news: float = Field(ge=0, le=1)
    weight_games: float = Field(ge=0, le=1)
    weight_shopping: float = Field(ge=0, le=1)

    # --- Mobility ---
    radius_of_gyration_km: float = Field(ge=0, description="Expected mobility radius")
    location_entropy: float = Field(ge=0, description="Diversity of locations visited")

    # --- Validation Anchors ---
    expected_exploit_ratio: float = Field(
        ge=0, le=1, description="Fraction of time on familiar apps"
    )
    expected_fragmentation_index: float = Field(
        ge=0, le=1, description="Session fragmentation level"
    )


# ============================================================
# SCHEDULE MODELS
# ============================================================


class ActivityType(str, Enum):
    SLEEPING = "sleeping"
    WAKING_UP = "waking_up"
    MORNING_ROUTINE = "morning_routine"
    COMMUTING = "commuting"
    WORKING = "working"
    LUNCH_BREAK = "lunch_break"
    EXERCISING = "exercising"
    ERRANDS = "errands"
    HOME_EVENING = "home_evening"
    LEISURE = "leisure"
    WINDING_DOWN = "winding_down"


class ContextType(str, Enum):
    HOME_MORNING = "home_morning"
    HOME_EVENING = "home_evening"
    HOME_NIGHT = "home_night"
    WORK_RESTRICTED = "work_restricted"
    WORK_FREE = "work_free"
    COMMUTE_TRANSIT = "commute_transit"
    COMMUTE_DRIVING = "commute_driving"
    COMMUTE_WALKING = "commute_walking"
    PUBLIC_PLACE = "public_place"
    EXERCISING = "exercising"
    SOCIAL_SETTING = "social_setting"


class PhoneSession(BaseModel):
    """A single phone usage session within a schedule segment."""

    start_offset_seconds: float = Field(description="Seconds from segment start")
    duration_seconds: float = Field(description="Session duration")
    is_glance: bool = Field(description="Whether this is a quick glance (<15s)")
    app_category: AppCategory = Field(description="Primary app category used")

    # Optional details
    specific_app: Optional[str] = None
    action_type: Optional[str] = None  # "scroll", "message", "watch", etc.


class ScheduleSegment(BaseModel):
    """A time segment in the daily schedule."""

    start_time: time = Field(description="Segment start time")
    end_time: time = Field(description="Segment end time")
    activity: ActivityType = Field(description="What the person is doing")
    context: ContextType = Field(description="Phone usage context")
    location_label: str = Field(description="e.g., 'home', 'office', 'gym'")
    location_coords: Optional[Tuple[float, float]] = Field(
        default=None, description="(lat, lon)"
    )

    # Phone behavior in this segment
    phone_accessible: bool = Field(default=True, description="Can person use phone?")
    expected_sessions: int = Field(default=0, description="Expected number of sessions")
    sessions: List[PhoneSession] = Field(default_factory=list)

    # Context multipliers applied
    duration_multiplier: float = Field(default=1.0)
    frequency_multiplier: float = Field(default=1.0)


class DailySchedule(BaseModel):
    """Complete 24-hour schedule with all segments and phone sessions."""

    date: str = Field(description="Date in YYYY-MM-DD format")
    day_type: Literal["weekday", "weekend"] = Field(description="Type of day")
    segments: List[ScheduleSegment] = Field(
        description="Ordered list of daily segments"
    )

    # Summary statistics
    total_phone_sessions: int = 0
    total_screen_time_minutes: float = 0
    total_glances: int = 0

    # Validation
    is_valid: bool = True
    validation_notes: List[str] = Field(default_factory=list)


# ============================================================
# API MODELS
# ============================================================


class GenerationRequest(BaseModel):
    """Request to generate a persona from survey input."""

    survey: ComprehensiveSurveyInput

    # Generation options
    generate_schedule: bool = True
    schedule_date: Optional[str] = None  # defaults to today
    day_type: Optional[Literal["weekday", "weekend"]] = None  # auto-detect from date

    # Execution options
    execute_on_device: bool = False
    max_execution_steps: int = 10


class GenerationResponse(BaseModel):
    """Response with generated persona and all artifacts."""

    # IDs
    persona_id: str
    generation_timestamp: str

    # Computed artifacts
    dimensions: BehavioralDimensions
    parameters: BehavioralParameters
    schedule: Optional[DailySchedule] = None

    # File paths
    save_directory: str
    persona_json_path: str
    traces_tar_path: Optional[str] = None

    # Literature grounding (for transparency)
    parameter_citations: List[LiteratureReference] = Field(default_factory=list)

    # Execution results (if run)
    execution_metrics: Optional[Dict[str, Any]] = None
