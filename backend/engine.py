# engine.py

"""
PersonaEngine - Core logic for generating synthetic smartphone personas.

This module contains the heuristics and mappings that convert survey responses
into behavioral dimensions, simulation parameters, and daily schedules.

All derivations are grounded in smartphone usage literature.
"""

import random
import math
from datetime import time, datetime, timedelta
from typing import Tuple, List, Optional, Dict, Any, Literal

from click import Option

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
    ActivityType,
    ContextType,
    # Models
    ComprehensiveSurveyInput,
    BehavioralDimensions,
    BehavioralParameters,
    DailySchedule,
    ScheduleSegment,
    PhoneSession,
    LiteratureReference,
)


# ============================================================
# MAPPING TABLES (Survey Options -> Numeric Values)
# ============================================================

# Wake time -> hour
WAKE_TIME_MAP = {
    WakeTime.BEFORE_6: 5,
    WakeTime.SIX_TO_8: 7,
    WakeTime.EIGHT_TO_10: 9,
    WakeTime.TEN_TO_12: 11,
    WakeTime.AFTER_12: 13,
}

# Sleep time -> hour
SLEEP_TIME_MAP = {
    SleepTime.BEFORE_10PM: 21,
    SleepTime.TEN_TO_12: 23,
    SleepTime.TWELVE_TO_2: 1,
    SleepTime.TWO_TO_4: 3,
    SleepTime.AFTER_4AM: 5,
}

# Chronotype self-report -> score (0 = morning, 1 = evening)
CHRONOTYPE_MAP = {
    ChronotypeLabel.DEFINITELY_MORNING: 0.0,
    ChronotypeLabel.MORE_MORNING: 0.25,
    ChronotypeLabel.NEITHER: 0.5,
    ChronotypeLabel.MORE_EVENING: 0.75,
    ChronotypeLabel.DEFINITELY_EVENING: 1.0,
}

# Peak usage time -> hour
PEAK_USAGE_MAP = {
    PeakUsageTime.MORNING: 9,
    PeakUsageTime.AFTERNOON: 14,
    PeakUsageTime.EVENING: 20,
    PeakUsageTime.LATE_NIGHT: 23,
    PeakUsageTime.EVENLY_DISTRIBUTED: 14,
}

# Screen time -> minutes per day
SCREEN_TIME_MAP = {
    ScreenTime.LESS_THAN_1_HOUR: 45,
    ScreenTime.ONE_TO_2_HOURS: 90,
    ScreenTime.TWO_TO_4_HOURS: 180,
    ScreenTime.FOUR_TO_6_HOURS: 300,
    ScreenTime.MORE_THAN_6_HOURS: 420,
}

# Checking frequency -> checks per hour
CHECKING_FREQ_MAP = {
    CheckingFrequency.LESS_THAN_ONCE_PER_HOUR: 0.5,
    CheckingFrequency.ONE_TO_2_PER_HOUR: 1.5,
    CheckingFrequency.THREE_TO_4_PER_HOUR: 3.5,
    CheckingFrequency.FIVE_TO_6_PER_HOUR: 5.5,
    CheckingFrequency.MORE_THAN_6_PER_HOUR: 8.0,
}

# Session type -> mean duration in seconds
SESSION_DURATION_MAP = {
    SessionType.VERY_SHORT_CHECKS: 10,
    SessionType.SHORT_CHECKS: 30,
    SessionType.MEDIUM_SESSIONS: 180,
    SessionType.LONG_SESSIONS: 420,
    SessionType.MIXED: 120,
}

# Glance frequency -> probability
GLANCE_PROB_MAP = {
    GlanceFrequency.RARELY: 0.1,
    GlanceFrequency.SOMETIMES: 0.25,
    GlanceFrequency.OFTEN: 0.4,
    GlanceFrequency.VERY_OFTEN: 0.6,
}

# Work restriction -> multipliers (frequency, duration)
WORK_RESTRICTION_MAP = {
    WorkPhoneRestriction.USE_FREELY: (1.0, 1.0),
    WorkPhoneRestriction.OCCASIONALLY: (0.6, 0.7),
    WorkPhoneRestriction.BRIEFLY_WHEN_NECESSARY: (0.3, 0.4),
    WorkPhoneRestriction.NOT_APPLICABLE: (1.0, 1.0),
}

# Evening session change -> duration multiplier
EVENING_CHANGE_MAP = {
    EveningSessionChange.MUCH_SHORTER: 0.5,
    EveningSessionChange.SOMEWHAT_SHORTER: 0.75,
    EveningSessionChange.ABOUT_SAME: 1.0,
    EveningSessionChange.SOMEWHAT_LONGER: 1.5,
    EveningSessionChange.MUCH_LONGER: 2.0,
}

# Routine structure -> stability score (0-1)
ROUTINE_STABILITY_MAP = {
    RoutineStructure.VERY_STRUCTURED: 0.9,
    RoutineStructure.SOMEWHAT_STRUCTURED: 0.7,
    RoutineStructure.MIXED: 0.5,
    RoutineStructure.SOMEWHAT_UNSTRUCTURED: 0.3,
    RoutineStructure.VERY_UNSTRUCTURED: 0.1,
}

# Commute days -> days per week
COMMUTE_DAYS_MAP = {
    CommuteDays.ZERO: 0,
    CommuteDays.ONE_TWO: 1.5,
    CommuteDays.THREE_FOUR: 3.5,
    CommuteDays.FIVE_PLUS: 5,
}

# Commute time -> minutes per day
COMMUTE_TIME_MAP = {
    CommuteTime.ALMOST_NONE: 5,
    CommuteTime.LESS_THAN_30: 20,
    CommuteTime.THIRTY_TO_60: 45,
    CommuteTime.ONE_TO_TWO_HOURS: 90,
    CommuteTime.MORE_THAN_2_HOURS: 150,
}

# Physical activity days -> days per week
ACTIVITY_DAYS_MAP = {
    PhysicalActivityDays.ZERO: 0,
    PhysicalActivityDays.ONE_TWO: 1.5,
    PhysicalActivityDays.THREE_FOUR: 3.5,
    PhysicalActivityDays.FIVE_SIX: 5.5,
    PhysicalActivityDays.EVERY_DAY: 7,
}

# Exploration style -> novelty score
EXPLORATION_MAP = {
    ExplorationStyle.MOSTLY_FAMILIAR: 0.2,
    ExplorationStyle.SOMETIMES_EXPLORE: 0.5,
    ExplorationStyle.BOTH_EQUALLY: 0.7,
}

# Commute mode -> phone usage multiplier
COMMUTE_MODE_MAP = {
    CommuteMode.DRIVING: 0.3,
    CommuteMode.PUBLIC_TRANSIT: 1.4,
    CommuteMode.WALKING_BIKING: 0.5,
    CommuteMode.WORK_FROM_HOME: 0.0,
    CommuteMode.STAY_HOME: 0.0,
    CommuteMode.MIXED: 0.8,
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================


def get_enum_value(mapping: dict, key, default=None):
    """Get value from mapping using enum key."""
    if key in mapping:
        return mapping[key]
    return default


def clamp(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Clamp value to range."""
    return max(min_val, min(max_val, value))


def normalize(value: float, min_val: float, max_val: float) -> float:
    """Normalize value to 0-1 range."""
    if max_val == min_val:
        return 0.5
    return clamp((value - min_val) / (max_val - min_val))


def time_to_minutes(t: time) -> int:
    """Convert time object to minutes since midnight."""
    return t.hour * 60 + t.minute


def minutes_to_time(minutes: int) -> time:
    """Convert minutes since midnight to time object."""
    minutes = minutes % 1440  # Wrap around midnight
    return time(hour=minutes // 60, minute=minutes % 60)


# ============================================================
# PERSONA ENGINE
# ============================================================


class PersonaEngine:
    """
    Engine for generating synthetic smartphone personas from survey data.

    The generation process:
    1. Extract behavioral dimensions from survey (8 normalized scores)
    2. Derive simulation parameters from dimensions (literature-grounded)
    3. Generate daily schedule with phone sessions (optional)
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the engine.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        if seed is not None:
            random.seed(seed)

        # Literature citations
        self.citations = self._build_citations()

    def _build_citations(self) -> List[LiteratureReference]:
        """Build list of literature references for parameter derivations."""
        return [
            LiteratureReference(
                parameter="sessions_per_day",
                formula="checks_per_hour * waking_hours * (1 - glance_prob * 0.5)",
                sources=[
                    "Dey et al. (2011) - Average 22 sessions/day",
                    "Andrews et al. (2015) - Mean 85 sessions/day for heavy users",
                ],
                notes="High variance across populations; self-report often underestimates",
            ),
            LiteratureReference(
                parameter="mean_session_duration_seconds",
                formula="Based on session_type survey response with intensity adjustment",
                sources=[
                    "Yan et al. (2012) - Mean session 72 seconds",
                    "Ferreira et al. (2014) - Median 30 seconds",
                    "Böhmer et al. (2011) - Average 59.5 seconds",
                ],
                notes="Log-normal distribution recommended for simulation",
            ),
            LiteratureReference(
                parameter="glance_probability",
                formula="Direct mapping from glance_frequency survey response",
                sources=[
                    "Oulasvirta et al. (2012) - 40% of sessions < 15 seconds",
                    "Banovic et al. (2014) - Glance vs. engagement distinction",
                ],
                notes="Glances are checking without meaningful engagement",
            ),
            LiteratureReference(
                parameter="chronotype_score",
                formula="0.3 * wake_score + 0.3 * sleep_score + 0.4 * self_report",
                sources=[
                    "Roenneberg et al. (2003) - MCTQ chronotype measure",
                    "Adan & Almirall (1991) - Morningness-Eveningness Questionnaire",
                ],
                notes="0 = strong morning preference, 1 = strong evening preference",
            ),
            LiteratureReference(
                parameter="total_daily_minutes",
                formula="Direct mapping from daily_screen_time survey response",
                sources=[
                    "eMarketer (2019) - US adults average 3h 43m/day",
                    "App Annie (2021) - Global average 4h 12m/day",
                    "Andone et al. (2016) - Median 2.5h/day in German sample",
                ],
                notes="Self-report tends to underestimate actual usage by 30-50%",
            ),
            LiteratureReference(
                parameter="app_category_weights",
                formula="Base weights boosted by most_used_categories selection",
                sources=[
                    "Böhmer et al. (2011) - App usage category distributions",
                    "Do & Gatica-Perez (2011) - App usage patterns by context",
                ],
                notes="Social and communication typically dominate",
            ),
            LiteratureReference(
                parameter="checking_burst_probability",
                formula="0.1 + 0.3 * usage_intensity + 0.2 * attentional_granularity",
                sources=[
                    "Oulasvirta et al. (2012) - Checking habits and bursts",
                    "Dey et al. (2011) - Session clustering patterns",
                ],
                notes="Bursts = multiple checks within short time window",
            ),
            LiteratureReference(
                parameter="context_multipliers",
                formula="Based on work_restriction and evening_change survey responses",
                sources=[
                    "Church et al. (2015) - Context-aware phone usage",
                    "Do & Gatica-Perez (2011) - Location-based usage patterns",
                ],
                notes="Strong context effects on both frequency and duration",
            ),
        ]

    def get_citations(self) -> List[LiteratureReference]:
        """Return literature citations."""
        return self.citations

    # --------------------------------------------------------
    # DIMENSION EXTRACTION
    # --------------------------------------------------------

    def extract_dimensions(
        self, survey: ComprehensiveSurveyInput
    ) -> BehavioralDimensions:
        """
        Extract 8 behavioral dimensions from survey responses.

        Each dimension is normalized to 0-1 scale.
        """

        # 1. Chronotype Score (0=morning, 1=evening)
        wake_hour = get_enum_value(WAKE_TIME_MAP, survey.wake_time, 7)
        sleep_hour = get_enum_value(SLEEP_TIME_MAP, survey.sleep_time, 23)
        chrono_self = get_enum_value(CHRONOTYPE_MAP, survey.chronotype_self_report, 0.5)

        # Later wake = more evening, later sleep = more evening
        wake_score = normalize(wake_hour, 5, 13)
        # Handle sleep hours after midnight
        adjusted_sleep = sleep_hour if sleep_hour > 12 else sleep_hour + 24
        sleep_score = normalize(adjusted_sleep, 21, 29)

        chronotype_score = clamp(
            0.3 * wake_score + 0.3 * sleep_score + 0.4 * chrono_self
        )

        # 2. Usage Intensity (0=light, 1=heavy)
        screen_minutes = get_enum_value(SCREEN_TIME_MAP, survey.daily_screen_time, 180)
        checks_per_hour = get_enum_value(
            CHECKING_FREQ_MAP, survey.checking_frequency, 3.5
        )

        screen_score = normalize(screen_minutes, 30, 480)
        check_score = normalize(checks_per_hour, 0.5, 8)

        usage_intensity = clamp(0.6 * screen_score + 0.4 * check_score)

        # 3. Attentional Granularity (0=long focused, 1=fragmented)
        session_duration = get_enum_value(
            SESSION_DURATION_MAP, survey.session_type, 120
        )
        glance_prob = get_enum_value(GLANCE_PROB_MAP, survey.glance_frequency, 0.25)

        # Shorter sessions = more fragmented
        duration_score = 1 - normalize(session_duration, 10, 420)

        attentional_granularity = clamp(
            0.6 * duration_score + 0.4 * (glance_prob / 0.6)
        )

        # 4. Contextual Sensitivity (0=same everywhere, 1=context-dependent)
        work_mult = get_enum_value(
            WORK_RESTRICTION_MAP, survey.work_phone_restriction, (1.0, 1.0)
        )
        evening_mult = get_enum_value(
            EVENING_CHANGE_MAP, survey.evening_session_change, 1.0
        )

        # More restriction at work = more sensitive
        work_score = 1 - work_mult[0]
        # More change in evening = more sensitive
        evening_score = min(abs(evening_mult - 1.0) / 1.0, 1.0)

        contextual_sensitivity = clamp(0.5 * work_score + 0.5 * evening_score)

        # 5. Social Orientation (0=consumption, 1=social)
        social_categories = {AppCategory.MESSAGING, AppCategory.SOCIAL_MEDIA}
        consumption_categories = {
            AppCategory.VIDEO_STREAMING,
            AppCategory.MUSIC_AUDIO,
            AppCategory.NEWS_READING,
            AppCategory.GAMES,
        }

        social_count = sum(
            1 for c in survey.most_used_categories if c in social_categories
        )
        consumption_count = sum(
            1 for c in survey.most_used_categories if c in consumption_categories
        )

        total = social_count + consumption_count
        social_orientation = (social_count / total) if total > 0 else 0.5

        # 6. Mobility Diversity (0=stationary, 1=mobile)
        places = survey.places_visited_daily
        commute_days = get_enum_value(COMMUTE_DAYS_MAP, survey.commute_days, 3.5)
        commute_mins = get_enum_value(COMMUTE_TIME_MAP, survey.commute_time, 45)

        places_score = normalize(places, 1, 6)
        commute_days_score = normalize(commute_days, 0, 5)
        commute_time_score = normalize(commute_mins, 5, 150)

        mobility_diversity = clamp(
            0.4 * places_score + 0.3 * commute_days_score + 0.3 * commute_time_score
        )

        # 7. Routine Stability (0=chaotic, 1=predictable)
        routine_stability = get_enum_value(
            ROUTINE_STABILITY_MAP, survey.routine_structure, 0.5
        )

        # 8. Novelty Seeking (0=familiar, 1=exploring)
        novelty_seeking = get_enum_value(EXPLORATION_MAP, survey.exploration_style, 0.5)

        return BehavioralDimensions(
            chronotype_score=round(chronotype_score, 3),
            usage_intensity=round(usage_intensity, 3),
            attentional_granularity=round(attentional_granularity, 3),
            contextual_sensitivity=round(contextual_sensitivity, 3),
            social_orientation=round(social_orientation, 3),
            mobility_diversity=round(mobility_diversity, 3),
            routine_stability=round(routine_stability, 3),
            novelty_seeking=round(novelty_seeking, 3),
        )

    # --------------------------------------------------------
    # PARAMETER DERIVATION
    # --------------------------------------------------------

    def derive_parameters(
        self, survey: ComprehensiveSurveyInput, dimensions: BehavioralDimensions
    ) -> BehavioralParameters:
        """
        Derive concrete simulation parameters from survey and dimensions.

        These parameters are grounded in smartphone usage literature.
        """

        # --- Temporal Distribution ---
        wake_hour = get_enum_value(WAKE_TIME_MAP, survey.wake_time, 7)
        sleep_hour = get_enum_value(SLEEP_TIME_MAP, survey.sleep_time, 23)
        peak_hour = get_enum_value(PEAK_USAGE_MAP, survey.peak_usage_time, 20)

        # Late night probability based on chronotype and sleep time
        late_night_prob = 0.1 + 0.4 * dimensions.chronotype_score
        if sleep_hour <= 4:  # sleeps after midnight
            late_night_prob += 0.3
        late_night_prob = clamp(late_night_prob)

        # --- Usage Volume ---
        screen_minutes = get_enum_value(SCREEN_TIME_MAP, survey.daily_screen_time, 180)
        checks_per_hour = get_enum_value(
            CHECKING_FREQ_MAP, survey.checking_frequency, 3.5
        )

        # Calculate waking hours
        if sleep_hour > wake_hour:
            waking_hours = sleep_hour - wake_hour
        else:
            waking_hours = (24 - wake_hour) + sleep_hour

        # Pickups and sessions
        pickups_per_day = checks_per_hour * waking_hours
        # Not all pickups lead to sessions (some are just glances)
        glance_prob = get_enum_value(GLANCE_PROB_MAP, survey.glance_frequency, 0.25)
        sessions_per_day = pickups_per_day * (1 - glance_prob * 0.5)

        # --- Session Structure ---
        mean_session_duration = get_enum_value(
            SESSION_DURATION_MAP, survey.session_type, 120
        )
        # Add some variation based on usage intensity
        mean_session_duration *= 0.8 + 0.4 * dimensions.usage_intensity

        # Standard deviation (log-normal assumption)
        session_duration_std = mean_session_duration * 0.8

        # --- Checking Behavior ---
        mean_isi = (
            (waking_hours * 60) / sessions_per_day if sessions_per_day > 0 else 30
        )

        # Burst probability - higher for heavy users with fragmented attention
        burst_prob = (
            0.1
            + 0.3 * dimensions.usage_intensity
            + 0.2 * dimensions.attentional_granularity
        )
        burst_prob = clamp(burst_prob, 0, 0.5)

        # --- Context Multipliers ---
        work_mult = get_enum_value(
            WORK_RESTRICTION_MAP, survey.work_phone_restriction, (1.0, 1.0)
        )
        evening_mult = get_enum_value(
            EVENING_CHANGE_MAP, survey.evening_session_change, 1.0
        )

        work_frequency_mult = work_mult[0]
        work_duration_mult = work_mult[1]
        home_evening_duration_mult = evening_mult

        # Commute multiplier
        commute_freq_mult = get_enum_value(COMMUTE_MODE_MAP, survey.commute_mode, 0.8)

        # Physical activity suppression
        activity_days = get_enum_value(
            ACTIVITY_DAYS_MAP, survey.physical_activity_days, 1.5
        )
        active_suppression = 0.2 + 0.1 * (activity_days / 7)

        # --- App Category Weights ---
        weights = {
            "social": 0.10,
            "messaging": 0.10,
            "video": 0.10,
            "music": 0.05,
            "navigation": 0.05,
            "productivity": 0.10,
            "news": 0.05,
            "games": 0.05,
            "shopping": 0.05,
        }

        # Boost weights based on most_used_categories
        category_to_weight = {
            AppCategory.MESSAGING: "messaging",
            AppCategory.SOCIAL_MEDIA: "social",
            AppCategory.MUSIC_AUDIO: "music",
            AppCategory.VIDEO_STREAMING: "video",
            AppCategory.MAPS_NAVIGATION: "navigation",
            AppCategory.SHOPPING: "shopping",
            AppCategory.NEWS_READING: "news",
            AppCategory.PRODUCTIVITY_WORK: "productivity",
            AppCategory.FITNESS_HEALTH: "productivity",
            AppCategory.GAMES: "games",
        }

        for cat in survey.most_used_categories:
            if cat in category_to_weight:
                weights[category_to_weight[cat]] += 0.15

        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: round(v / total_weight, 3) for k, v in weights.items()}

        # --- Mobility ---
        commute_mins = get_enum_value(COMMUTE_TIME_MAP, survey.commute_time, 45)

        # Radius of gyration estimate based on area type
        if survey.area_type == AreaType.URBAN:
            base_radius = 5
        elif survey.area_type == AreaType.SUBURBAN:
            base_radius = 15
        else:  # rural
            base_radius = 25

        radius_of_gyration = base_radius * (0.5 + commute_mins / 60)

        # Location entropy
        location_entropy = 0.3 + 0.5 * dimensions.mobility_diversity

        # --- Validation Anchors ---
        exploit_ratio = 1 - dimensions.novelty_seeking
        fragmentation_index = dimensions.attentional_granularity

        return BehavioralParameters(
            # Temporal
            waking_hour_start=wake_hour,
            waking_hour_end=sleep_hour,
            temporal_peak_hour=peak_hour,
            late_night_probability=round(late_night_prob, 3),
            # Usage volume
            total_daily_minutes=float(screen_minutes),
            sessions_per_day=round(sessions_per_day, 1),
            pickups_per_day=round(pickups_per_day, 1),
            # Session structure
            mean_session_duration_seconds=round(mean_session_duration, 1),
            session_duration_std=round(session_duration_std, 1),
            glance_probability=glance_prob,
            # Checking behavior
            mean_inter_session_interval_minutes=round(mean_isi, 1),
            checking_burst_probability=round(burst_prob, 3),
            # Context multipliers
            work_frequency_multiplier=work_frequency_mult,
            work_duration_multiplier=work_duration_mult,
            home_evening_duration_multiplier=home_evening_duration_mult,
            commute_frequency_multiplier=commute_freq_mult,
            active_suppression_factor=round(active_suppression, 3),
            # App preferences (individual weights)
            weight_social=weights.get("social", 0.1),
            weight_messaging=weights.get("messaging", 0.1),
            weight_video=weights.get("video", 0.1),
            weight_music=weights.get("music", 0.05),
            weight_navigation=weights.get("navigation", 0.05),
            weight_productivity=weights.get("productivity", 0.1),
            weight_news=weights.get("news", 0.05),
            weight_games=weights.get("games", 0.05),
            weight_shopping=weights.get("shopping", 0.05),
            # Mobility
            radius_of_gyration_km=round(radius_of_gyration, 1),
            location_entropy=round(location_entropy, 3),
            # Validation anchors
            expected_exploit_ratio=round(exploit_ratio, 3),
            expected_fragmentation_index=round(fragmentation_index, 3),
        )

    # --------------------------------------------------------
    # MAIN GENERATION METHOD
    # --------------------------------------------------------

    def generate_persona(
        self, survey: ComprehensiveSurveyInput
    ) -> Tuple[BehavioralDimensions, BehavioralParameters]:
        """
        Generate a complete persona from survey input.

        Args:
            survey: Comprehensive survey responses

        Returns:
            Tuple of (BehavioralDimensions, BehavioralParameters)
        """
        # Step 1: Extract behavioral dimensions
        dimensions = self.extract_dimensions(survey)

        # Step 2: Derive simulation parameters
        parameters = self.derive_parameters(survey, dimensions)

        return dimensions, parameters

    # --------------------------------------------------------
    # SCHEDULE GENERATION
    # --------------------------------------------------------

    def generate_schedule(
        self,
        dimensions: BehavioralDimensions,
        parameters: BehavioralParameters,
        survey: ComprehensiveSurveyInput,
        day_type: Optional[str] = None,
        date_str: Optional[str] = None,
    ) -> DailySchedule:
        """
        Generate a 24-hour daily schedule with phone sessions.

        Args:
            dimensions: Behavioral dimensions
            parameters: Simulation parameters
            survey: Original survey (for context)
            day_type: "weekday" or "weekend" (auto-detected from date if None)
            date_str: Date string YYYY-MM-DD (defaults to today)
        """

        def weekday_or_weekend_from_calendar(d: str) -> Literal["weekday", "weekend"]:
            dt = datetime.strptime(d, "%Y-%m-%d")
            return "weekend" if dt.weekday() >= 5 else "weekday"

        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")

        if day_type is None:
            schedule_day_type = weekday_or_weekend_from_calendar(date_str)
        else:
            low = str(day_type).lower().strip()
            if low == "weekend":
                schedule_day_type = "weekend"
            elif low == "weekday":
                schedule_day_type = "weekday"
            else:
                schedule_day_type = weekday_or_weekend_from_calendar(date_str)

        # Build daily segments
        segments = self._build_day_segments(
            parameters=parameters,
            survey=survey,
            dimensions=dimensions,
            day_type=schedule_day_type,
        )

        # Generate phone sessions for each segment
        for segment in segments:
            if segment.phone_accessible:
                sessions = self._generate_segment_sessions(
                    segment=segment,
                    parameters=parameters,
                    dimensions=dimensions,
                )
                segment.sessions = sessions
                segment.expected_sessions = len(sessions)

        # Calculate totals
        total_sessions = sum(len(seg.sessions) for seg in segments)
        total_glances = sum(
            sum(1 for s in seg.sessions if s.is_glance) for seg in segments
        )
        total_screen_seconds = sum(
            sum(s.duration_seconds for s in seg.sessions) for seg in segments
        )

        return DailySchedule(
            date=date_str,
            day_type=schedule_day_type,
            segments=segments,
            total_phone_sessions=total_sessions,
            total_screen_time_minutes=round(total_screen_seconds / 60, 1),
            total_glances=total_glances,
            is_valid=True,
            validation_notes=[],
        )

    def _build_day_segments(
        self,
        parameters: BehavioralParameters,
        survey: ComprehensiveSurveyInput,
        dimensions: BehavioralDimensions,
        day_type: Literal["weekday", "weekend"],
    ) -> List[ScheduleSegment]:
        """Build activity segments for the day."""

        wake_hour = parameters.waking_hour_start
        sleep_hour = parameters.waking_hour_end

        if day_type == "weekday":
            return self._build_weekday_segments(
                wake_hour, sleep_hour, parameters, survey
            )
        else:
            return self._build_weekend_segments(
                wake_hour, sleep_hour, parameters, survey, dimensions
            )

    def _build_weekday_segments(
        self,
        wake_hour: int,
        sleep_hour: int,
        parameters: BehavioralParameters,
        survey: ComprehensiveSurveyInput,
    ) -> List[ScheduleSegment]:
        """Build weekday schedule segments."""
        segments = []

        commute_mins = get_enum_value(COMMUTE_TIME_MAP, survey.commute_time, 45)
        commute_days = get_enum_value(COMMUTE_DAYS_MAP, survey.commute_days, 3.5)
        has_commute = commute_days > 0 and commute_mins > 10

        # Determine commute context based on mode
        commute_context = self._get_commute_context(survey.commute_mode)
        commute_accessible = commute_context != ContextType.COMMUTE_DRIVING

        # Work restriction
        work_restricted = survey.work_phone_restriction in [
            WorkPhoneRestriction.BRIEFLY_WHEN_NECESSARY,
            WorkPhoneRestriction.OCCASIONALLY,
        ]
        work_context = (
            ContextType.WORK_RESTRICTED if work_restricted else ContextType.WORK_FREE
        )

        current_hour = wake_hour

        # 1. Waking up (15-30 min)
        segments.append(
            ScheduleSegment(
                start_time=minutes_to_time(current_hour * 60),
                end_time=minutes_to_time(current_hour * 60 + 20),
                activity=ActivityType.WAKING_UP,
                context=ContextType.HOME_MORNING,
                location_label="home",
                phone_accessible=True,
                duration_multiplier=0.8,
                frequency_multiplier=1.2,
            )
        )

        # 2. Morning routine
        morning_end = current_hour + 1
        segments.append(
            ScheduleSegment(
                start_time=minutes_to_time(current_hour * 60 + 20),
                end_time=minutes_to_time(morning_end * 60),
                activity=ActivityType.MORNING_ROUTINE,
                context=ContextType.HOME_MORNING,
                location_label="home",
                phone_accessible=True,
                duration_multiplier=0.7,
                frequency_multiplier=0.8,
            )
        )
        current_hour = morning_end

        # 3. Morning commute (if applicable)
        if has_commute:
            commute_end_mins = current_hour * 60 + commute_mins // 2
            segments.append(
                ScheduleSegment(
                    start_time=minutes_to_time(current_hour * 60),
                    end_time=minutes_to_time(commute_end_mins),
                    activity=ActivityType.COMMUTING,
                    context=commute_context,
                    location_label="transit",
                    phone_accessible=commute_accessible,
                    duration_multiplier=0.9,
                    frequency_multiplier=parameters.commute_frequency_multiplier,
                )
            )
            current_hour = commute_end_mins // 60 + 1

        # 4. Work morning
        work_start = max(current_hour, 9)
        segments.append(
            ScheduleSegment(
                start_time=minutes_to_time(work_start * 60),
                end_time=minutes_to_time(12 * 60),
                activity=ActivityType.WORKING,
                context=work_context,
                location_label="office",
                phone_accessible=True,
                duration_multiplier=parameters.work_duration_multiplier,
                frequency_multiplier=parameters.work_frequency_multiplier,
            )
        )

        # 5. Lunch break
        segments.append(
            ScheduleSegment(
                start_time=minutes_to_time(12 * 60),
                end_time=minutes_to_time(13 * 60),
                activity=ActivityType.LUNCH_BREAK,
                context=ContextType.WORK_FREE,
                location_label="office",
                phone_accessible=True,
                duration_multiplier=1.0,
                frequency_multiplier=1.2,
            )
        )

        # 6. Work afternoon
        work_end = 17
        segments.append(
            ScheduleSegment(
                start_time=minutes_to_time(13 * 60),
                end_time=minutes_to_time(work_end * 60),
                activity=ActivityType.WORKING,
                context=work_context,
                location_label="office",
                phone_accessible=True,
                duration_multiplier=parameters.work_duration_multiplier,
                frequency_multiplier=parameters.work_frequency_multiplier,
            )
        )

        current_hour = work_end

        # 7. Evening commute
        if has_commute:
            commute_end_mins = current_hour * 60 + commute_mins // 2
            segments.append(
                ScheduleSegment(
                    start_time=minutes_to_time(current_hour * 60),
                    end_time=minutes_to_time(commute_end_mins),
                    activity=ActivityType.COMMUTING,
                    context=commute_context,
                    location_label="transit",
                    phone_accessible=commute_accessible,
                    duration_multiplier=1.0,
                    frequency_multiplier=parameters.commute_frequency_multiplier,
                )
            )
            current_hour = commute_end_mins // 60 + 1

        # 8. Home evening
        evening_start = max(current_hour, 18)
        wind_down_hour = sleep_hour - 1 if sleep_hour > 12 else sleep_hour + 23

        segments.append(
            ScheduleSegment(
                start_time=minutes_to_time(evening_start * 60),
                end_time=minutes_to_time((wind_down_hour % 24) * 60),
                activity=ActivityType.HOME_EVENING,
                context=ContextType.HOME_EVENING,
                location_label="home",
                phone_accessible=True,
                duration_multiplier=parameters.home_evening_duration_multiplier,
                frequency_multiplier=1.3,
            )
        )

        # 9. Winding down
        segments.append(
            ScheduleSegment(
                start_time=minutes_to_time((wind_down_hour % 24) * 60),
                end_time=minutes_to_time((sleep_hour % 24) * 60),
                activity=ActivityType.WINDING_DOWN,
                context=ContextType.HOME_NIGHT,
                location_label="home",
                phone_accessible=True,
                duration_multiplier=0.9,
                frequency_multiplier=0.7,
            )
        )

        return segments

    def _build_weekend_segments(
        self,
        wake_hour: int,
        sleep_hour: int,
        parameters: BehavioralParameters,
        survey: ComprehensiveSurveyInput,
        dimensions: BehavioralDimensions,
    ) -> List[ScheduleSegment]:
        """Build weekend schedule segments."""
        segments = []

        # Wake slightly later on weekends
        actual_wake = wake_hour + 1
        current_hour = actual_wake

        # 1. Waking up
        segments.append(
            ScheduleSegment(
                start_time=minutes_to_time(current_hour * 60),
                end_time=minutes_to_time(current_hour * 60 + 30),
                activity=ActivityType.WAKING_UP,
                context=ContextType.HOME_MORNING,
                location_label="home",
                phone_accessible=True,
                duration_multiplier=1.2,
                frequency_multiplier=1.0,
            )
        )

        # 2. Morning routine (longer on weekends)
        morning_end = current_hour + 2
        segments.append(
            ScheduleSegment(
                start_time=minutes_to_time(current_hour * 60 + 30),
                end_time=minutes_to_time(morning_end * 60),
                activity=ActivityType.MORNING_ROUTINE,
                context=ContextType.HOME_MORNING,
                location_label="home",
                phone_accessible=True,
                duration_multiplier=1.0,
                frequency_multiplier=1.0,
            )
        )
        current_hour = morning_end

        # 3. Exercise (if active person)
        activity_days = get_enum_value(
            ACTIVITY_DAYS_MAP, survey.physical_activity_days, 1.5
        )
        if activity_days >= 2 and random.random() < 0.6:
            segments.append(
                ScheduleSegment(
                    start_time=minutes_to_time(current_hour * 60),
                    end_time=minutes_to_time((current_hour + 1) * 60),
                    activity=ActivityType.EXERCISING,
                    context=ContextType.EXERCISING,
                    location_label="gym",
                    phone_accessible=False,
                    duration_multiplier=0.3,
                    frequency_multiplier=parameters.active_suppression_factor,
                )
            )
            current_hour += 1

        # 4. Leisure / Errands (midday)
        if dimensions.mobility_diversity > 0.5 and random.random() < 0.5:
            # Out and about
            segments.append(
                ScheduleSegment(
                    start_time=minutes_to_time(current_hour * 60),
                    end_time=minutes_to_time(14 * 60),
                    activity=ActivityType.ERRANDS,
                    context=ContextType.PUBLIC_PLACE,
                    location_label="shops",
                    phone_accessible=True,
                    duration_multiplier=0.8,
                    frequency_multiplier=0.9,
                )
            )
        else:
            segments.append(
                ScheduleSegment(
                    start_time=minutes_to_time(current_hour * 60),
                    end_time=minutes_to_time(14 * 60),
                    activity=ActivityType.LEISURE,
                    context=ContextType.HOME_MORNING,
                    location_label="home",
                    phone_accessible=True,
                    duration_multiplier=1.3,
                    frequency_multiplier=1.2,
                )
            )

        # 5. Afternoon leisure
        segments.append(
            ScheduleSegment(
                start_time=minutes_to_time(14 * 60),
                end_time=minutes_to_time(18 * 60),
                activity=ActivityType.LEISURE,
                context=ContextType.HOME_EVENING,
                location_label="home",
                phone_accessible=True,
                duration_multiplier=parameters.home_evening_duration_multiplier,
                frequency_multiplier=1.3,
            )
        )

        # 6. Evening
        wind_down_hour = sleep_hour - 1 if sleep_hour > 12 else sleep_hour + 23
        segments.append(
            ScheduleSegment(
                start_time=minutes_to_time(18 * 60),
                end_time=minutes_to_time((wind_down_hour % 24) * 60),
                activity=ActivityType.HOME_EVENING,
                context=ContextType.HOME_EVENING,
                location_label="home",
                phone_accessible=True,
                duration_multiplier=parameters.home_evening_duration_multiplier,
                frequency_multiplier=1.4,
            )
        )

        # 7. Winding down
        segments.append(
            ScheduleSegment(
                start_time=minutes_to_time((wind_down_hour % 24) * 60),
                end_time=minutes_to_time((sleep_hour % 24) * 60),
                activity=ActivityType.WINDING_DOWN,
                context=ContextType.HOME_NIGHT,
                location_label="home",
                phone_accessible=True,
                duration_multiplier=1.0,
                frequency_multiplier=0.8,
            )
        )

        return segments

    def _get_commute_context(self, mode: CommuteMode) -> ContextType:
        """Map commute mode to context type."""
        if mode == CommuteMode.DRIVING:
            return ContextType.COMMUTE_DRIVING
        elif mode == CommuteMode.PUBLIC_TRANSIT:
            return ContextType.COMMUTE_TRANSIT
        elif mode == CommuteMode.WALKING_BIKING:
            return ContextType.COMMUTE_WALKING
        else:
            return ContextType.COMMUTE_TRANSIT

    def _generate_segment_sessions(
        self,
        segment: ScheduleSegment,
        parameters: BehavioralParameters,
        dimensions: BehavioralDimensions,
    ) -> List[PhoneSession]:
        """Generate phone sessions for a schedule segment."""
        sessions = []

        # Calculate segment duration in seconds
        start_mins = time_to_minutes(segment.start_time)
        end_mins = time_to_minutes(segment.end_time)

        if end_mins <= start_mins:
            end_mins += 1440  # Handle midnight crossing

        duration_mins = end_mins - start_mins
        duration_secs = duration_mins * 60

        if duration_mins <= 0:
            return sessions

        # Calculate expected sessions for this segment
        hours = duration_mins / 60
        base_sessions_per_hour = (
            parameters.sessions_per_day / 16
        )  # Assume 16 waking hrs
        expected_sessions = (
            base_sessions_per_hour * hours * segment.frequency_multiplier
        )

        # Add randomness
        num_sessions = max(
            0, int(random.gauss(expected_sessions, expected_sessions * 0.3))
        )
        num_sessions = min(num_sessions, duration_mins // 2)  # At least 2 min apart

        # Generate session offsets within segment
        offsets = sorted(
            [random.uniform(0, duration_secs - 30) for _ in range(num_sessions)]
        )

        # Build app weights dict for selection
        app_weights = {
            AppCategory.SOCIAL_MEDIA: parameters.weight_social,
            AppCategory.MESSAGING: parameters.weight_messaging,
            AppCategory.VIDEO_STREAMING: parameters.weight_video,
            AppCategory.MUSIC_AUDIO: parameters.weight_music,
            AppCategory.MAPS_NAVIGATION: parameters.weight_navigation,
            AppCategory.PRODUCTIVITY_WORK: parameters.weight_productivity,
            AppCategory.NEWS_READING: parameters.weight_news,
            AppCategory.GAMES: parameters.weight_games,
            AppCategory.SHOPPING: parameters.weight_shopping,
        }

        for offset in offsets:
            is_glance = random.random() < parameters.glance_probability

            if is_glance:
                duration = random.uniform(3, 15)
            else:
                mean_dur = (
                    parameters.mean_session_duration_seconds
                    * segment.duration_multiplier
                )
                std_dur = parameters.session_duration_std
                duration = max(15, random.gauss(mean_dur, std_dur))
                duration = min(duration, 1800)  # Cap at 30 min

            app_category = self._select_app_category(app_weights, segment.context)

            sessions.append(
                PhoneSession(
                    start_offset_seconds=round(offset, 1),
                    duration_seconds=round(duration, 1),
                    is_glance=is_glance,
                    app_category=app_category,
                )
            )

        return sessions

    def _select_app_category(
        self,
        weights: Dict[AppCategory, float],
        context: ContextType,
    ) -> AppCategory:
        """Select app category based on weights and context."""
        adjusted = dict(weights)

        # Context adjustments
        if context in [ContextType.WORK_RESTRICTED, ContextType.WORK_FREE]:
            adjusted[AppCategory.PRODUCTIVITY_WORK] *= 2.0
            adjusted[AppCategory.GAMES] *= 0.3
        elif context in [ContextType.COMMUTE_TRANSIT, ContextType.COMMUTE_WALKING]:
            adjusted[AppCategory.MUSIC_AUDIO] *= 2.0
            adjusted[AppCategory.MAPS_NAVIGATION] *= 2.0
        elif context == ContextType.HOME_EVENING:
            adjusted[AppCategory.VIDEO_STREAMING] *= 1.5
            adjusted[AppCategory.SOCIAL_MEDIA] *= 1.3

        # Normalize and select
        total = sum(adjusted.values())
        if total == 0:
            return AppCategory.SOCIAL_MEDIA

        r = random.random() * total
        cumulative = 0
        for cat, w in adjusted.items():
            cumulative += w
            if r <= cumulative:
                return cat

        return AppCategory.SOCIAL_MEDIA
