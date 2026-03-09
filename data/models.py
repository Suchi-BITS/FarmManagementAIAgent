# data/models.py
# Shared data models for the farm management system

from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime, date


class WeatherData(BaseModel):
    """Current and forecast weather conditions."""
    timestamp: datetime = Field(default_factory=datetime.now)
    location: str
    temperature_c: float
    humidity_percent: float
    precipitation_mm: float
    wind_speed_kmh: float
    forecast_7day: list[dict] = Field(default_factory=list)
    frost_risk: bool = False
    heat_stress_risk: bool = False
    severe_weather_alert: Optional[str] = None


class SoilSensorData(BaseModel):
    """Data from soil sensors across the farm."""
    timestamp: datetime = Field(default_factory=datetime.now)
    zone_id: str
    crop_type: str
    moisture_percent: float
    temperature_c: float
    ph_level: float
    nitrogen_ppm: float
    phosphorus_ppm: float
    potassium_ppm: float
    organic_matter_percent: float
    compaction_index: float = Field(description="0-100, higher means more compact")


class CropGrowthData(BaseModel):
    """Current growth stage and health metrics for a crop zone."""
    timestamp: datetime = Field(default_factory=datetime.now)
    zone_id: str
    crop_type: str
    growth_stage: Literal[
        "pre_planting", "germination", "seedling",
        "vegetative", "flowering", "grain_fill",
        "maturation", "harvest_ready"
    ]
    days_since_planting: Optional[int] = None
    estimated_days_to_harvest: Optional[int] = None
    canopy_coverage_percent: float
    leaf_area_index: float
    pest_pressure_level: Literal["none", "low", "medium", "high"] = "none"
    disease_risk: Literal["none", "low", "medium", "high"] = "none"
    yield_forecast_kg_per_ha: Optional[float] = None


class IrrigationAction(BaseModel):
    """Irrigation schedule action."""
    zone_id: str
    action: Literal["start", "stop", "adjust", "schedule"]
    water_volume_liters: Optional[float] = None
    duration_minutes: Optional[int] = None
    scheduled_time: Optional[datetime] = None
    priority: Literal["critical", "high", "normal", "low"] = "normal"
    reason: str


class PlantingAction(BaseModel):
    """Planting schedule recommendation."""
    zone_id: str
    crop_type: str
    recommended_date: date
    seed_variety: str
    seeding_rate_kg_per_ha: float
    row_spacing_cm: float
    planting_depth_cm: float
    pre_plant_tasks: list[str] = Field(default_factory=list)
    confidence_score: float = Field(ge=0.0, le=1.0)
    reasoning: str


class HarvestAction(BaseModel):
    """Harvest schedule recommendation."""
    zone_id: str
    crop_type: str
    recommended_harvest_window_start: date
    recommended_harvest_window_end: date
    estimated_yield_tons: float
    moisture_content_percent: float
    harvest_method: str
    equipment_needed: list[str] = Field(default_factory=list)
    weather_window_suitable: bool
    urgency: Literal["immediate", "this_week", "scheduled", "monitor"] = "scheduled"
    reasoning: str


class FarmSchedule(BaseModel):
    """Consolidated farm management schedule."""
    generated_at: datetime = Field(default_factory=datetime.now)
    planning_horizon_days: int
    irrigation_actions: list[IrrigationAction] = Field(default_factory=list)
    planting_actions: list[PlantingAction] = Field(default_factory=list)
    harvest_actions: list[HarvestAction] = Field(default_factory=list)
    alerts: list[str] = Field(default_factory=list)
    overall_farm_status: Literal["critical", "attention_needed", "normal", "optimal"] = "normal"
    summary: str = ""


class AgentState(BaseModel):
    """
    The shared state that flows through the LangGraph agent graph.
    Each agent reads from and writes to this state.
    """
    # Input data collected by monitoring agents
    weather_data: Optional[WeatherData] = None
    soil_data: list[SoilSensorData] = Field(default_factory=list)
    crop_data: list[CropGrowthData] = Field(default_factory=list)

    # Analysis outputs
    weather_analysis: Optional[str] = None
    soil_analysis: Optional[str] = None
    growth_analysis: Optional[str] = None

    # Decisions made by planning agents
    irrigation_plan: list[IrrigationAction] = Field(default_factory=list)
    planting_plan: list[PlantingAction] = Field(default_factory=list)
    harvest_plan: list[HarvestAction] = Field(default_factory=list)

    # Final consolidated schedule
    farm_schedule: Optional[FarmSchedule] = None

    # Control flow
    current_agent: str = "supervisor"
    iteration_count: int = 0
    errors: list[str] = Field(default_factory=list)
    messages: list[dict] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True
