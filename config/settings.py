# config/settings.py
# Farm Management AI System Configuration

from pydantic import BaseModel, Field
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()


class FarmConfig(BaseModel):
    """Core configuration for the farm management system."""

    # LLM settings
    openai_api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    model_name: str = "gpt-4o"
    temperature: float = 0.1
    max_tokens: int = 2048

    # Farm identity
    farm_name: str = "AgriSense Farm"
    farm_location: str = "Central Valley, CA"
    farm_area_acres: float = 150.0

    # Crop types being managed
    crops: list[str] = ["wheat", "corn", "soybeans"]

    # Monitoring intervals (in seconds, for simulation)
    weather_check_interval: int = 3600       # 1 hour
    soil_check_interval: int = 1800          # 30 minutes
    growth_check_interval: int = 86400       # 24 hours

    # Decision thresholds
    critical_soil_moisture_low: float = 20.0   # percent
    critical_soil_moisture_high: float = 80.0  # percent
    frost_warning_temp_c: float = 2.0
    heat_stress_temp_c: float = 35.0

    # Scheduling horizon (days)
    planning_horizon_days: int = 14


farm_config = FarmConfig()
