# config/settings.py
# Farm Management AI Agent v2 — Hierarchical Multi-Agent + RAG

import os
from dataclasses import dataclass, field
from typing import List

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@dataclass
class FarmConfig:
    # LLM
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    model_name:     str  = "gpt-4o"
    temperature:    float = 0.1

    # Farm identity
    farm_name:       str   = "AgriSense Farm"
    farm_location:   str   = "Central Valley, CA"
    farm_area_acres: float = 150.0
    crops:           List[str] = field(default_factory=lambda: ["wheat", "corn", "soybeans"])

    # Thresholds
    critical_moisture_low:  float = 20.0
    critical_moisture_high: float = 80.0
    frost_warning_c:        float = 2.0
    heat_stress_c:          float = 35.0
    planning_horizon_days:  int   = 14

    disclaimer: str = (
        "AI-generated farm recommendations. Always verify with your agronomist "
        "before executing high-cost or irreversible field operations."
    )


farm_config = FarmConfig()
