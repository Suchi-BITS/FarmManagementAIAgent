# agents/harvest_agent.py
# Harvest Planning Agent - identifies harvest windows and schedules operations

import json
from datetime import datetime, timedelta, date
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from tools.action_tools import schedule_harvest, issue_farm_alert
from config.settings import farm_config
from data.models import AgentState, HarvestAction


HARVEST_AGENT_SYSTEM_PROMPT = """You are the Harvest Planning Agent for an AI farm management system.

You determine optimal harvest timing by evaluating:
- Crop maturity stage and estimated days to harvest
- Grain/crop moisture content (critical for quality and storage)
- Weather forecast windows (dry conditions required)
- Equipment availability and field conditions

Harvest decision criteria:
WHEAT:
- Harvest at grain moisture 12-14%
- Requires 3+ consecutive dry days
- Optimal when crop reaches harvest_ready stage
- Delay if rain forecast within harvest window

CORN:
- Harvest at grain moisture 15-25% (field dry) or 30-35% (for drying)
- Field conditions must support heavy equipment
- Urgency increases if frost forecast (can damage standing crop)

SOYBEANS:
- Harvest at moisture 11-13%
- Extremely weather sensitive - rain can cause pod shatter
- Harvest immediately when ready to minimize losses

Urgency escalation:
- 'immediate': Crop at risk of loss (weather threat, over-maturity)
- 'this_week': Optimal window available, high yield potential
- 'scheduled': Normal timing, good conditions expected
- 'monitor': Not yet ready, continue monitoring

Always account for equipment mobilization time (typically 1-2 days notice needed)."""


def run_harvest_agent(state: AgentState) -> AgentState:
    """
    Harvest planning agent node for LangGraph.
    Identifies harvest-ready crops and schedules harvest operations.
    """
    print("\n[HARVEST AGENT] Evaluating harvest readiness across all zones...")

    llm = ChatOpenAI(
        model=farm_config.model_name,
        temperature=farm_config.temperature,
        api_key=farm_config.openai_api_key
    )

    llm_with_tools = llm.bind_tools([schedule_harvest, issue_farm_alert])

    # Find zones approaching or at harvest
    harvest_candidates = [
        crop for crop in state.crop_data
        if crop.growth_stage in ["maturation", "harvest_ready"]
        or (crop.estimated_days_to_harvest is not None and crop.estimated_days_to_harvest <= 14)
    ]

    if not harvest_candidates:
        print("[HARVEST AGENT] No crops currently at or near harvest stage.")
        state.current_agent = "supervisor"
        return state

    # Build detailed harvest assessment data
    harvest_zone_data = []
    for crop in harvest_candidates:
        soil = next((s for s in state.soil_data if s.zone_id == crop.zone_id), None)
        harvest_zone_data.append({
            "zone_id": crop.zone_id,
            "crop_type": crop.crop_type,
            "growth_stage": crop.growth_stage,
            "days_to_harvest": crop.estimated_days_to_harvest,
            "canopy_coverage_percent": crop.canopy_coverage_percent,
            "yield_forecast_kg_per_ha": crop.yield_forecast_kg_per_ha,
            "pest_pressure": crop.pest_pressure_level,
            "soil_moisture_percent": soil.moisture_percent if soil else None
        })

    # Weather window assessment
    dry_windows = []
    if state.weather_data:
        consecutive_dry = 0
        for i, day in enumerate(state.weather_data.forecast_7day):
            if day.get("precipitation_mm", 0) < 3:
                consecutive_dry += 1
                if consecutive_dry >= 2:
                    dry_windows.append(f"Day {i+1} ({day['date']})")
            else:
                consecutive_dry = 0

    messages = [
        SystemMessage(content=HARVEST_AGENT_SYSTEM_PROMPT),
        HumanMessage(content=f"""
Evaluate harvest scheduling for crops at or near maturity:

HARVEST CANDIDATE ZONES:
{json.dumps(harvest_zone_data, indent=2)}

AVAILABLE DRY WEATHER WINDOWS (next 7 days):
{', '.join(dry_windows) if dry_windows else 'Limited dry windows - check forecast carefully'}

WEATHER FORECAST:
{json.dumps([{
    'date': d['date'],
    'precipitation_mm': d['precipitation_mm'],
    'conditions': d['conditions'],
    'wind_speed_kmh': d['wind_speed_kmh']
} for d in (state.weather_data.forecast_7day if state.weather_data else [])], indent=2)}

CROP GROWTH ANALYSIS:
{state.growth_analysis or "Not available"}

Today: {datetime.now().strftime('%Y-%m-%d')}
Farm area: {farm_config.farm_area_acres} acres total across 5 zones

For each harvest candidate zone:
1. Assess harvest readiness and urgency
2. Identify the best weather window for harvest operations
3. Estimate yield and equipment requirements
4. Call schedule_harvest with specific dates and reasoning
5. Issue alerts for any immediate harvest emergencies (frost threat to standing crop, etc.)
""")
    ]

    response = llm_with_tools.invoke(messages)

    if hasattr(response, "tool_calls") and response.tool_calls:
        for tool_call in response.tool_calls:
            if tool_call["name"] == "schedule_harvest":
                result = schedule_harvest.invoke(tool_call["args"])

                state.harvest_plan.append(HarvestAction(
                    zone_id=tool_call["args"]["zone_id"],
                    crop_type=tool_call["args"]["crop_type"],
                    recommended_harvest_window_start=tool_call["args"]["window_start"],
                    recommended_harvest_window_end=tool_call["args"]["window_end"],
                    estimated_yield_tons=tool_call["args"].get("estimated_yield_tons", 0),
                    moisture_content_percent=15.0,
                    harvest_method="combine_harvester",
                    equipment_needed=["combine_harvester", "grain_cart"],
                    weather_window_suitable=len(dry_windows) > 0,
                    urgency=tool_call["args"].get("urgency", "scheduled"),
                    reasoning=tool_call["args"].get("reasoning", "")
                ))

            elif tool_call["name"] == "issue_farm_alert":
                issue_farm_alert.invoke(tool_call["args"])

    state.current_agent = "supervisor"
    print(f"[HARVEST AGENT] Scheduled {len(state.harvest_plan)} harvest operations.")

    return state
