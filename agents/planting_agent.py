# agents/planting_agent.py
# Planting Planning Agent - optimizes planting schedules based on conditions

import json
from datetime import datetime, timedelta, date
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from tools.action_tools import schedule_planting, issue_farm_alert
from config.settings import farm_config
from data.models import AgentState, PlantingAction


PLANTING_AGENT_SYSTEM_PROMPT = """You are the Planting Planning Agent for an AI farm management system.

You determine optimal planting schedules by analyzing:
- Soil temperature, moisture, and structure readiness
- Weather forecasts for post-planting establishment conditions
- Current crop rotation and zone availability
- Seasonal windows for each crop type

Optimal planting conditions:
WHEAT:
- Soil temperature: 10-24°C at seeding depth
- Soil moisture: 40-60% (adequate but not waterlogged)
- Best window: Fall (winter wheat) or early spring
- Avoid: Waterlogged soils, frost forecast within 5 days

CORN:
- Soil temperature: minimum 10°C, optimal 16-18°C
- Soil moisture: 50-70%
- Best window: Spring after last frost
- Avoid: Soil temperature below 10°C, compaction index > 60

SOYBEANS:
- Soil temperature: minimum 10°C, optimal 15-20°C
- Soil moisture: 45-65%
- Best window: Late spring, after stable warm period
- Avoid: Wet, cold soils; compaction reduces nodulation

Only recommend planting for zones in 'pre_planting' growth stage.
Assign realistic confidence scores based on how well conditions match optimal ranges.
Provide variety recommendations based on current season timing."""


def run_planting_agent(state: AgentState) -> AgentState:
    """
    Planting planning agent node for LangGraph.
    Creates planting schedules for zones ready for planting.
    """
    print("\n[PLANTING AGENT] Evaluating planting opportunities...")

    llm = ChatOpenAI(
        model=farm_config.model_name,
        temperature=farm_config.temperature,
        api_key=farm_config.openai_api_key
    )

    llm_with_tools = llm.bind_tools([schedule_planting, issue_farm_alert])

    # Find zones that are in pre-planting stage
    pre_planting_zones = [
        crop for crop in state.crop_data
        if crop.growth_stage == "pre_planting"
    ]

    if not pre_planting_zones:
        print("[PLANTING AGENT] No zones currently in pre-planting stage. Skipping.")
        state.current_agent = "harvest_agent"
        return state

    # Get soil data for pre-planting zones
    planting_zone_soil = [
        soil for soil in state.soil_data
        if any(z.zone_id == soil.zone_id for z in pre_planting_zones)
    ]

    # Weather forecast for planting window assessment
    forecast_summary = []
    if state.weather_data:
        for day in state.weather_data.forecast_7day[:7]:
            forecast_summary.append(
                f"{day['date']}: High {day['temperature_high_c']}C / "
                f"Low {day['temperature_low_c']}C, "
                f"Rain: {day['precipitation_mm']}mm, "
                f"Conditions: {day['conditions']}"
            )

    messages = [
        SystemMessage(content=PLANTING_AGENT_SYSTEM_PROMPT),
        HumanMessage(content=f"""
Evaluate planting opportunities for zones currently in pre-planting stage:

PRE-PLANTING ZONES:
{json.dumps([{
    'zone_id': z.zone_id,
    'planned_crop': z.crop_type
} for z in pre_planting_zones], indent=2)}

SOIL CONDITIONS FOR THESE ZONES:
{json.dumps([{
    'zone_id': s.zone_id,
    'crop_type': s.crop_type,
    'moisture_percent': s.moisture_percent,
    'temperature_c': s.temperature_c,
    'ph_level': s.ph_level,
    'compaction_index': s.compaction_index,
    'nitrogen_ppm': s.nitrogen_ppm,
    'phosphorus_ppm': s.phosphorus_ppm
} for s in planting_zone_soil], indent=2)}

7-DAY WEATHER FORECAST:
{chr(10).join(forecast_summary)}

SOIL ANALYSIS CONTEXT:
{state.soil_analysis or "Not available"}

Today's date: {datetime.now().strftime('%Y-%m-%d')}

For each zone, determine:
1. Whether conditions are suitable for planting now or in the next 14 days
2. Optimal planting date based on soil and weather conditions
3. Best seed variety for current timing
4. Seeding rate and method recommendations
5. Confidence score for your recommendation

Call schedule_planting for each zone where planting is recommended.
If conditions are not yet suitable, explain what conditions need to change.
""")
    ]

    response = llm_with_tools.invoke(messages)

    if hasattr(response, "tool_calls") and response.tool_calls:
        for tool_call in response.tool_calls:
            if tool_call["name"] == "schedule_planting":
                result = schedule_planting.invoke(tool_call["args"])

                state.planting_plan.append(PlantingAction(
                    zone_id=tool_call["args"]["zone_id"],
                    crop_type=tool_call["args"]["crop_type"],
                    recommended_date=tool_call["args"]["recommended_date"],
                    seed_variety=tool_call["args"]["seed_variety"],
                    seeding_rate_kg_per_ha=tool_call["args"].get("seeding_rate_kg_per_ha", 150.0),
                    row_spacing_cm=20.0,
                    planting_depth_cm=3.0,
                    confidence_score=tool_call["args"].get("confidence_score", 0.7),
                    reasoning=tool_call["args"].get("reasoning", "")
                ))

    state.current_agent = "harvest_agent"
    print(f"[PLANTING AGENT] Scheduled {len(state.planting_plan)} planting operations.")

    return state
