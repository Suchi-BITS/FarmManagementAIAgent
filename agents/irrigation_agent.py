# agents/irrigation_agent.py
# Irrigation Planning Agent - makes irrigation scheduling decisions

import json
from datetime import datetime, timedelta
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from tools.action_tools import schedule_irrigation, issue_farm_alert
from config.settings import farm_config
from data.models import AgentState, IrrigationAction


IRRIGATION_AGENT_SYSTEM_PROMPT = """You are the Irrigation Planning Agent for an AI farm management system.

You make data-driven irrigation scheduling decisions by synthesizing:
- Real-time soil moisture sensor data
- Weather forecasts (upcoming rain reduces irrigation need)
- Crop growth stage water requirements (these vary significantly by stage)
- Field conditions (avoid irrigating before predicted rain or during heat of day)

Water requirements by growth stage:
- Pre-planting/Germination: 20-25mm/week (critical for establishment)
- Seedling: 15-20mm/week (sensitive to both under and over-watering)
- Vegetative: 25-35mm/week (peak growth period)
- Flowering: 30-40mm/week (critical stage, stress causes yield loss)
- Grain fill: 25-30mm/week (consistent supply essential)
- Maturation: 10-15mm/week (reduce to encourage ripening)
- Harvest ready: Minimal to none

Decision rules:
- Soil moisture < 25%: Urgent irrigation required
- Soil moisture 25-35%: Schedule irrigation within 24 hours
- Soil moisture 35-60%: Normal irrigation scheduling
- Soil moisture > 65%: Withhold irrigation, check drainage
- Rain forecast > 15mm in next 48h: Defer irrigation
- Temperature > 35C: Night or early morning irrigation preferred

You must call the schedule_irrigation tool for each zone requiring action.
Format financial and operational decisions with clear priority levels."""


def run_irrigation_agent(state: AgentState) -> AgentState:
    """
    Irrigation planning agent node for LangGraph.
    Creates specific irrigation schedules based on all available data.
    """
    print("\n[IRRIGATION AGENT] Planning irrigation schedules...")

    llm = ChatOpenAI(
        model=farm_config.model_name,
        temperature=farm_config.temperature,
        api_key=farm_config.openai_api_key
    )

    llm_with_tools = llm.bind_tools([schedule_irrigation, issue_farm_alert])

    # Build comprehensive data summary
    soil_data_summary = []
    for soil in state.soil_data:
        crop_stage = "unknown"
        for crop in state.crop_data:
            if crop.zone_id == soil.zone_id:
                crop_stage = crop.growth_stage
                break
        soil_data_summary.append({
            "zone_id": soil.zone_id,
            "crop_type": soil.crop_type,
            "growth_stage": crop_stage,
            "moisture_percent": soil.moisture_percent,
            "temperature_c": soil.temperature_c
        })

    weather_summary = {}
    if state.weather_data:
        weather_summary = {
            "current_temp_c": state.weather_data.temperature_c,
            "precipitation_today_mm": state.weather_data.precipitation_mm,
            "forecast_rain_48h": sum(
                d.get("precipitation_mm", 0)
                for d in state.weather_data.forecast_7day[:2]
            ),
            "heat_stress_risk": state.weather_data.heat_stress_risk,
            "frost_risk": state.weather_data.frost_risk
        }

    messages = [
        SystemMessage(content=IRRIGATION_AGENT_SYSTEM_PROMPT),
        HumanMessage(content=f"""
Based on the sensor data and analyses, create irrigation schedules for all farm zones.

ZONE SOIL AND CROP STATUS:
{json.dumps(soil_data_summary, indent=2)}

WEATHER SUMMARY:
{json.dumps(weather_summary, indent=2)}

SOIL ANALYSIS:
{state.soil_analysis or "Not available"}

WEATHER ANALYSIS:
{state.weather_analysis or "Not available"}

For each zone, determine:
1. Whether irrigation is needed now, should be scheduled, or should be withheld
2. Appropriate water volume and duration
3. Priority level
4. Best timing given weather and crop stage

Call schedule_irrigation for each zone that needs action.
Call issue_farm_alert for any critical irrigation emergencies (severe drought stress, waterlogging).
""")
    ]

    # Run the agent with tool use
    response = llm_with_tools.invoke(messages)
    tool_results = []

    if hasattr(response, "tool_calls") and response.tool_calls:
        for tool_call in response.tool_calls:
            if tool_call["name"] == "schedule_irrigation":
                result = schedule_irrigation.invoke(tool_call["args"])
                tool_results.append(result)

                # Create IrrigationAction record for state
                state.irrigation_plan.append(IrrigationAction(
                    zone_id=tool_call["args"]["zone_id"],
                    action=tool_call["args"]["action"],
                    water_volume_liters=tool_call["args"].get("water_volume_liters", 0),
                    duration_minutes=tool_call["args"].get("duration_minutes", 0),
                    priority=tool_call["args"].get("priority", "normal"),
                    reason=tool_call["args"].get("reason", "")
                ))

            elif tool_call["name"] == "issue_farm_alert":
                issue_farm_alert.invoke(tool_call["args"])

    state.current_agent = "planting_agent"
    print(f"[IRRIGATION AGENT] Scheduled {len(state.irrigation_plan)} irrigation actions.")

    return state
