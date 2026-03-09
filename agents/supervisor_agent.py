# agents/supervisor_agent.py
# Supervisor Agent - orchestrates all agents and produces final farm schedule

import json
from datetime import datetime
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from config.settings import farm_config
from data.models import AgentState, FarmSchedule


SUPERVISOR_SYSTEM_PROMPT = """You are the Farm Management Supervisor AI for {farm_name}.

Your role is to synthesize all agent analyses and decisions into a clear, prioritized farm management report.

You receive inputs from:
- Weather Monitoring Agent
- Soil Monitoring Agent  
- Crop Growth Monitoring Agent
- Irrigation Planning Agent
- Planting Planning Agent
- Harvest Planning Agent

Your responsibilities:
1. Review all agent outputs for conflicts or inconsistencies
2. Prioritize actions by urgency and importance
3. Identify any critical alerts requiring immediate farmer attention
4. Produce an executive summary of the farm's current status
5. Validate that planned actions are coherent and achievable
6. Determine overall farm operational status

Status levels:
- CRITICAL: Immediate action required to prevent crop loss or damage
- ATTENTION NEEDED: Important issues requiring action within 24 hours
- NORMAL: Operations proceeding as expected, scheduled maintenance
- OPTIMAL: Excellent conditions across all metrics

Your output should be a concise, actionable farm status report that a farm manager 
can read in 2 minutes and know exactly what needs to be done today and this week."""


def run_supervisor_agent(state: AgentState) -> AgentState:
    """
    Supervisor agent - synthesizes all agent outputs into final farm schedule.
    This is both the entry point and final consolidation node.
    """
    print("\n[SUPERVISOR] Consolidating all agent outputs into farm schedule...")

    llm = ChatOpenAI(
        model=farm_config.model_name,
        temperature=farm_config.temperature,
        api_key=farm_config.openai_api_key
    )

    # If this is the initial call, just set up routing
    if not state.weather_analysis and not state.soil_analysis:
        print("[SUPERVISOR] Initiating farm monitoring cycle...")
        state.current_agent = "weather_agent"
        state.iteration_count += 1
        return state

    # Build comprehensive summary for synthesis
    irrigation_summary = []
    for action in state.irrigation_plan:
        irrigation_summary.append(
            f"Zone {action.zone_id}: {action.action} - {action.water_volume_liters}L "
            f"({action.duration_minutes} min) - Priority: {action.priority}"
        )

    planting_summary = []
    for action in state.planting_plan:
        planting_summary.append(
            f"Zone {action.zone_id}: Plant {action.crop_type} ({action.seed_variety}) "
            f"on {action.recommended_date} - Confidence: {action.confidence_score:.0%}"
        )

    harvest_summary = []
    for action in state.harvest_plan:
        harvest_summary.append(
            f"Zone {action.zone_id}: Harvest {action.crop_type} "
            f"{action.recommended_harvest_window_start} to {action.recommended_harvest_window_end} "
            f"- {action.estimated_yield_tons}t - Urgency: {action.urgency}"
        )

    # Build alerts list
    alerts = []
    if state.weather_data:
        if state.weather_data.frost_risk:
            alerts.append("FROST RISK: Protect sensitive crops, delay planting")
        if state.weather_data.heat_stress_risk:
            alerts.append("HEAT STRESS: Increase irrigation frequency, monitor closely")
        if state.weather_data.severe_weather_alert:
            alerts.append(f"WEATHER ALERT: {state.weather_data.severe_weather_alert}")

    for crop in state.crop_data:
        if crop.pest_pressure_level == "high":
            alerts.append(f"HIGH PEST PRESSURE in {crop.zone_id} ({crop.crop_type})")
        if crop.disease_risk == "high":
            alerts.append(f"HIGH DISEASE RISK in {crop.zone_id} ({crop.crop_type})")

    for soil in state.soil_data:
        if soil.moisture_percent < farm_config.critical_soil_moisture_low:
            alerts.append(f"CRITICAL LOW MOISTURE in {soil.zone_id}: {soil.moisture_percent}%")
        if soil.moisture_percent > farm_config.critical_soil_moisture_high:
            alerts.append(f"WATERLOGGING RISK in {soil.zone_id}: {soil.moisture_percent}%")

    # Generate executive summary via LLM
    messages = [
        SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT.format(
            farm_name=farm_config.farm_name
        )),
        HumanMessage(content=f"""
Synthesize all farm agent outputs into a final executive summary and farm status.

WEATHER ANALYSIS:
{state.weather_analysis or "Not available"}

SOIL ANALYSIS:
{state.soil_analysis or "Not available"}

CROP GROWTH ANALYSIS:
{state.growth_analysis or "Not available"}

IRRIGATION ACTIONS SCHEDULED ({len(state.irrigation_plan)} total):
{chr(10).join(irrigation_summary) if irrigation_summary else "None scheduled"}

PLANTING ACTIONS SCHEDULED ({len(state.planting_plan)} total):
{chr(10).join(planting_summary) if planting_summary else "None scheduled"}

HARVEST ACTIONS SCHEDULED ({len(state.harvest_plan)} total):
{chr(10).join(harvest_summary) if harvest_summary else "None scheduled"}

CRITICAL ALERTS:
{chr(10).join(alerts) if alerts else "No critical alerts"}

Produce:
1. Overall farm status (CRITICAL/ATTENTION NEEDED/NORMAL/OPTIMAL)
2. Executive summary (3-4 sentences for the farm manager)
3. Top 3 priority actions for today
4. This week's key operations calendar
5. Any risk factors to monitor
""")
    ]

    response = llm.invoke(messages)

    # Determine overall status
    overall_status = "normal"
    if any("CRITICAL" in a or "FROST" in a or "WATERLOGGING" in a for a in alerts):
        overall_status = "critical"
    elif len(alerts) > 2 or any("HIGH PEST" in a or "HIGH DISEASE" in a for a in alerts):
        overall_status = "attention_needed"
    elif not alerts and len(state.irrigation_plan) == 0:
        overall_status = "optimal"

    # Create final consolidated farm schedule
    state.farm_schedule = FarmSchedule(
        planning_horizon_days=farm_config.planning_horizon_days,
        irrigation_actions=state.irrigation_plan,
        planting_actions=state.planting_plan,
        harvest_actions=state.harvest_plan,
        alerts=alerts,
        overall_farm_status=overall_status,
        summary=response.content
    )

    state.current_agent = "complete"
    print(f"[SUPERVISOR] Farm schedule complete. Status: {overall_status.upper()}")
    print(f"[SUPERVISOR] {len(alerts)} alerts, {len(state.irrigation_plan)} irrigation, "
          f"{len(state.planting_plan)} planting, {len(state.harvest_plan)} harvest actions")

    return state
