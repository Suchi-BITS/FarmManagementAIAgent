# agents/soil_agent.py
# Soil Monitoring Agent - analyzes soil sensor data across all farm zones

import json
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from tools.sensor_tools import read_soil_sensors, get_historical_farm_data
from config.settings import farm_config
from data.models import AgentState, SoilSensorData


SOIL_AGENT_SYSTEM_PROMPT = """You are the Soil Monitoring Agent for an AI farm management system.

Your responsibilities:
1. Analyze real-time soil sensor data across all farm zones
2. Identify soil conditions that require immediate intervention (moisture stress, pH imbalance, nutrient deficiency)
3. Assess soil readiness for planned operations (planting, cultivation, harvest)
4. Cross-reference soil conditions with weather forecasts to anticipate changes
5. Recommend nutrient management actions

Critical thresholds to monitor:
- Soil moisture: Critical low < {moisture_low}%, Critical high > {moisture_high}%
- pH range for optimal crop growth: 6.0-7.0 (crop-specific variation applies)
- Compaction index: >70 requires intervention before field operations

Crops managed: {crops}

Your analysis must identify:
- Zones requiring immediate irrigation
- Zones where irrigation should be withheld
- Soil health concerns (pH, nutrients, compaction)
- Zones ready vs not ready for field operations
- Trends indicating future problems

Be specific about zone IDs, values, and urgency levels."""


def run_soil_agent(state: AgentState) -> AgentState:
    """
    Soil monitoring agent node for LangGraph.
    Reads all soil sensors and analyzes conditions across farm zones.
    """
    print("\n[SOIL AGENT] Reading soil sensors across all zones...")

    llm = ChatOpenAI(
        model=farm_config.model_name,
        temperature=farm_config.temperature,
        api_key=farm_config.openai_api_key
    )

    zone_ids = ["zone_a", "zone_b", "zone_c", "zone_d", "zone_e"]

    # Fetch soil sensor data from all zones
    soil_readings_raw = read_soil_sensors.invoke({"zone_ids": zone_ids})

    # Parse into models
    state.soil_data = [SoilSensorData(**reading) for reading in soil_readings_raw]

    # Format soil data for LLM analysis
    soil_summary = []
    for reading in soil_readings_raw:
        soil_summary.append(
            f"Zone {reading['zone_id']} ({reading['crop_type']}): "
            f"Moisture={reading['moisture_percent']}%, "
            f"pH={reading['ph_level']}, "
            f"Temp={reading['temperature_c']}°C, "
            f"N={reading['nitrogen_ppm']}ppm, "
            f"P={reading['phosphorus_ppm']}ppm, "
            f"K={reading['potassium_ppm']}ppm, "
            f"Compaction={reading['compaction_index']}"
        )

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=SOIL_AGENT_SYSTEM_PROMPT.format(
            moisture_low=farm_config.critical_soil_moisture_low,
            moisture_high=farm_config.critical_soil_moisture_high,
            crops=", ".join(farm_config.crops)
        )),
        HumanMessage(content=f"""
Analyze the following soil sensor readings from all farm zones:

SOIL SENSOR DATA:
{chr(10).join(soil_summary)}

CURRENT WEATHER CONTEXT:
{state.weather_analysis or "Weather analysis not yet available"}

Full sensor data:
{json.dumps(soil_readings_raw, indent=2)}

Provide analysis covering:
1. Zones requiring immediate irrigation (ranked by urgency)
2. Zones where irrigation should be held or reduced
3. Soil health concerns by zone
4. Field operation readiness assessment
5. Nutrient management recommendations
6. Any critical alerts needing immediate action
""")
    ])

    response = llm.invoke(prompt.format_messages())
    state.soil_analysis = response.content
    state.current_agent = "crop_growth_agent"

    print(f"[SOIL AGENT] Analyzed {len(zone_ids)} zones. Analysis complete.")

    return state
