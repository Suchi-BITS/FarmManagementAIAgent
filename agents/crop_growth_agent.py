# agents/crop_growth_agent.py
# Crop Growth Monitoring Agent - tracks growth stages and crop health

import json
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from tools.sensor_tools import read_crop_growth_sensors
from config.settings import farm_config
from data.models import AgentState, CropGrowthData


CROP_GROWTH_SYSTEM_PROMPT = """You are the Crop Growth Monitoring Agent for an AI farm management system.

Your responsibilities:
1. Track current growth stages across all farm zones
2. Identify crops approaching critical growth milestones (flowering, harvest readiness)
3. Assess pest pressure and disease risks that require intervention
4. Determine harvest readiness windows and optimal timing
5. Flag crops that need special care based on current growth stage

Growth stage knowledge:
- Pre-planting: Soil preparation and input planning
- Germination: Critical moisture maintenance, temperature sensitivity
- Seedling: Vulnerable to frost, weeds, and moisture stress  
- Vegetative: Nutrient uptake peak, irrigation critical
- Flowering: Pollination sensitive to temperature extremes and wind
- Grain fill: Consistent moisture essential for yield
- Maturation: Reduce irrigation, monitor moisture content
- Harvest ready: Time-critical, weather window assessment essential

Crops managed: {crops}

For each zone, assess:
- Whether current management matches growth stage requirements
- Upcoming stage transitions requiring action changes
- Pest/disease pressure needing intervention
- Harvest timing recommendations for mature crops
- Resource allocation priorities across zones"""


def run_crop_growth_agent(state: AgentState) -> AgentState:
    """
    Crop growth monitoring agent node for LangGraph.
    Reads growth sensors and assesses crop status across all zones.
    """
    print("\n[CROP GROWTH AGENT] Reading crop growth sensors and assessing status...")

    llm = ChatOpenAI(
        model=farm_config.model_name,
        temperature=farm_config.temperature,
        api_key=farm_config.openai_api_key
    )

    zone_ids = ["zone_a", "zone_b", "zone_c", "zone_d", "zone_e"]

    # Fetch crop growth data
    growth_readings_raw = read_crop_growth_sensors.invoke({"zone_ids": zone_ids})

    # Parse into models
    state.crop_data = [CropGrowthData(**reading) for reading in growth_readings_raw]

    # Format growth data for LLM
    growth_summary = []
    for reading in growth_readings_raw:
        harvest_info = ""
        if reading.get("estimated_days_to_harvest") is not None:
            harvest_info = f", ~{reading['estimated_days_to_harvest']} days to harvest"

        growth_summary.append(
            f"Zone {reading['zone_id']} ({reading['crop_type']}): "
            f"Stage={reading['growth_stage']}, "
            f"Days planted={reading.get('days_since_planting', 'N/A')}{harvest_info}, "
            f"Canopy={reading['canopy_coverage_percent']}%, "
            f"LAI={reading['leaf_area_index']}, "
            f"Pest={reading['pest_pressure_level']}, "
            f"Disease={reading['disease_risk']}"
        )

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=CROP_GROWTH_SYSTEM_PROMPT.format(
            crops=", ".join(farm_config.crops)
        )),
        HumanMessage(content=f"""
Analyze the current crop growth status across all farm zones:

CROP GROWTH DATA:
{chr(10).join(growth_summary)}

WEATHER CONTEXT:
{state.weather_analysis or "Not available"}

SOIL CONTEXT:
{state.soil_analysis or "Not available"}

Full sensor data:
{json.dumps(growth_readings_raw, indent=2)}

Provide assessment covering:
1. Crops ready for harvest or approaching harvest readiness
2. Growth stage transitions expected in next 7-14 days
3. Pest and disease interventions required
4. Management changes needed based on current growth stages
5. Resource prioritization recommendations across zones
6. Any growth anomalies or concerns requiring investigation
""")
    ])

    response = llm.invoke(prompt.format_messages())
    state.growth_analysis = response.content
    state.current_agent = "irrigation_planner"

    # Count critical conditions
    harvest_ready = [r for r in growth_readings_raw if r["growth_stage"] == "harvest_ready"]
    high_pest = [r for r in growth_readings_raw if r["pest_pressure_level"] == "high"]
    print(f"[CROP GROWTH AGENT] {len(harvest_ready)} zones harvest-ready, "
          f"{len(high_pest)} zones with high pest pressure.")

    return state
