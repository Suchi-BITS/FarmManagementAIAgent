# agents/weather_agent.py
# Weather Monitoring Agent - continuously monitors weather and assesses farm impact

import json
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from tools.sensor_tools import fetch_weather_data
from config.settings import farm_config
from data.models import AgentState


WEATHER_AGENT_SYSTEM_PROMPT = """You are the Weather Monitoring Agent for an AI farm management system.

Your responsibilities:
1. Analyze current weather conditions and 7-day forecasts
2. Identify weather risks: frost, heat stress, excessive rain, drought, severe storms
3. Assess how weather conditions affect each crop type and growth stage
4. Provide specific, actionable weather impact assessments for farm planning

Farm context:
- Location: {farm_location}
- Crops managed: {crops}
- Planning horizon: {planning_horizon} days

When analyzing weather data, always:
- Flag any immediate threats requiring urgent action
- Identify optimal windows for field operations (planting, spraying, harvesting)
- Note precipitation forecasts that affect irrigation scheduling
- Warn about conditions that could damage crops or delay operations

Be precise and data-driven. State temperature values, precipitation amounts, and timing explicitly.
Format your analysis as structured paragraphs addressing: current conditions, forecast outlook, 
risk assessment, and operational windows."""


def run_weather_agent(state: AgentState) -> AgentState:
    """
    Weather monitoring agent node for LangGraph.
    Fetches current weather data and produces an analysis for other agents.
    """
    print("\n[WEATHER AGENT] Fetching weather data and analyzing conditions...")

    llm = ChatOpenAI(
        model=farm_config.model_name,
        temperature=farm_config.temperature,
        api_key=farm_config.openai_api_key
    )

    # Bind the weather tool to the LLM
    llm_with_tools = llm.bind_tools([fetch_weather_data])

    # Step 1: Fetch weather data
    weather_raw = fetch_weather_data.invoke({"location": farm_config.farm_location})
    state.weather_data = type('WeatherData', (), weather_raw)()  # store as raw dict accessible form

    # Step 2: Use LLM to analyze the weather data
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=WEATHER_AGENT_SYSTEM_PROMPT.format(
            farm_location=farm_config.farm_location,
            crops=", ".join(farm_config.crops),
            planning_horizon=farm_config.planning_horizon_days
        )),
        HumanMessage(content=f"""
Analyze the following weather data for our farm and provide a comprehensive assessment:

CURRENT CONDITIONS:
- Temperature: {weather_raw['temperature_c']}°C
- Humidity: {weather_raw['humidity_percent']}%
- Precipitation: {weather_raw['precipitation_mm']}mm
- Wind Speed: {weather_raw['wind_speed_kmh']} km/h
- Frost Risk: {weather_raw['frost_risk']}
- Heat Stress Risk: {weather_raw['heat_stress_risk']}
- Severe Weather Alert: {weather_raw.get('severe_weather_alert', 'None')}

7-DAY FORECAST:
{json.dumps(weather_raw['forecast_7day'], indent=2)}

Provide your analysis covering:
1. Current conditions impact on crops
2. Forecast risks and opportunities
3. Recommended operational windows for the next 7 days
4. Any urgent alerts that need immediate farmer attention
""")
    ])

    response = llm.invoke(prompt.format_messages())
    state.weather_analysis = response.content
    state.current_agent = "soil_agent"

    # Store weather data properly
    from data.models import WeatherData
    import datetime
    state.weather_data = WeatherData(
        location=weather_raw["location"],
        temperature_c=weather_raw["temperature_c"],
        humidity_percent=weather_raw["humidity_percent"],
        precipitation_mm=weather_raw["precipitation_mm"],
        wind_speed_kmh=weather_raw["wind_speed_kmh"],
        forecast_7day=weather_raw["forecast_7day"],
        frost_risk=weather_raw["frost_risk"],
        heat_stress_risk=weather_raw["heat_stress_risk"],
        severe_weather_alert=weather_raw.get("severe_weather_alert")
    )

    print(f"[WEATHER AGENT] Analysis complete. Frost risk: {weather_raw['frost_risk']}, "
          f"Heat stress: {weather_raw['heat_stress_risk']}")

    return state
