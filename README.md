# AI Farm Management Agent System

An agentic AI system built with LangGraph and LangChain that continuously monitors
weather forecasts, soil sensors, and crop growth stages to dynamically plan planting,
irrigation, and harvesting schedules.

## Architecture Overview

The system uses a multi-agent graph where specialized AI agents collaborate:

```
[SUPERVISOR]
     |
     v
[WEATHER AGENT] --> [SOIL AGENT] --> [CROP GROWTH AGENT]
                                              |
                                              v
                              [IRRIGATION PLANNER]
                                              |
                                              v
                              [PLANTING AGENT]
                                              |
                                              v
                              [HARVEST AGENT]
                                              |
                                              v
                              [SUPERVISOR] (synthesis)
                                              |
                                              v
                                           [END]
```

## Agent Descriptions

### Supervisor Agent
- Entry point and final synthesizer
- Routes execution to appropriate monitoring agents
- Consolidates all agent outputs into a unified farm schedule
- Determines overall farm operational status (CRITICAL / ATTENTION NEEDED / NORMAL / OPTIMAL)

### Weather Monitoring Agent
- Fetches current weather conditions and 7-day forecasts
- Identifies frost risk, heat stress, severe weather alerts
- Determines optimal windows for field operations
- Influences irrigation timing (rain forecast reduces irrigation need)

### Soil Monitoring Agent
- Reads soil sensors from all 5 farm zones
- Monitors moisture, temperature, pH, nutrients (N/P/K), and compaction
- Flags zones requiring urgent irrigation vs. withholding
- Assesses field readiness for operations

### Crop Growth Monitoring Agent
- Tracks growth stages: pre_planting -> germination -> seedling -> vegetative
  -> flowering -> grain_fill -> maturation -> harvest_ready
- Monitors pest pressure and disease risk
- Forecasts harvest timing and yield estimates
- Determines crop-specific resource requirements

### Irrigation Planning Agent
- Synthesizes soil moisture data, weather forecasts, and growth stage water requirements
- Creates zone-specific irrigation schedules with volume and timing
- Issues critical alerts for drought stress or waterlogging
- Uses LangChain tool calling to schedule_irrigation actions

### Planting Planning Agent
- Evaluates zones in pre-planting stage for readiness
- Checks soil temperature, moisture, and compaction against crop requirements
- Selects optimal planting dates and seed varieties
- Provides confidence scores for each recommendation

### Harvest Planning Agent
- Identifies crops at maturation or harvest_ready stage
- Finds suitable weather windows (consecutive dry days)
- Determines urgency: immediate / this_week / scheduled / monitor
- Accounts for equipment mobilization requirements

## Project Structure

```
farm_agents/
|-- main.py                     # Entry point, runs complete management cycle
|-- requirements.txt
|-- .env.example
|
|-- config/
|   |-- settings.py             # Farm configuration (thresholds, crop types, etc.)
|
|-- data/
|   |-- models.py               # Pydantic models: AgentState, WeatherData, etc.
|
|-- agents/
|   |-- supervisor_agent.py     # Orchestrator and report synthesizer
|   |-- weather_agent.py        # Weather monitoring and analysis
|   |-- soil_agent.py           # Soil sensor monitoring and analysis
|   |-- crop_growth_agent.py    # Crop stage and health monitoring
|   |-- irrigation_agent.py     # Irrigation scheduling with tool use
|   |-- planting_agent.py       # Planting schedule optimization
|   |-- harvest_agent.py        # Harvest window identification
|
|-- tools/
|   |-- sensor_tools.py         # LangChain tools: fetch_weather, read_soil_sensors, etc.
|   |-- action_tools.py         # LangChain tools: schedule_irrigation, schedule_harvest, etc.
|
|-- graph/
|   |-- farm_graph.py           # LangGraph StateGraph definition and compilation
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 3. Run the system

```bash
python main.py
```

## Key Technical Decisions

### LangGraph StateGraph
- Uses a directed graph with conditional routing from the supervisor
- `AgentState` (Pydantic model) flows through all nodes as serialized dict
- `MemorySaver` checkpointer enables conversation persistence across cycles
- Each agent is a pure function: `(state: dict) -> dict`

### LangChain Tool Use
- Sensor tools use `@tool` decorator for schema auto-generation
- Action tools (irrigation, planting, harvest scheduling) use `bind_tools()` for 
  structured LLM function calling
- Tool results are parsed and stored back into `AgentState`

### Agent Communication
- Agents communicate exclusively through shared `AgentState`
- Each monitoring agent enriches state with its analysis string
- Planning agents read all analyses before making decisions
- No direct agent-to-agent communication (decoupled design)

### Simulation vs Production
The sensor tools currently simulate data using `random` for demonstration.
In production, replace with:
- Weather: OpenWeatherMap API, Tomorrow.io, or similar
- Soil: IoT sensor APIs (LoRaWAN gateway, MQTT broker)
- Growth: Drone imagery analysis, satellite NDVI data

## Configuration

Edit `config/settings.py` to customize:

```python
farm_config = FarmConfig(
    farm_name="My Farm",
    farm_location="Iowa, USA",
    farm_area_acres=300.0,
    crops=["wheat", "corn", "soybeans"],
    critical_soil_moisture_low=20.0,   # % - triggers urgent irrigation
    critical_soil_moisture_high=80.0,  # % - waterlogging risk
    frost_warning_temp_c=2.0,
    heat_stress_temp_c=35.0,
    planning_horizon_days=14
)
```

## Output

The system produces:
1. Real-time console output showing each agent's progress
2. A formatted farm management report with status, alerts, and schedules
3. A JSON report file saved with timestamp (e.g., `farm_report_20241115_143022.json`)

## Extending the System

To add a new agent:
1. Create `agents/new_agent.py` with a `run_new_agent(state: AgentState) -> AgentState` function
2. Add any new tools to `tools/`
3. Register the node in `graph/farm_graph.py`
4. Add routing logic for the new agent

## Production Deployment Considerations

- Schedule `main.py` with cron or a task scheduler for continuous monitoring
- Store `AgentState` in a database for historical analysis
- Add webhook notifications for critical alerts (SMS, push notifications)
- Implement sensor data validation and error handling for hardware failures
- Add authentication for the action scheduling tools
