# AI Farm Management Agent System v2

A hierarchical multi-agent AI system built with LangGraph and LangChain that
continuously monitors crop health, soil conditions, irrigation needs, and harvest
readiness across a multi-zone farm operation. Four specialist agents run under a
central supervisor, each retrieving domain-specific agronomic knowledge from an
embedded RAG knowledge base before calling the LLM.

The system runs fully end-to-end in demo mode with no external dependencies, and
upgrades automatically to live GPT-4o reasoning when an OpenAI API key is present.

---

## Table of Contents

1. [Use Case and Problem Statement](#use-case-and-problem-statement)
2. [System Architecture](#system-architecture)
3. [Architectural Pattern — Hierarchical Multi-Agent](#architectural-pattern)
4. [RAG Knowledge Base](#rag-knowledge-base)
5. [Agent Graph Topology](#agent-graph-topology)
6. [Step-by-Step Workflow](#step-by-step-workflow)
7. [File-by-File Explanation](#file-by-file-explanation)
8. [Data Layer and Simulation Design](#data-layer-and-simulation-design)
9. [Farm Zones and Crop Configuration](#farm-zones-and-crop-configuration)
10. [Tool Reference](#tool-reference)
11. [Production Deployment Guide](#production-deployment-guide)
12. [Run Modes and Quick Start](#run-modes-and-quick-start)
13. [Sample Output](#sample-output)
14. [Disclaimer](#disclaimer)

---

## Use Case and Problem Statement

### The Domain

Modern farm management requires simultaneous decision-making across at least four
independent domains: soil health and nutrient status, crop growth and pest pressure,
water scheduling and deficit management, and harvest timing aligned with market prices.
A 150-acre mixed-crop farm generates continuous data streams from soil sensors, drone
multispectral imagery, weather stations, and market price feeds. A farm manager cannot
manually integrate all of these streams in real time.

### The Problem

Three specific gaps make this an appropriate AI agent problem.

First, the decisions are domain-specific but interdependent. A soil moisture reading
below 20 percent triggers irrigation only if the crop is in a water-sensitive growth
stage and no significant rainfall is forecast in the next 48 hours. Making that
determination requires simultaneously reading soil data, crop growth stage data, and
weather forecast data — three domains that no single human specialist monitors
continuously.

Second, the volume of data exceeds what a single person can track. Eight soil sensors,
four drone overpasses per week, daily weather forecasts, and three commodity price
streams across four crop zones create roughly 150 to 200 data points per day that
require interpretation.

Third, the cost of delayed or wrong decisions is high. Failing to irrigate a corn
field during silking (the R1 growth stage) when soil moisture drops below 40 percent
of field capacity can reduce yield by 40 to 50 percent within 48 hours. Missing the
optimal harvest window by four days can drop grain moisture content below 11 percent,
causing shattering losses of 5 to 8 percent of total yield.

### What This System Delivers

A continuous monitoring cycle that produces a single prioritized farm operations report
every time it is invoked. The report answers: what needs to happen today, what is
scheduled for this week, what the current market conditions suggest for contracting
strategy, and what alerts require the farm manager's immediate attention.

---

## System Architecture

```
+------------------------------------------------------------------+
|  DATA LAYER                                                      |
|                                                                  |
|  get_weather()          OpenWeatherMap / NOAA NWS (production)   |
|  get_soil_readings()    Sentek / Stevens HydraProbe sensors      |
|  get_crop_readings()    Sentinel-2 NDVI / DJI drone              |
|  get_market_prices()    CME Group API / USDA NASS                |
+------------------------------------------------------------------+
         |
         v
+------------------------------------------------------------------+
|  RAG KNOWLEDGE BASE (rag/knowledge_base.py)                     |
|                                                                  |
|  10 agronomic documents covering:                                |
|    soil moisture thresholds, pH guidelines, compaction          |
|    crop growth stage decision points, pest thresholds           |
|    irrigation ET scheduling, waterlogging protocols             |
|    harvest quality standards, commodity price rules             |
|                                                                  |
|  Pure-Python TF-IDF vector store — no external dependencies     |
|  Retrieved context injected into each specialist's system prompt |
+------------------------------------------------------------------+
         |
         v
+------------------------------------------------------------------+
|  LANGGRAPH HIERARCHICAL MULTI-AGENT GRAPH                       |
|                                                                  |
|  +------------------+                                           |
|  | SUPERVISOR AGENT | <-- entry point: dispatches to specialists|
|  +------------------+                                           |
|         |                                                        |
|    (sequential pipeline)                                         |
|         |                                                        |
|  +------+-----------------------------------------------+       |
|  |                                                       |       |
|  v                 v                  v            v             |
|  SOIL AGENT    CROP AGENT    IRRIGATION AGENT   HARVEST AGENT   |
|  (moisture,    (NDVI, pest,  (ET scheduling,   (timing,         |
|  pH, N/P/K)   disease,       waterlogging)     market prices)  |
|               yield)                                            |
|  |                                                       |       |
|  +------+-----------------------------------------------+       |
|         |                                                        |
|  +------------------+                                           |
|  | SUPERVISOR AGENT | <-- synthesis: merges all outputs         |
|  +------------------+                                           |
+------------------------------------------------------------------+
         |
         v
+------------------------------------------------------------------+
|  OUTPUT LAYER                                                    |
|  Farm status (critical / attention_needed / normal / optimal)   |
|  Prioritized alert list                                          |
|  Irrigation schedule with zone-level actions                    |
|  Harvest readiness assessment with market context               |
|  Integrated farm operations report                              |
+------------------------------------------------------------------+
```

---

## Architectural Pattern

### Hierarchical Multi-Agent

This system uses the hierarchical multi-agent pattern, where a supervisor agent
orchestrates multiple specialist agents, each of which operates independently within
its domain before returning results to the supervisor for integration.

This pattern is appropriate for farm management because the four domains (soil, crop,
irrigation, harvest) are genuinely independent at the observation level. The soil
agent does not need to wait for the crop agent's results before reading soil sensors.
Each specialist can execute in its domain without knowledge of what the others are
doing. The supervisor then resolves cross-domain dependencies when building the
integrated report.

The alternative — a single agent reading all data — would produce less reliable
results because the LLM's attention mechanism would need to span soil science, plant
physiology, hydraulics, and commodity markets simultaneously within one reasoning chain.
Decomposing by domain allows each specialist to use a focused system prompt calibrated
to its domain's vocabulary and decision logic.

The supervisor runs twice. On the first call it initializes the monitoring cycle and
triggers the specialist pipeline. On the second call (after all specialists have run)
it reads all specialist outputs from the shared state and synthesizes the final report.
This double-call pattern is a standard technique in hierarchical LangGraph systems.

### Why RAG Is Added

Each specialist's LLM call is grounded by RAG-retrieved documents from the agronomic
knowledge base. Without RAG, the LLM might cite incorrect irrigation trigger thresholds
(e.g., stating that corn needs irrigation below 50 percent soil moisture when the
verified agronomic standard is 40 percent of field capacity during vegetative stages,
dropping to 30 percent being critical only during silking). It might also misstate
harvest moisture targets or pest economic thresholds.

RAG injects the verified threshold values directly into the specialist's system prompt
before the LLM call, so the model reasons against documented agronomic standards rather
than its training-set approximation of them.

---

## RAG Knowledge Base

The knowledge base (`rag/knowledge_base.py`) contains 10 curated agronomic document
chunks organized into 6 categories.

| Category | Documents | Coverage |
|---|---|---|
| `soil` | SOIL-01, SOIL-02, SOIL-03 | Moisture thresholds, pH/nutrient availability, compaction |
| `crop_management` | CROP-01, CROP-02 | Growth stage decision points, pest/disease thresholds |
| `irrigation` | IRR-01, IRR-02 | ET-based scheduling, waterlogging protocols |
| `fertilization` | FERT-01 | N/P/K application guidelines by crop |
| `harvest` | HARVEST-01 | Moisture targets, quality grades, weather windows |
| `market` | MARKET-01 | Forward contracting thresholds, basis, storage premium |

Each specialist agent issues two TF-IDF retrieval queries tailored to its domain before
calling the LLM. The retrieved document text is prepended to the specialist's system
prompt under a `RETRIEVED AGRONOMIC KNOWLEDGE` heading.

The vector store uses pure-Python TF-IDF cosine similarity with no external libraries.
In production, replace `KnowledgeBase` with ChromaDB or FAISS backed by
sentence-transformer embeddings. The `retrieve(query, top_k, category)` interface
remains unchanged.

---

## Agent Graph Topology

```
START
  |
  v
supervisor_agent (init)
  | sets current_agent = "specialists"
  |
  v
soil_agent
  | RAG queries: moisture thresholds, pH/nutrient guidelines
  | reads: get_soil_readings()
  | writes: state["soil_analysis"], state["soil_alerts"]
  |
  v
crop_agent
  | RAG queries: growth stage decision points, pest thresholds
  | reads: get_crop_readings(), state["weather_data"]
  | writes: state["crop_analysis"], state["crop_alerts"]
  |
  v
irrigation_agent
  | RAG queries: ET scheduling, waterlogging protocols
  | reads: state["soil_data"], state["crop_data"], state["weather_data"]
  | writes: state["irrigation_analysis"], state["irrigation_actions"]
  |
  v
harvest_agent
  | RAG queries: harvest timing/quality, commodity pricing rules
  | reads: state["crop_data"], state["weather_data"], get_market_prices()
  | writes: state["harvest_analysis"], state["harvest_ready"]
  |
  v
supervisor_agent (synthesis)
  | RAG queries: integrated farm management priorities
  | reads: all specialist analyses and alerts
  | writes: state["farm_report"], state["overall_status"]
  |
  v
END
```

Each agent reads from and writes to the shared `FarmState` TypedDict. No agent
calls another agent directly — all communication is through the state dictionary.

---

## Step-by-Step Workflow

### Step 1: Supervisor Initialization

The supervisor checks whether any specialist has run by testing if
`state["soil_analysis"] is None`. On the first call this is true, so it prints the
monitoring cycle header, sets `current_agent = "specialists"` to trigger the
conditional routing edge toward the soil agent, and returns. No LLM call is made at
this step.

### Step 2: Soil Agent

The soil agent calls `get_soil_readings()` to retrieve moisture, temperature, pH,
nitrogen, phosphorus, potassium, organic matter, and compaction index for all four
farm zones. It then issues two RAG retrieval queries: one for soil moisture threshold
context and one for pH and nutrient availability.

The agent applies deterministic threshold logic to classify each zone: critical low
moisture (below 20 percent triggers a critical alert), waterlogging risk (above 80
percent triggers a stop-irrigation alert), pH out of the optimal 5.8 to 7.2 range,
nitrogen below 20 ppm (deficiency requiring immediate application), and compaction
above 50 (requiring tillage consideration).

The LLM call receives all zone readings plus the RAG context and produces a written
soil analysis covering urgent issues, nutrient application needs, and irrigation
priorities by zone. In demo mode the analysis is built directly from the threshold
results without an LLM call.

State written: `soil_data`, `soil_analysis`, `soil_alerts`

### Step 3: Crop Agent

The crop agent calls `get_crop_readings()` to retrieve growth stage, days since
planting, days to harvest, NDVI, pest pressure level, disease risk level, and yield
forecast for each zone. Weather data is read from the state (written by the soil agent
which fetched it first).

RAG queries retrieve growth stage action requirements (what nitrogen applications,
irrigation changes, or pest scouting actions are appropriate at the current stage) and
pest and disease economic threshold values (the point at which treatment cost is
justified by expected yield protection).

The agent flags zones where pest pressure is high (triggers a crop alert) or disease
risk is medium or high (triggers a crop alert). NDVI below 0.4 triggers a stress
warning that is included in the analysis text.

State written: `crop_data`, `crop_analysis`, `crop_alerts`

### Step 4: Irrigation Agent

The irrigation agent reads soil and crop data from state along with the weather
forecast. It computes a zone-by-zone irrigation schedule using the following decision
logic, which is validated against the RAG-retrieved ET scheduling guidance.

If soil moisture exceeds 80 percent: issue a stop or defer action to prevent
waterlogging. If moisture is below the critical low of 20 percent: issue an emergency
irrigate action with high water volume. If moisture is below 40 percent and the crop
is in a reproductive stage (flowering or grain fill) and rainfall forecast for the
next 48 hours is below 5mm: issue a high-priority irrigation action. If significant
rainfall is forecast: defer irrigation. Otherwise: maintain the current irrigation
schedule.

Each zone action is tagged with priority (critical, high, normal, low), a water
volume in litres, and a reason string that the farm manager can read directly.

State written: `irrigation_analysis`, `irrigation_actions`

### Step 5: Harvest Agent

The harvest agent reads crop data from state and fetches current commodity prices from
`get_market_prices()`. It applies harvest readiness criteria from the RAG-retrieved
harvest timing document.

Zones with 5 or fewer days to harvest are classified as ready (urgency: now if 2 or
fewer days, urgency: this week otherwise). Zones with 6 to 14 days are classified as
upcoming. The agent assesses the weather window by counting forecast dry days (less
than 5mm precipitation) in the 7-day outlook — 3 or more consecutive dry days
constitute a viable harvest window.

The market analysis compares current commodity prices against the RAG-retrieved
contracting threshold values (wheat above $7.00/bu, corn above $5.50/bu, soybeans
above $13.50/bu) and generates a forward contracting signal for each crop.

State written: `harvest_analysis`, `harvest_ready`, `harvest_upcoming`, `market_prices`

### Step 6: Supervisor Synthesis

The supervisor is called a second time after all specialists have run. It reads all
four specialist analyses plus all alerts and market prices from state. It issues a
RAG query for integrated farm management context, then calls the LLM (or builds a
demo report) with the full context.

The overall status is computed deterministically: critical if any critical alerts are
present or three or more total alerts, attention_needed if two or more alerts, optimal
if no alerts and no irrigation actions required, normal otherwise.

The final farm report covers: overall status, executive summary, soil zone status,
crop health by zone, irrigation schedule, harvest timing and market context, top 3
priority actions today, and this week's operations calendar.

State written: `farm_report`, `overall_status`, `all_alerts`

---

## File-by-File Explanation

### main.py

Entry point. Creates the initial `FarmState` dict with all keys set to None or empty
lists. Attempts to build and invoke the LangGraph compiled graph. Falls back to direct
sequential function calls if LangGraph is not installed. After the pipeline completes,
`print_report()` formats and prints: overall status, alert list, zone-by-zone
irrigation actions with priorities, harvest readiness table, market prices, and the
full farm report text.

### config/settings.py

Single stdlib dataclass `FarmConfig` holding all configuration. OpenAI API key from
environment, farm identity (name, location, area, crop list), and decision thresholds
(critical moisture boundaries at 20 and 80 percent, frost warning at 2 degrees C,
heat stress at 35 degrees C, planning horizon of 14 days).

### data/simulation.py

Deterministic simulation layer with four data functions. `get_weather()` returns
current conditions and a 7-day forecast using day-seeded RNG. `get_soil_readings()`
returns sensor data for all four zones with zone-specific baselines calibrated to
crop type and position. `get_crop_readings()` returns growth stage, NDVI, pest and
disease pressure, and yield forecast per zone. `get_market_prices()` returns current
commodity prices for wheat, corn, and soybeans.

All functions use the date-seeded `_rng(salt)` generator, ensuring identical values
within a calendar day and natural day-to-day variation.

### rag/knowledge_base.py

Pure-Python TF-IDF in-memory vector store. `KnowledgeBase._build()` computes IDF
weights across all 10 document chunks and stores TF-IDF vectors per chunk. `retrieve(
query, top_k, category)` computes the query TF-IDF vector and returns the top-k chunks
by cosine similarity. `retrieve_text()` formats the results as an annotated text block.

The singleton `FARM_KB` is imported by `specialist_agents.py` and queried twice per
agent. Production replacement: swap `KnowledgeBase` with ChromaDB backed by
`all-MiniLM-L6-v2` sentence-transformer embeddings. The `retrieve()` interface is
unchanged.

### agents/base.py

`_demo_mode()` returns True when no valid API key is present. `call_llm()` dispatches
to either a demo response string or a live `ChatOpenAI.invoke()` call. Shared by all
four specialists and the supervisor.

### agents/specialist_agents.py

All four specialist agents in one file. Each agent follows the same pattern:
(1) call RAG for domain-relevant documents, (2) fetch or read simulation data,
(3) apply deterministic threshold logic to compute alerts and summaries,
(4) call LLM with RAG context in system prompt and data in user message,
(5) write results and alerts to state, (6) set `current_agent` to next agent name.

The four agents are `run_soil_agent`, `run_crop_agent`, `run_irrigation_agent`, and
`run_harvest_agent`.

### agents/supervisor_agent.py

`run_supervisor()` is called twice per cycle. The first call detects that no specialist
has run (soil_analysis is None) and initializes routing. The second call reads all
specialist outputs, queries RAG for integrated context, and produces the consolidated
farm report. Status determination is deterministic; only the report narrative is
LLM-generated.

### graph/farm_graph.py

`FarmState` TypedDict covers all state keys. `build_farm_graph()` registers all five
nodes (supervisor plus four specialists), sets supervisor as entry point, adds a
conditional edge from supervisor routing to soil_agent or END, and wires the four
specialist nodes sequentially to each other and back to supervisor. Returns the
compiled LangGraph graph.

---

## Data Layer and Simulation Design

The four data functions are physically self-consistent. Crop zones are assigned a
fixed base moisture, crop type, and NDVI baseline that reflects real agronomic
relationships: corn zones have higher baseline water demand than wheat zones, soybeans
have higher susceptibility to mid-season drought stress than wheat.

The RNG salt parameters ensure independent variation streams. Weather uses salt 1,
soil sensors use salts 100 through 103 (one per zone), crop readings use salts 200
through 203, and market prices use salt 300. This means weather and soil readings can
independently show elevated or reduced values without forcing correlation between them,
which is realistic — a warm dry day does not always correspond to the lowest soil
moisture (which reflects cumulative depletion over days, not just the current day).

---

## Farm Zones and Crop Configuration

The farm operates four production zones.

| Zone ID | Crop | Area (acres) | Notes |
|---|---|---|---|
| Z-NORTH | Wheat | 40.0 | Dryland wheat — highest drought tolerance |
| Z-SOUTH | Corn | 45.0 | Highest water demand, critical during silking |
| Z-EAST | Soybeans | 35.0 | Moderate water demand, sensitive at pod fill |
| Z-WEST | Wheat | 30.0 | Adjacent to creek — waterlogging risk in wet seasons |

Total managed area: 150 acres. All zones use the same monitoring frequency and
trigger thresholds, but the LLM's RAG context adapts its recommendations to each
crop's specific growth stage requirements.

---

## Tool Reference

All data functions in `data/simulation.py` are called directly by the specialist
agents (not via LangChain `@tool` wrappers in this version). Production replacement
points are documented in each function's docstring.

| Function | Returns | Production API |
|---|---|---|
| `get_weather()` | Current conditions + 7-day forecast | OpenWeatherMap / Tomorrow.io / NOAA NWS |
| `get_soil_readings()` | Per-zone sensor data | Sentek EnviroSCAN / Stevens HydraProbe LoRaWAN |
| `get_crop_readings()` | NDVI, growth stage, pest/disease | Sentinel-2 GEE / DJI Agras + MicaSense |
| `get_market_prices()` | Commodity prices (wheat/corn/soybeans) | CME Group API / USDA NASS Quick Stats |

---

## Production Deployment Guide

### Replacing Soil Sensor Data

Replace `get_soil_readings()` with reads from a LoRaWAN gateway aggregating Sentek
EnviroSCAN or Stevens HydraProbe sensors. These sensors transmit volumetric water
content, temperature, and electrical conductivity every 15 to 30 minutes over LoRa
to a farm gateway, which forwards to a cloud API. Nutrient sensors (nitrogen,
phosphorus, potassium) typically require dedicated probes such as HACH nutrient
monitors or periodic lab analysis rather than continuous IoT sensing.

### Replacing Crop Growth Data

Replace `get_crop_readings()` with one of two approaches. For broad-acre field
monitoring, use Google Earth Engine Python API to pull Sentinel-2 NDVI composite
images (`COPERNICUS/S2_SR`) at 10-meter resolution. For high-resolution canopy
analysis and disease detection, use DJI Agras drone overpasses with MicaSense RedEdge
multispectral camera, processed through Pix4Dfields or Agisoft Metashape.

### Replacing Weather Data

Replace `get_weather()` with Tomorrow.io API (high-resolution, hyper-local forecasts
including field-level precipitation nowcasting) or NOAA National Digital Forecast
Database (NDFD) API (free, 2.5km resolution, 7-day hourly forecasts).

### Replacing Market Prices

Replace `get_market_prices()` with CME Group DataMine API for real-time futures prices
or USDA NASS Quick Stats API for cash price reporting. Factor in local basis by
connecting to a regional elevator API or manually configuring the basis offset in
`config/settings.py`.

### Replacing the RAG Vector Store

Replace `rag/knowledge_base.KnowledgeBase` with ChromaDB backed by
`sentence-transformers/all-MiniLM-L6-v2` embeddings for production-quality semantic
retrieval. The `retrieve(query, top_k, category)` method signature stays the same.
Add new document chunks by appending `Chunk` objects to `CORPUS` — no other code
changes required.

---

## Run Modes and Quick Start

### Demo Mode (no API key required)

```bash
cd farm_v2
python main.py
```

All four specialists run. LLM analysis sections contain structured demo text built
directly from the threshold computations. All state fields (alerts, irrigation
actions, harvest readiness, market prices) are fully populated with real simulation
data. RAG retrieval runs and prints retrieved document titles to console.

### Live LLM Mode

```bash
pip install langgraph langchain-core langchain-openai python-dotenv
cp .env.example .env
# Add OPENAI_API_KEY=sk-your-key to .env
python main.py
```

All four specialists produce GPT-4o analysis grounded in the RAG-retrieved agronomic
documents. The supervisor synthesizes a full farm operations narrative.

---

## Sample Output

```
AI FARM MANAGEMENT AGENT SYSTEM v2
Architecture: Hierarchical Multi-Agent + RAG
Farm: AgriSense Farm | Central Valley, CA
Mode: DEMO (no API key)

[SUPERVISOR] Farm monitoring cycle started — dispatching specialist agents...
  [SOIL AGENT]       Done — 0 alert(s)
  [CROP AGENT]       Done — 2 alert(s)
  [IRRIGATION AGENT] Done — 4 zone action(s)
  [HARVEST AGENT]    Done — 0 zone(s) ready

[SUPERVISOR] Farm report complete. Status: ATTENTION_NEEDED

OVERALL STATUS: ATTENTION_NEEDED

ALERTS (2 total):
  [PEST_HIGH] Zone Z-NORTH (wheat)
  [PEST_HIGH] Zone Z-WEST  (wheat)

IRRIGATION (4 zones, 0 urgent):
  Z-NORTH: MAINTAIN [normal]  — Moisture 64.9% — within range
  Z-SOUTH: MAINTAIN [normal]  — Moisture 35.0% — within range
  Z-EAST:  MAINTAIN [normal]  — Moisture 68.4% — within range
  Z-WEST:  MAINTAIN [normal]  — Moisture 70.2% — within range

MARKET PRICES:
  Wheat $8.40/bu | Corn $4.62/bu | Soybeans $11.93/bu

RAG KNOWLEDGE RETRIEVED:
  [CROP-02] Pest and Disease Intervention Thresholds (score: 0.412)
  [IRR-01]  Irrigation Scheduling Best Practices     (score: 0.388)
```

---
How to Run
1. Clone the repo
2. Create venv
3. Update .env
4. pip install -r requirements.txt
5. python main.py
   
Whats Next
1. Fix logic vs LLM mismatch

Ensure:

LLM respects agent outputs
2. Add confidence scores
"confidence": 0.82
3. Add retry / fallback

If LLM fails → deterministic summary

4. Add logging per agent

Useful for debugging + FinOps

🔧 5. Add cost tracking (very relevant to you)
Soil Agent → $0.002
Crop Agent → $0.003
