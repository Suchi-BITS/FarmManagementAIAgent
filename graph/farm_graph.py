# graph/farm_graph.py
# LangGraph graph definition for the AI Farm Management System

from typing import Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from data.models import AgentState
from agents.weather_agent import run_weather_agent
from agents.soil_agent import run_soil_agent
from agents.crop_growth_agent import run_crop_growth_agent
from agents.irrigation_agent import run_irrigation_agent
from agents.planting_agent import run_planting_agent
from agents.harvest_agent import run_harvest_agent
from agents.supervisor_agent import run_supervisor_agent


def route_from_supervisor(state: AgentState) -> Literal[
    "weather_agent", "soil_agent", "crop_growth_agent",
    "irrigation_planner", "planting_agent", "harvest_agent", "__end__"
]:
    """
    Routing function that determines which agent to invoke next
    based on the current state from the supervisor.
    """
    next_agent = state.current_agent

    routing_map = {
        "weather_agent": "weather_agent",
        "soil_agent": "soil_agent",
        "crop_growth_agent": "crop_growth_agent",
        "irrigation_planner": "irrigation_planner",
        "planting_agent": "planting_agent",
        "harvest_agent": "harvest_agent",
        "complete": "__end__",
    }

    return routing_map.get(next_agent, "__end__")


def route_from_agent(state: AgentState) -> Literal[
    "supervisor", "soil_agent", "crop_growth_agent",
    "irrigation_planner", "planting_agent", "harvest_agent"
]:
    """
    After each monitoring/planning agent, determine next step.
    Most agents flow sequentially; supervisor does final synthesis.
    """
    next_agent = state.current_agent
    routing_map = {
        "soil_agent": "soil_agent",
        "crop_growth_agent": "crop_growth_agent",
        "irrigation_planner": "irrigation_planner",
        "planting_agent": "planting_agent",
        "harvest_agent": "harvest_agent",
        "supervisor": "supervisor",
        "complete": "supervisor"
    }
    return routing_map.get(next_agent, "supervisor")


def build_farm_graph() -> StateGraph:
    """
    Build and compile the LangGraph state graph for the farm management system.

    Graph flow:
    supervisor (init) 
        -> weather_agent 
        -> soil_agent 
        -> crop_growth_agent 
        -> irrigation_planner 
        -> planting_agent 
        -> harvest_agent 
        -> supervisor (synthesis) 
        -> END

    The supervisor routes dynamically based on what analysis/planning is needed.
    """
    # Convert AgentState pydantic model to a dict-based state for LangGraph
    # LangGraph works with TypedDict or dict schemas
    from typing import TypedDict, Optional, Any

    # Build the graph using a dict-based state (LangGraph requirement)
    # We pass AgentState as a dict and reconstruct as needed

    workflow = StateGraph(dict)

    # Add all agent nodes
    def supervisor_node(state: dict) -> dict:
        agent_state = AgentState(**state)
        result = run_supervisor_agent(agent_state)
        return result.model_dump()

    def weather_node(state: dict) -> dict:
        agent_state = AgentState(**state)
        result = run_weather_agent(agent_state)
        return result.model_dump()

    def soil_node(state: dict) -> dict:
        agent_state = AgentState(**state)
        result = run_soil_agent(agent_state)
        return result.model_dump()

    def crop_growth_node(state: dict) -> dict:
        agent_state = AgentState(**state)
        result = run_crop_growth_agent(agent_state)
        return result.model_dump()

    def irrigation_node(state: dict) -> dict:
        agent_state = AgentState(**state)
        result = run_irrigation_agent(agent_state)
        return result.model_dump()

    def planting_node(state: dict) -> dict:
        agent_state = AgentState(**state)
        result = run_planting_agent(agent_state)
        return result.model_dump()

    def harvest_node(state: dict) -> dict:
        agent_state = AgentState(**state)
        result = run_harvest_agent(agent_state)
        return result.model_dump()

    # Register nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("weather_agent", weather_node)
    workflow.add_node("soil_agent", soil_node)
    workflow.add_node("crop_growth_agent", crop_growth_node)
    workflow.add_node("irrigation_planner", irrigation_node)
    workflow.add_node("planting_agent", planting_node)
    workflow.add_node("harvest_agent", harvest_node)

    # Set entry point
    workflow.set_entry_point("supervisor")

    # Define edges from supervisor (conditional routing)
    workflow.add_conditional_edges(
        "supervisor",
        lambda state: state["current_agent"],
        {
            "weather_agent": "weather_agent",
            "complete": END
        }
    )

    # Sequential pipeline: each monitoring agent flows to the next
    workflow.add_edge("weather_agent", "soil_agent")
    workflow.add_edge("soil_agent", "crop_growth_agent")
    workflow.add_edge("crop_growth_agent", "irrigation_planner")
    workflow.add_edge("irrigation_planner", "planting_agent")
    workflow.add_edge("planting_agent", "harvest_agent")
    workflow.add_edge("harvest_agent", "supervisor")

    # Compile with memory for conversation persistence
    memory = MemorySaver()
    compiled = workflow.compile(checkpointer=memory)

    return compiled


def get_graph_visualization() -> str:
    """
    Return a text representation of the graph structure.
    """
    return """
    FARM MANAGEMENT AI AGENT GRAPH
    ================================
    
    [ENTRY]
        |
        v
    [SUPERVISOR] -----> [WEATHER AGENT]
        ^                    |
        |                    v
    [HARVEST AGENT]    [SOIL AGENT]
        ^                    |
        |                    v
    [PLANTING AGENT]   [CROP GROWTH AGENT]
        ^                    |
        |                    v
        +------- [IRRIGATION PLANNER]
        
    Final: SUPERVISOR (synthesis) --> [END]
    
    Agents:
    - Supervisor:        Orchestrates flow, synthesizes final report
    - Weather Agent:     Monitors forecasts, identifies weather risks
    - Soil Agent:        Reads soil sensors, detects moisture/nutrient issues
    - Crop Growth Agent: Tracks growth stages, pest/disease monitoring
    - Irrigation Planner: Creates irrigation schedules based on all data
    - Planting Agent:    Determines optimal planting windows
    - Harvest Agent:     Schedules harvest operations and weather windows
    """
