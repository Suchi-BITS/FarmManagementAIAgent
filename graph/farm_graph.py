# graph/farm_graph.py
# Farm Management v2 — Hierarchical Multi-Agent LangGraph
#
# Topology:
#   START
#     -> supervisor (init)       sets up routing
#     -> soil_agent              parallel specialist 1
#     -> crop_agent              parallel specialist 2
#     -> irrigation_agent        parallel specialist 3
#     -> harvest_agent           parallel specialist 4
#     -> supervisor (synthesis)  merges all outputs -> END
#
# All four specialists run after supervisor init using sequential edges.
# In a production deployment with true parallelism, replace sequential edges
# with LangGraph parallel fan-out (send_many / map-reduce).

from typing import TypedDict, Any, Optional, List
from langgraph.graph import StateGraph, END

from agents.supervisor_agent import run_supervisor
from agents.specialist_agents import (
    run_soil_agent, run_crop_agent, run_irrigation_agent, run_harvest_agent
)


class FarmState(TypedDict, total=False):
    current_agent:       str
    iteration_count:     int
    emergency_triggered: bool
    # Observation data
    weather_data:    Any
    soil_data:       Any
    crop_data:       Any
    market_prices:   Any
    # Specialist analyses
    soil_analysis:       Optional[str]
    crop_analysis:       Optional[str]
    irrigation_analysis: Optional[str]
    harvest_analysis:    Optional[str]
    # Specialist outputs
    soil_alerts:         List[Any]
    crop_alerts:         List[Any]
    irrigation_actions:  List[Any]
    harvest_ready:       List[Any]
    harvest_upcoming:    List[Any]
    all_alerts:          List[Any]
    # Final outputs
    farm_report:    Optional[str]
    overall_status: Optional[str]


def _route_from_supervisor(state: FarmState) -> str:
    agent = state.get("current_agent", "complete")
    if agent == "specialists":
        return "soil_agent"
    return END


def build_farm_graph():
    g = StateGraph(FarmState)

    g.add_node("supervisor",        run_supervisor)
    g.add_node("soil_agent",        run_soil_agent)
    g.add_node("crop_agent",        run_crop_agent)
    g.add_node("irrigation_agent",  run_irrigation_agent)
    g.add_node("harvest_agent",     run_harvest_agent)

    g.set_entry_point("supervisor")

    # Supervisor routes to specialists or END
    g.add_conditional_edges(
        "supervisor",
        _route_from_supervisor,
        {"soil_agent": "soil_agent", END: END},
    )

    # Sequential specialist pipeline
    g.add_edge("soil_agent",       "crop_agent")
    g.add_edge("crop_agent",       "irrigation_agent")
    g.add_edge("irrigation_agent", "harvest_agent")
    g.add_edge("harvest_agent",    "supervisor")

    return g.compile()
