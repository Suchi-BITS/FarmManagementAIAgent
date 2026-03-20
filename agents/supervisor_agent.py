# agents/supervisor_agent.py
# Farm Supervisor Agent — orchestrates specialists and synthesises final report.
#
# Hierarchical pattern:
#   supervisor (init) -> all 4 specialists run -> supervisor (synthesis) -> END
#
# The supervisor does NOT run the specialists in sequence itself — LangGraph
# handles that via parallel fan-out edges (see graph/farm_graph.py).
# The supervisor's second call receives all specialist outputs merged into state
# and synthesises the final farm management report.

import json
from datetime import datetime
from config.settings import farm_config
from rag.knowledge_base import FARM_KB
from agents.base import call_llm


def run_supervisor(state: dict) -> dict:
    """
    Entry point: sets up the initial routing.
    Synthesis: called after all specialists have written their analyses.
    """
    # ── INITIAL CALL: route to specialists ───────────────────────────────────
    if state.get("soil_analysis") is None:
        print("\n[SUPERVISOR] Farm monitoring cycle started — dispatching specialist agents...")
        state["current_agent"]    = "specialists"
        state["iteration_count"]  = state.get("iteration_count", 0) + 1
        return state

    # ── SYNTHESIS CALL: all specialists done ─────────────────────────────────
    print("\n[SUPERVISOR] All specialists complete — synthesising farm report...")

    soil_alerts  = state.get("soil_alerts", [])
    crop_alerts  = state.get("crop_alerts", [])
    irr_actions  = state.get("irrigation_actions", [])
    harvest_rdy  = state.get("harvest_ready", [])
    prices       = state.get("market_prices", {})

    all_alerts = soil_alerts + crop_alerts
    critical   = [a for a in all_alerts if a.get("type", "").startswith("moisture_critical")
                  or a.get("type") == "pest_high"]

    # RAG: retrieve integrated decision context for supervisor
    rag_context = FARM_KB.retrieve_text(
        "farm status critical alert priority action irrigation harvest nitrogen", top_k=2
    )

    if state.get("emergency_triggered") or len(critical) >= 3:
        overall_status = "critical"
    elif len(all_alerts) >= 2:
        overall_status = "attention_needed"
    elif not all_alerts:
        overall_status = "optimal"
    else:
        overall_status = "normal"

    demo = (
        f"FARM STATUS: {overall_status.upper()}\n"
        f"Farm: {farm_config.farm_name} | {farm_config.farm_location} | "
        f"{farm_config.farm_area_acres} acres\n"
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        f"EXECUTIVE SUMMARY:\n"
        f"Farm monitoring cycle complete. {len(all_alerts)} total alert(s) across all zones. "
        f"Soil: {len(soil_alerts)} issue(s). "
        f"Crop: {len(crop_alerts)} issue(s). "
        f"Irrigation: {sum(1 for a in irr_actions if a['action'] in ('irrigate','stop'))} "
        f"urgent action(s). "
        f"Harvest: {len(harvest_rdy)} zone(s) ready.\n\n"
        f"SOIL ANALYSIS:\n{state.get('soil_analysis','N/A')[:300]}\n\n"
        f"CROP ANALYSIS:\n{state.get('crop_analysis','N/A')[:300]}\n\n"
        f"IRRIGATION PLAN:\n{state.get('irrigation_analysis','N/A')[:300]}\n\n"
        f"HARVEST ASSESSMENT:\n{state.get('harvest_analysis','N/A')[:300]}\n\n"
        f"MARKET PRICES:\n"
        f"  Wheat: ${prices.get('wheat_usd_per_bushel','N/A')}/bu | "
        f"Corn: ${prices.get('corn_usd_per_bushel','N/A')}/bu | "
        f"Soybeans: ${prices.get('soybeans_usd_per_bushel','N/A')}/bu\n\n"
        f"TOP 3 PRIORITY ACTIONS TODAY:\n"
        + "\n".join(
            f"  {i+1}. [{a.get('type','alert').upper()}] Zone {a['zone']}"
            for i, a in enumerate(sorted(all_alerts, key=lambda x:
                0 if "critical" in x.get("type","") else 1)[:3])
        ) + f"\n\n{farm_config.disclaimer}"
    )

    summary = call_llm(
        system_prompt=(
            f"You are the Farm Management Supervisor AI for {farm_config.farm_name}.\n"
            f"Synthesise all specialist analyses into a prioritised farm operations report.\n\n"
            f"RETRIEVED AGRONOMIC KNOWLEDGE:\n{rag_context}"
        ),
        user_prompt=(
            f"SOIL ANALYSIS:\n{state.get('soil_analysis','N/A')}\n\n"
            f"CROP ANALYSIS:\n{state.get('crop_analysis','N/A')}\n\n"
            f"IRRIGATION PLAN:\n{state.get('irrigation_analysis','N/A')}\n\n"
            f"HARVEST ASSESSMENT:\n{state.get('harvest_analysis','N/A')}\n\n"
            f"ALL ALERTS: {json.dumps(all_alerts)}\n"
            f"MARKET PRICES: {json.dumps(prices)}\n\n"
            f"Produce: overall status, executive summary, top 3 priority actions today, "
            f"this week's operations calendar, and any cross-agent conflicts."
        ),
        demo_response=demo,
    )

    state["farm_report"]    = summary
    state["overall_status"] = overall_status
    state["all_alerts"]     = all_alerts
    state["current_agent"]  = "complete"
    print(f"[SUPERVISOR] Farm report complete. Status: {overall_status.upper()}")
    return state
