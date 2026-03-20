#!/usr/bin/env python3
# main.py — Farm Management AI Agent v2
# Architecture: Hierarchical Multi-Agent (Supervisor + 4 Specialists) + RAG
#
# Usage:
#   python main.py

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime
from agents.base import _demo_mode
from config.settings import farm_config

def make_initial_state() -> dict:
    return {
        "current_agent": None, "iteration_count": 0, "emergency_triggered": False,
        "weather_data": None, "soil_data": None, "crop_data": None,
        "soil_analysis": None, "crop_analysis": None,
        "irrigation_analysis": None, "harvest_analysis": None,
        "soil_alerts": [], "crop_alerts": [], "irrigation_actions": [],
        "harvest_ready": [], "harvest_upcoming": [], "all_alerts": [],
        "farm_report": None, "overall_status": None,
    }

def print_report(state: dict) -> None:
    print("\n" + "=" * 70)
    print("  FARM MANAGEMENT REPORT")
    print(f"  {farm_config.farm_name} | {farm_config.farm_location}")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Mode: {'DEMO' if _demo_mode() else 'LIVE LLM'}")
    print("=" * 70)

    print(f"\nOVERALL STATUS: {(state.get('overall_status') or 'unknown').upper()}")

    alerts = state.get("all_alerts", [])
    print(f"\nALERTS ({len(alerts)} total):")
    for a in alerts:
        print(f"  [{a.get('type','?').upper()}] Zone {a.get('zone','?')}")

    irr = state.get("irrigation_actions", [])
    urgent_irr = [a for a in irr if a.get("priority") in ("critical", "high")]
    print(f"\nIRRIGATION ({len(irr)} zones, {len(urgent_irr)} urgent):")
    for a in irr:
        print(f"  {a['zone']}: {a['action'].upper()} [{a['priority']}] — {a['reason']}")

    harvest = state.get("harvest_ready", [])
    print(f"\nHARVEST READY ({len(harvest)} zone(s)):")
    for h in harvest:
        print(f"  {h['zone']} ({h['crop']}): {h['days_to_harvest']}d — {h['urgency'].upper()}")

    prices = state.get("market_prices", {})
    if prices:
        print(f"\nMARKET PRICES:")
        print(f"  Wheat ${prices.get('wheat_usd_per_bushel','?')}/bu | "
              f"Corn ${prices.get('corn_usd_per_bushel','?')}/bu | "
              f"Soybeans ${prices.get('soybeans_usd_per_bushel','?')}/bu")

    report = state.get("farm_report", "")
    if report:
        print("\nFARM REPORT:")
        print("-" * 50)
        print(report[:1200])

    print("\n" + "=" * 70)
    print(f"  {farm_config.disclaimer}")
    print("=" * 70)

def main():
    print("=" * 70)
    print("  AI FARM MANAGEMENT AGENT SYSTEM v2")
    print(f"  Architecture: Hierarchical Multi-Agent + RAG")
    print(f"  Farm: {farm_config.farm_name} | {farm_config.farm_location}")
    print(f"  Mode: {'DEMO (no API key)' if _demo_mode() else 'LIVE — GPT-4o'}")
    print("=" * 70)

    try:
        from graph.farm_graph import build_farm_graph
        graph = build_farm_graph()
        state = make_initial_state()
        result = graph.invoke(state)
    except ImportError:
        print("\n[INFO] LangGraph not installed — running agents directly...")
        from agents.supervisor_agent import run_supervisor
        from agents.specialist_agents import (
            run_soil_agent, run_crop_agent, run_irrigation_agent, run_harvest_agent
        )
        state = make_initial_state()
        state = run_supervisor(state)
        state = run_soil_agent(state)
        state = run_crop_agent(state)
        state = run_irrigation_agent(state)
        state = run_harvest_agent(state)
        result = run_supervisor(state)

    print_report(result)

if __name__ == "__main__":
    main()
