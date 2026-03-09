# main.py
# Main entry point for the AI Farm Management Agent System

import os
import json
import sys
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graph.farm_graph import build_farm_graph, get_graph_visualization
from data.models import AgentState
from config.settings import farm_config


def print_header():
    print("=" * 70)
    print("  AI FARM MANAGEMENT AGENT SYSTEM")
    print(f"  Farm: {farm_config.farm_name}")
    print(f"  Location: {farm_config.farm_location}")
    print(f"  Area: {farm_config.farm_area_acres} acres")
    print(f"  Crops: {', '.join(farm_config.crops)}")
    print(f"  Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


def print_farm_report(final_state: dict):
    """Print formatted final farm management report."""
    print("\n" + "=" * 70)
    print("  FARM MANAGEMENT REPORT")
    print("=" * 70)

    schedule = final_state.get("farm_schedule")
    if not schedule:
        print("No schedule generated.")
        return

    status = schedule.get("overall_farm_status", "unknown").upper().replace("_", " ")
    print(f"\nOVERALL FARM STATUS: {status}")
    print(f"Generated: {schedule.get('generated_at', 'N/A')}")
    print(f"Planning Horizon: {schedule.get('planning_horizon_days')} days")

    print("\n--- EXECUTIVE SUMMARY ---")
    print(schedule.get("summary", "No summary available"))

    # Alerts
    alerts = schedule.get("alerts", [])
    if alerts:
        print(f"\n--- ALERTS ({len(alerts)}) ---")
        for alert in alerts:
            print(f"  ! {alert}")

    # Irrigation actions
    irrigation = schedule.get("irrigation_actions", [])
    if irrigation:
        print(f"\n--- IRRIGATION SCHEDULE ({len(irrigation)} actions) ---")
        for action in irrigation:
            print(f"  Zone {action.get('zone_id')}: {action.get('action')} | "
                  f"{action.get('water_volume_liters', 0):.0f}L | "
                  f"{action.get('duration_minutes', 0)} min | "
                  f"Priority: {action.get('priority')}")

    # Planting actions
    planting = schedule.get("planting_actions", [])
    if planting:
        print(f"\n--- PLANTING SCHEDULE ({len(planting)} operations) ---")
        for action in planting:
            print(f"  Zone {action.get('zone_id')}: {action.get('crop_type')} | "
                  f"Date: {action.get('recommended_date')} | "
                  f"Variety: {action.get('seed_variety')} | "
                  f"Confidence: {action.get('confidence_score', 0):.0%}")

    # Harvest actions
    harvest = schedule.get("harvest_actions", [])
    if harvest:
        print(f"\n--- HARVEST SCHEDULE ({len(harvest)} operations) ---")
        for action in harvest:
            print(f"  Zone {action.get('zone_id')}: {action.get('crop_type')} | "
                  f"Window: {action.get('recommended_harvest_window_start')} - "
                  f"{action.get('recommended_harvest_window_end')} | "
                  f"Yield: {action.get('estimated_yield_tons', 0):.1f}t | "
                  f"Urgency: {action.get('urgency')}")

    print("\n" + "=" * 70)


def save_report(final_state: dict, filename: str = None):
    """Save the farm report to a JSON file."""
    if not filename:
        filename = f"farm_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(filename, "w") as f:
        # Convert state to serializable format
        serializable = {}
        for k, v in final_state.items():
            try:
                json.dumps(v)
                serializable[k] = v
            except (TypeError, ValueError):
                serializable[k] = str(v)
        json.dump(serializable, f, indent=2, default=str)

    print(f"\nReport saved to: {filename}")
    return filename


def run_farm_management_cycle():
    """
    Execute a single farm management cycle.
    In production, this would run on a schedule (e.g., every hour).
    """
    print_header()

    # Validate API key
    if not farm_config.openai_api_key:
        print("\nERROR: OPENAI_API_KEY not set in environment variables.")
        print("Please set it in your .env file or environment.")
        print("\nExample .env file:")
        print("  OPENAI_API_KEY=sk-your-key-here")
        sys.exit(1)

    print(get_graph_visualization())

    print("\nStarting farm management cycle...\n")
    print("-" * 70)

    # Build the LangGraph graph
    graph = build_farm_graph()

    # Initialize state
    initial_state = AgentState().model_dump()

    # Configuration for LangGraph execution
    config = {
        "configurable": {
            "thread_id": f"farm-cycle-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        }
    }

    # Run the agent graph
    final_state = None
    try:
        for step in graph.stream(initial_state, config=config):
            # Each step returns {node_name: state}
            for node_name, state in step.items():
                print(f"  Completed: [{node_name}]")
                final_state = state

    except Exception as e:
        print(f"\nERROR during graph execution: {e}")
        raise

    if final_state:
        print_farm_report(final_state)
        report_file = save_report(final_state)
        return final_state
    else:
        print("No final state produced.")
        return None


if __name__ == "__main__":
    run_farm_management_cycle()
