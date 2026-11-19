"""Dubai AI Travel Planner — temporary project overview page.

Placeholder for the public deployment until the planner UI (plan Task 13) replaces this file.
"""

from pathlib import Path

import streamlit as st

DOCS = Path(__file__).parent / "Project Context"

WORKFLOW = """
User Travel Request
        |
Parse Requirements
        |
        v
   Orchestrator
        |
        +-------------------+
        |                   |
        v                   v
 Destination Agent     Logistics Agent
        |                   |
        +---------+---------+
                  |
                  v
             Budget Agent
                  |
                  v
              Review Agent
                  |
           +------+------+
           |             |
         Pass          Revise
           |             |
           v             |
     Final Itinerary <---+
"""

AGENTS = [
    ("Orchestrator", "Parses the request, extracts constraints, coordinates agents, assembles the final itinerary."),
    ("Destination Research", "Finds attractions, restaurants and neighbourhoods that match the traveller's interests."),
    ("Logistics", "Picks the hotel area, estimates travel times and fares, sequences each day geographically."),
    ("Budget", "Estimates accommodation, food, transport and activity costs in AED; proposes cheaper alternatives."),
    ("Review", "Checks budget, schedule feasibility, preferences and duplicates; sends the plan back for revision."),
]


def read_doc(name: str) -> str:
    path = DOCS / name
    return path.read_text(encoding="utf-8") if path.exists() else f"_{name} not found._"


def main() -> None:
    st.set_page_config(page_title="Dubai AI Travel Planner", page_icon="🌴", layout="wide")

    st.title("Dubai AI Travel Planner")
    st.caption("Multi-agent travel planning with LangGraph and CrewAI")
    st.info("🚧 Under construction. The interactive planner will appear here as the build progresses.")

    overview, statement, plan = st.tabs(["Overview", "Problem Statement", "Implementation Plan"])

    with overview:
        st.subheader("Example request")
        st.markdown(
            "> Plan a 4-day trip to Dubai with a budget of AED 5,000, excluding flights. I enjoy cultural "
            "attractions, local food, and beaches. I prefer affordable hotels and want to avoid crowded places."
        )
        left, right = st.columns([3, 2])
        with left:
            st.subheader("Agents")
            for name, role in AGENTS:
                st.markdown(f"**{name}** — {role}")
        with right:
            st.subheader("Workflow")
            st.code(WORKFLOW, language=None)

    with statement:
        st.markdown(read_doc("Problem Statement.md"))

    with plan:
        st.markdown(read_doc("Implementation Plan.md"))


main()  # craftsman-ignore: PY007 (Streamlit executes the script top to bottom)
