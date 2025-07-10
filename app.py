# app.py — Multi‑Agent LangGraph SQL Chatbot (updated)
# ----------------------------------------------------
# This version validates table names against a known list
# extracted from your SQLite database screenshots and
# routes queries either to a schema agent or a SQL agent.

import os
import re
from typing import TypedDict, Optional

from langchain_openai import ChatOpenAI
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_sql_agent
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda

# ---------------------- Config -----------------------
DB_PATH = "vehicles.db"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("Missing OPENAI_API_KEY env var")

llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=OPENAI_API_KEY)

db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")

# Toolkit provides list‑tables and schema‑fetch tools
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
# tools[0] → sql_db_list_tables, tools[1] → sql_db_schema
tools = toolkit.get_tools()

# ---------------- Known tables (from screenshots) ----
VALID_TABLES = [
    "agency",
    "bus_specifications",
    "calendar",
    "calendar_dates",
    "candidates_bus_block_end_soc",
    "fare_attributes",
    "fare_rules",
    "feed_info",
    "frequencies",
    "realtime_ev_soc",
    "realtime_inservice_bus_soc_forecast",
    "realtime_inservice_dispatch_data",
    "routes",
    "shapes",
    "stop_times",
    "stops",
    "transfers",
    "trips",
]

# ---------------- LangGraph state --------------------
class AgentState(TypedDict):
    query: str
    sql_result: Optional[str]
    schema_help: Optional[str]
    final_response: Optional[str]

# ---------------- Helper functions -------------------

def extract_table_name(query: str) -> Optional[str]:
    """Return the last valid table name found in the query, else None."""
    tokens = re.findall(r"\b\w+\b", query.lower())
    for tok in reversed(tokens):
        if tok in VALID_TABLES:
            return tok
    return None

# ---------------- Planner node -----------------------

def planner_node(state: AgentState) -> str:
    q_lower = state["query"].lower()
    if any(w in q_lower for w in ["table", "column", "schema", "structure"]):
        return "schema_agent"
    return "sql_agent"

# ---------------- SQL agent node ---------------------

sql_agent_executor = create_sql_agent(llm=llm, toolkit=toolkit, verbose=False)


def run_sql_agent(state: AgentState) -> AgentState:
    try:
        output = sql_agent_executor.invoke({"input": state["query"]})
        return {**state, "sql_result": str(output)}
    except Exception as e:
        return {**state, "sql_result": f"❌ SQL agent error: {e}"}

# ---------------- Schema agent node ------------------

def run_schema_agent(state: AgentState) -> AgentState:
    q_lower = state["query"].lower()
    if "list" in q_lower:
        result = tools[0].invoke({})  # sql_db_list_tables
    else:
        table_name = extract_table_name(state["query"])
        if not table_name:
            return {**state, "schema_help": "❌ No valid table name found in query."}
        result = tools[1].invoke({"table_names": [table_name]})  # sql_db_schema
    return {**state, "schema_help": str(result)}

# ---------------- Summarizer node --------------------

def summarizer_node(state: AgentState) -> AgentState:
    summary = state.get("sql_result") or state.get("schema_help") or "No output generated."
    return {**state, "final_response": summary}

# ---------------- Build LangGraph --------------------

graph = StateGraph(AgentState)

# Pass‑through planner node; routing logic handled by conditional edges
graph.add_node("planner", RunnableLambda(lambda s: s))
graph.add_conditional_edges("planner", planner_node)

graph.add_node("sql_agent", RunnableLambda(run_sql_agent))
graph.add_node("schema_agent", RunnableLambda(run_schema_agent))
graph.add_node("summarizer", RunnableLambda(summarizer_node))

graph.add_edge("sql_agent", "summarizer")
graph.add_edge("schema_agent", "summarizer")
graph.add_edge("summarizer", END)

app = graph.compile()

# ---------------- Command‑line entry -----------------
if __name__ == "__main__":
    import sys

    user_query = sys.argv[1] if len(sys.argv) > 1 else "List all tables in the database"
    result = app.invoke({"query": user_query})
    print("\n📤 Final Answer:\n", result["final_response"])
