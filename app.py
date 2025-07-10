"""app.py — Robust LangGraph SQL Chatbot (v2)
-------------------------------------------------------------
• Auto‑detect correct toolkit tools by *name* (no brittle indexes)
• Handles missing/locked DB files gracefully
• Smarter routing & automatic schema fallback
• Streamlit UI + CLI fallback
-------------------------------------------------------------
Requirements (see requirements.txt):
  langchain>=0.3.26,<0.4.0
  langchain-core>=0.3.66,<1.0.0
  langchain-community>=0.0.28,<1.0.0
  langchain-openai>=0.1.17,<1.0.0
  langgraph>=0.5.2,<1.0.0
  openai>=1.15.0,<2.0.0
  streamlit>=1.35.0
  sqlalchemy>=2.0.30, pandas>=2.2.0, psycopg2-binary>=2.9.9
"""

from __future__ import annotations
import os, re, sys
from pathlib import Path
from typing import List, Optional, TypedDict

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda, RunnableWithFallbacks
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import AgentExecutor, create_sql_agent

# ─── Configuration ──────────────────────────────────────────────────
DB_PATH = os.getenv("DB_PATH", "vehicles.db")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("Missing OPENAI_API_KEY env var")

DB_URI = f"sqlite:///{Path(DB_PATH).expanduser().resolve()}"

# Attempt to initialise DB -------------------------------------------------
try:
    db = SQLDatabase.from_uri(DB_URI)
    VALID_TABLES: List[str] = [tbl.lower() for tbl in db.get_usable_table_names()]
except Exception as e:
    msg = f"❌ Could not open SQLite DB at {DB_URI}\n{e}"
    if st.runtime.exists():
        st.error(msg)
        st.stop()
    else:
        print(msg, file=sys.stderr)
        sys.exit(1)

# ─── LLM & Toolkit ---------------------------------------------------------
llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=OPENAI_API_KEY)

toolkit = SQLDatabaseToolkit(db=db, llm=llm)
all_tools = toolkit.get_tools()

# Grab the exact tools by name (order‑agnostic) -----------------------------
list_tables_tool = next(t for t in all_tools if t.name == "sql_db_list_tables")
schema_tool = next(t for t in all_tools if t.name == "sql_db_schema")

# (Optional) raw query tool in case you want it later -----------------------
@tool
def db_query_tool(query: str) -> str:
    """Run a raw SQL query against the SQLite database and return its results."""
    result = db.run_no_throw(query)
    return str(result) if result else "Error: Query failed."

# ─── Agent State -----------------------------------------------------------
class AgentState(TypedDict):
    query: str
    sql_result: Optional[str]
    schema_help: Optional[str]
    final_response: Optional[str]

# ─── Helpers ---------------------------------------------------------------
_word_re = re.compile(r"\b\w+\b")

def extract_table(text: str) -> Optional[str]:
    """Return the last token that matches an existing table name."""
    for w in reversed(_word_re.findall(text.lower())):
        if w in VALID_TABLES:
            return w
    return None

# ─── Planner Node ----------------------------------------------------------

def planner_node(state: AgentState) -> str:
    q = state["query"].lower()
    schema_triggers = ("table", "column", "schema", "structure", "describe", "list", "database")
    return "schema_agent" if any(tok in q for tok in schema_triggers) else "sql_agent"

# ─── SQL Agent Node --------------------------------------------------------
sql_executor: AgentExecutor = create_sql_agent(llm=llm, toolkit=toolkit, verbose=False)

sql_node = RunnableWithFallbacks(
    runnable=sql_executor,
    fallbacks=[RunnableLambda(lambda _: ToolMessage(tool_call_id="sql_query_tool", content="Error: query execution failed."))],
)

def run_sql_agent(state: AgentState) -> AgentState:
    result = sql_node.invoke({"input": state["query"]})
    return {**state, "sql_result": str(result)}

# ─── Schema Agent Node -----------------------------------------------------

def run_schema_agent(state: AgentState) -> AgentState:
    q_lower = state["query"].lower()
    if re.search(r"\blist\b|\bshow\b.*\btables?\b", q_lower):
        res = list_tables_tool.invoke({})
    else:
        tbl = extract_table(state["query"])
        if not tbl:
            return {**state, "schema_help": "❌ No valid table name found."}
        res = schema_tool.invoke({"table_names": [tbl]})
    return {**state, "schema_help": str(res)}

# ─── Summariser Node -------------------------------------------------------

def summariser_node(state: AgentState) -> AgentState:
    answer = state.get("sql_result") or state.get("schema_help") or "No output generated."
    return {**state, "final_response": answer}

# ─── LangGraph Build -------------------------------------------------------
graph = StateGraph(AgentState)

graph.add_node("planner", RunnableLambda(lambda s: s))
graph.add_node("sql_agent", run_sql_agent)
graph.add_node("schema_agent", run_schema_agent)
graph.add_node("summariser", summariser_node)

# Entry & edges
graph.set_entry_point("planner")
graph.add_conditional_edges("planner", planner_node)

def route_after_sql(state: AgentState) -> str:
    return "schema_agent" if str(state.get("sql_result", "")).startswith("Error") else "summariser"

graph.add_conditional_edges("sql_agent", route_after_sql)
graph.add_edge("schema_agent", "summariser")
graph.add_edge("summariser", END)

app = graph.compile()

# ─── Streamlit UI ----------------------------------------------------------
st.set_page_config(page_title="🧠 LangGraph SQL Chatbot", layout="centered")
st.title("🧠 LangGraph SQL Chatbot")

user_prompt = st.text_input("Ask a database question:", placeholder="e.g. describe routes")
if st.button("Run") and user_prompt:
    with st.spinner("Thinking…"):
        state_out = app.invoke({"query": user_prompt})
    st.success("Done!")
    st.subheader("Answer")
    st.markdown(state_out["final_response"])
    with st.expander("🔍 Debug trace"):
        st.json(state_out)

# ─── CLI fallback ----------------------------------------------------------
if __name__ == "__main__" and not st.runtime.exists():
    q = " ".join(sys.argv[1:]) or "list tables"
    res = app.invoke({"query": q})
    print("\nAnswer:\n", res["final_response"])
