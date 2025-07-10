"""app.py — Robust LangGraph SQL Chatbot (v3)
────────────────────────────────────────────────────────
• Opens DB in **read‑only mode** (`?mode=ro`) to bypass WAL issues
• Auto‑deletes stray WAL/SHM files (optional) before connecting
• Logs Python vs. system SQLite versions for diagnostics
• Tool selection by name (order‑proof)
• Smarter planner + fallback remain unchanged
• Works in Streamlit UI *and* CLI
────────────────────────────────────────────────────────
Required libs (see requirements.txt):
  langchain>=0.3.26,<0.4.0
  langchain-core>=0.3.66,<1.0.0
  langchain-community>=0.0.28,<1.0.0
  langgraph>=0.5.2,<1.0.0
  langchain-openai>=0.1.17,<1.0.0
  openai>=1.15.0,<2.0.0
  streamlit>=1.35.0
  sqlalchemy>=2.0.30, pandas>=2.2.0, pysqlite3-binary>=0.5.2 (if needed)
"""

from __future__ import annotations
import os, re, sys
from pathlib import Path
from typing import List, Optional, TypedDict

import sqlite3, sqlalchemy
import streamlit as st
from sqlalchemy import text
from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda, RunnableWithFallbacks
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import AgentExecutor, create_sql_agent

# ─── Configuration ────────────────────────────────────────────────
DB_PATH = os.getenv("DB_PATH", "vehicles.db")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("Missing OPENAI_API_KEY env var")

# Absolute path & read‑only URI (avoids write‑lock / WAL headaches)
db_abs = Path(DB_PATH).expanduser().resolve()
DB_URI = f"sqlite:///{db_abs}?mode=ro"  # read‑only → still fine for schema queries

# Diagnostic prints (appear in Streamlit terminal or CLI)
print("Python SQLite runtime:", sqlite3.sqlite_version, file=sys.stderr)
print("SQLAlchemy version   :", sqlalchemy.__version__, file=sys.stderr)
print("Connecting URI       :", DB_URI, file=sys.stderr)

# Optional: clean up orphaned WAL/SHM files (safe if no writers)
for suffix in ("-wal", "-shm"):
    p = db_abs.with_suffix(db_abs.suffix + suffix)
    if p.exists():
        try:
            p.unlink()
            print("Removed stale", p.name, file=sys.stderr)
        except OSError as e:
            print("Could not remove", p.name, "→", e, file=sys.stderr)

# ─── Open DB with safe flags ───────────────────────────────────────
_engine_args = {
    "connect_args": {
        "check_same_thread": False,  # LangChain may spawn threads
        "uri": True                 # allow query‑string in URI
    }
}

try:
    db = SQLDatabase.from_uri(DB_URI, engine_args=_engine_args)
    # Quick integrity sanity‑check
    with db._engine.begin() as conn:
        ok = conn.execute(text("PRAGMA quick_check")).scalar()
        if ok != "ok":
            raise RuntimeError(f"quick_check failed: {ok}")
    VALID_TABLES: List[str] = [t.lower() for t in db.get_usable_table_names()]
except Exception as e:
    msg = (
        f"❌ SQLite connect failed @ {DB_URI}\n{type(e).__name__}: {e}\n"  # noqa: E501
        "Suggestions:\n"
        "  • Ensure no other process has the DB open with a write lock.\n"
        "  • If you recently upgraded SQLite, install `pysqlite3-binary` to match.\n"
    )
    if st.runtime.exists():
        st.error(msg)
        st.stop()
    else:
        sys.stderr.write(msg)
        sys.exit(1)

# ─── LLM & Toolkit ────────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.7, api_key=OPENAI_API_KEY)

toolkit = SQLDatabaseToolkit(db=db, llm=llm)
all_tools = toolkit.get_tools()
list_tables_tool = next(t for t in all_tools if t.name == "sql_db_list_tables")
schema_tool = next(t for t in all_tools if t.name == "sql_db_schema")

@tool
def db_query_tool(query: str) -> str:
    """Run a raw SQL query against the SQLite database and return its results."""
    result = db.run_no_throw(query)
    return str(result) if result else "Error: Query failed."

# ─── Agent State ──────────────────────────────────────────────────
class AgentState(TypedDict):
    query: str
    sql_result: Optional[str]
    schema_help: Optional[str]
    final_response: Optional[str]

# ─── Helpers ------------------------------------------------------
_word_re = re.compile(r"\b\w+\b")

def extract_table(text: str) -> Optional[str]:
    for w in reversed(_word_re.findall(text.lower())):
        if w in VALID_TABLES:
            return w
    return None

# ─── Planner Node -------------------------------------------------

def planner_node(state: AgentState) -> str:
    q = state["query"].lower()
    triggers = ("table", "column", "schema", "structure", "describe", "list", "database")
    return "schema_agent" if any(tok in q for tok in triggers) else "sql_agent"

# ─── SQL Agent Node ----------------------------------------------
sql_exec: AgentExecutor = create_sql_agent(llm=llm, toolkit=toolkit, verbose=False)

def fallback_handler(state):
    return {"sql_result": "Error: query execution failed."}

sql_node = RunnableWithFallbacks(
    runnable=sql_exec,
    fallbacks=[RunnableLambda(fallback_handler)],
)

def run_sql_agent(state: AgentState) -> AgentState:
    try:
        res = sql_node.invoke({"input": state["query"]})
        # Extract the actual result from the agent response
        if hasattr(res, 'get') and 'output' in res:
            result = res['output']
        else:
            result = str(res)
        return {**state, "sql_result": result}
    except Exception as e:
        return {**state, "sql_result": f"Error: {str(e)}"}

# ─── Schema Agent Node -------------------------------------------

def run_schema_agent(state: AgentState) -> AgentState:
    try:
        q_lower = state["query"].lower()
        if re.search(r"\blist\b|\bshow\b.*\btables?\b", q_lower):
            res = list_tables_tool.invoke("")  # Empty string input
        else:
            tbl = extract_table(state["query"])
            if not tbl:
                return {**state, "schema_help": "❌ No valid table name found."}
            # Pass the table name as a string, not a dictionary
            res = schema_tool.invoke(tbl)
        return {**state, "schema_help": str(res)}
    except Exception as e:
        return {**state, "schema_help": f"Error getting schema: {str(e)}"}

# ─── Summariser Node ---------------------------------------------

def summariser_node(state: AgentState) -> AgentState:
    answer = state.get("sql_result") or state.get("schema_help") or "No output generated."
    return {**state, "final_response": answer}

# ─── Build LangGraph ---------------------------------------------
graph = StateGraph(AgentState)

graph.add_node("planner", RunnableLambda(lambda s: s))
graph.add_node("sql_agent", run_sql_agent)
graph.add_node("schema_agent", run_schema_agent)
graph.add_node("summariser", summariser_node)

# Routing
graph.set_entry_point("planner")
graph.add_conditional_edges("planner", planner_node)

def route_after_sql(state: AgentState) -> str:
    return "schema_agent" if str(state.get("sql_result", "")).startswith("Error") else "summariser"

graph.add_conditional_edges("sql_agent", route_after_sql)
graph.add_edge("schema_agent", "summariser")
graph.add_edge("summariser", END)

app = graph.compile()

# ─── Streamlit UI -------------------------------------------------
if st.runtime.exists():
    st.set_page_config(page_title="🧠 LangGraph SQL Chatbot", layout="centered")
    st.title("🧠 LangGraph SQL Chatbot")

    prompt = st.text_input("Ask a database question:", placeholder="e.g. describe routes")
    if st.button("Run") and prompt:
        with st.spinner("Thinking…"):
            try:
                out_state = app.invoke({"query": prompt})
                st.success("Done!")
                st.subheader("Answer")
                st.markdown(out_state["final_response"])
                with st.expander("🔍 Debug trace"):
                    st.json(out_state)
            except Exception as e:
                st.error(f"Error: {str(e)}")

# ─── CLI fallback -------------------------------------------------
if __name__ == "__main__" and not st.runtime.exists():
    q = " ".join(sys.argv[1:]) or "list tables"
    try:
        res = app.invoke({"query": q})
        print("\nAnswer:\n", res["final_response"])
    except Exception as e:
        print(f"Error: {str(e)}")