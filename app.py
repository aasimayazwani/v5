# app.py — Modular Supervisor SQL Agent (LangGraph + Streamlit + LangChain)

import os
from pathlib import Path
from typing import TypedDict, Optional, List

import pandas as pd
import streamlit as st
from jinja2 import Template

from langchain_community.utilities.sql_database import SQLDatabase
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph

# ─────────────────────────── Configuration ────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("Missing OPENAI_API_KEY in environment variables.")

DB_PATH = "vehicles.db"
db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)

# ───────────────────── Prompt-loading helpers ─────────────────────────
def load_modular_system_prompt(folder: str) -> str:
    folder_path = Path(folder)
    if not folder_path.exists():
        raise FileNotFoundError(f"Prompt folder not found: {folder_path.resolve()}")
    return "\n\n".join(
        p.read_text("utf-8").strip() for p in sorted(folder_path.glob("*.*"))
    )

def render_modular_prompt(folder: str, **kwargs) -> str:
    raw_prompt = load_modular_system_prompt(folder)
    return Template(raw_prompt).render(**kwargs)

# ─────────────────────────── LangGraph state ──────────────────────────
class SQLAgentState(TypedDict):
    user_query: str
    selected_tables: Optional[List[str]]
    generated_sql: Optional[str]
    sql_result: Optional[List[dict]]
    final_answer: Optional[str]
    error: Optional[str]

# ───────────────────────────── Agent nodes ────────────────────────────
def supervisor_agent(state: SQLAgentState) -> str:
    if state.get("error"):
        return "end"
    if not state.get("selected_tables"):
        return "table_selection"
    if not state.get("generated_sql"):
        return "sql_generation"
    if state.get("sql_result") is None:
        return "sql_execution"
    if not state.get("final_answer"):
        return "formatting"
    return "end"

def table_selection_agent(state: SQLAgentState) -> SQLAgentState:
    user_q = state["user_query"].lower()
    table_infos = db.get_table_info()
    matches = [
        t for t in table_infos if any(word in t.lower() for word in user_q.split())
    ]
    if not matches:
        matches = [t.split(":")[0] for t in table_infos]
    return {**state, "selected_tables": matches}

def sql_generation_agent(state: SQLAgentState) -> SQLAgentState:
    try:
        prompt = render_modular_prompt(
            "prompts/sql_generator",
            user_query=state["user_query"],
            table_list=", ".join(state["selected_tables"]),
        )
        response = llm.invoke(prompt)
        sql_text = response.content.strip()  # extract string, not full object

        return {
            "user_query": state["user_query"],
            "selected_tables": state["selected_tables"],
            "generated_sql": sql_text,
            "sql_result": None,
            "final_answer": None,
            "error": None,
        }
    except Exception as e:
        return {**state, "error": str(e)}

def sql_execution_agent(state: SQLAgentState) -> SQLAgentState:
    sql = state["generated_sql"]
    if not sql or "select" not in sql.lower():
        return {**state, "error": "Refused non-SELECT statement."}
    try:
        df = db.run(sql, fetch="pandas")
        records = df.to_dict(orient="records")  # ✅ safe
        return {
            "user_query": state["user_query"],
            "selected_tables": state["selected_tables"],
            "generated_sql": state["generated_sql"],
            "sql_result": records,  # ✅ not a DataFrame!
            "final_answer": None,
            "error": None,
        }
    except Exception as e:
        return {**state, "error": str(e)}

def formatting_agent(state: SQLAgentState) -> SQLAgentState:
    records = state["sql_result"]
    if not records:
        return {**state, "final_answer": "🚫 **No data returned for this query.**"}

    df = pd.DataFrame(records)
    answer_md = df.to_markdown(index=False)
    return {**state, "final_answer": answer_md}

# ───────────────────────── LangGraph assembly ─────────────────────────
def build_graph():
    builder = StateGraph(SQLAgentState)
    builder.add_node("supervisor", supervisor_agent)
    builder.add_node("table_selection", table_selection_agent)
    builder.add_node("sql_generation", sql_generation_agent)
    builder.add_node("sql_execution", sql_execution_agent)
    builder.add_node("formatting", formatting_agent)

    builder.set_entry_point("supervisor")
    builder.add_edge("supervisor", "table_selection")
    builder.add_edge("table_selection", "supervisor")
    builder.add_edge("sql_generation", "supervisor")
    builder.add_edge("sql_execution", "supervisor")
    builder.add_edge("formatting", "supervisor")
    builder.set_finish_point("supervisor")
    return builder.compile()

graph = build_graph()

# ─────────────────────────── Streamlit UI ─────────────────────────────
st.set_page_config(page_title="Supervisor SQL Agent", layout="wide")
st.title("🚦 Supervisor SQL Agent (LangGraph + Streamlit)")

q = st.text_input(
    "Ask a database question:",
    placeholder="e.g. What is the SOC of all EVs in service?",
)

if st.button("Run") and q:
    with st.spinner("Thinking…"):
        init_state: SQLAgentState = {
            "user_query": q,
            "selected_tables": None,
            "generated_sql": None,
            "sql_result": None,
            "final_answer": None,
            "error": None,
        }
        final_state = graph.invoke(init_state)

    if final_state.get("error"):
        st.error(final_state["error"])
    else:
        st.markdown(final_state["final_answer"])
