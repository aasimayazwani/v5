# app.py — Modular Supervisor SQL Agent (LangGraph + Streamlit + LangChain)

import json
import os
from pathlib import Path
from typing import List, TypedDict

import pandas as pd
import streamlit as st
from jinja2 import Template
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

# ─────────────────────── Configuration ───────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("Missing OPENAI_API_KEY environment variable.")

DB_PATH = "vehicles.db"                       # ensure this file exists
db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)

# ─────────────────── Prompt-loading helpers ───────────────────
def load_modular_system_prompt(folder: str) -> str:
    fp = Path(folder)
    if not fp.exists():
        raise FileNotFoundError(fp.resolve())
    return "\n\n".join(p.read_text("utf-8").strip() for p in sorted(fp.glob("*.*")))

def render_modular_prompt(folder: str, **kw) -> str:
    return Template(load_modular_system_prompt(folder)).render(**kw)

# ─────────────────────── LangGraph state ──────────────────────
class SQLAgentState(TypedDict, total=False):
    user_query: str
    selected_tables: List[str]
    generated_sql: str
    sql_result: List[dict]
    final_answer: str
    error: str

# ───────────────────────── Agent nodes ────────────────────────
def supervisor_agent(_: SQLAgentState) -> SQLAgentState:
    """Pass-through; routing handled by add_conditional_edges."""
    return {}

def table_selection_agent(state: SQLAgentState) -> SQLAgentState:
    q = state["user_query"].lower()
    tables = db.get_table_info()
    matches = [t for t in tables if any(word in t.lower() for word in q.split())] or [
        t.split(":")[0] for t in tables
    ]
    return {"selected_tables": matches}

def sql_generation_agent(state: SQLAgentState) -> SQLAgentState:
    try:
        prompt = render_modular_prompt(
            "prompts/sql_generator",
            user_query=state["user_query"],
            table_list=", ".join(state["selected_tables"]),
        )
        sql_text = llm.invoke(prompt).content.strip()
        return {"generated_sql": sql_text}
    except Exception as e:
        return {"error": str(e)}

import re

def sql_generation_agent(state: SQLAgentState) -> SQLAgentState:
    try:
        prompt = render_modular_prompt(
            "prompts/sql_generator",
            user_query=state["user_query"],
            table_list=", ".join(state["selected_tables"]),
        )
        response = llm.invoke(prompt)
        raw = response.content.strip()

        # ✅ Extract the first SQL SELECT query from the model output
        match = re.search(r"(?is)\bselect\b[\s\S]+?;", raw)
        if not match:
            return {"error": "No valid SQL SELECT statement found in LLM output."}

        sql_text = match.group(0).strip()

        return {
            "user_query": state["user_query"],
            "selected_tables": state["selected_tables"],
            "generated_sql": sql_text,
            "sql_result": None,
            "final_answer": None,
            "error": None,
        }
    except Exception as e:
        return {"error": str(e)}

def formatting_agent(state: SQLAgentState) -> SQLAgentState:
    records = state.get("sql_result", [])
    if not records:
        return {"final_answer": "🚫 **No data returned for this query.**"}
    md = pd.DataFrame(records).to_markdown(index=False)
    return {"final_answer": md}

# ───────────────────── Graph assembly ────────────────────────
def build_graph():
    g = StateGraph(SQLAgentState)

    # nodes
    g.add_node("supervisor", supervisor_agent)
    g.add_node("table_selection", table_selection_agent)
    g.add_node("sql_generation", sql_generation_agent)
    g.add_node("sql_execution", sql_execution_agent)
    g.add_node("formatting", formatting_agent)

    # return-to-supervisor edges
    g.add_edge("table_selection", "supervisor")
    g.add_edge("sql_generation", "supervisor")
    g.add_edge("sql_execution", "supervisor")
    g.add_edge("formatting", "supervisor")

    # routing logic
    def _route(state: SQLAgentState) -> str:
        if state.get("error"):
            return "end"
        if "selected_tables" not in state:
            return "table_selection"
        if "generated_sql" not in state:
            return "sql_generation"
        if "sql_result" not in state:
            return "sql_execution"
        if "final_answer" not in state:
            return "formatting"
        return "end"

    g.add_conditional_edges(
        "supervisor",
        _route,
        {
            "table_selection": "table_selection",
            "sql_generation": "sql_generation",
            "sql_execution": "sql_execution",
            "formatting": "formatting",
            "end": END,
        },
    )

    g.set_entry_point("supervisor")
    g.set_finish_point("supervisor")   # reached only if _route returns "end"
    return g.compile()

graph = build_graph()

# ─────────────────────── Streamlit UI ────────────────────────
st.set_page_config(page_title="Supervisor SQL Agent", layout="wide")
st.title("🚦 Supervisor SQL Agent (LangGraph + Streamlit)")

question = st.text_input("Ask a database question:",
                          placeholder="e.g. What is the SOC of all EVs in service?")

if st.button("Run") and question:
    with st.spinner("Thinking…"):
        init_state: SQLAgentState = {"user_query": question}
        final_state = graph.invoke(init_state)

    if final_state.get("error"):
        st.error(final_state["error"])
    else:
        st.markdown(final_state["final_answer"])
