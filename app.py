# app.py — Resilient LangGraph SQL Agent w/ Tool Fallbacks and Modular Tooling

import os
import json
import pandas as pd
from pathlib import Path
from typing import Annotated, TypedDict, List, Optional
import streamlit as st
from jinja2 import Template

from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda, RunnableWithFallbacks
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.prompts import ChatPromptTemplate

# -------------------- Config ----------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("Missing OPENAI_API_KEY env var")

DB_PATH = "vehicles.db"
db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")
llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=OPENAI_API_KEY)

# -------------------- Prompt Utils ----------------------
def load_modular_system_prompt(folder: str) -> str:
    folder_path = Path(folder)
    if not folder_path.exists():
        raise FileNotFoundError(f"Prompt folder not found: {folder_path.resolve()}")
    return "\n\n".join(p.read_text("utf-8").strip() for p in sorted(folder_path.glob("*.*")))

def render_modular_prompt(folder: str, **kwargs) -> str:
    raw_prompt = load_modular_system_prompt(folder)
    return Template(raw_prompt).render(**kwargs)

# -------------------- Tools ----------------------
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()

list_tables_tool = next(t for t in tools if t.name == "sql_db_list_tables")
get_schema_tool = next(t for t in tools if t.name == "sql_db_schema")

def db_query_tool(query: str) -> str:
    result = db.run_no_throw(query)
    if not result:
        return "Error: Query failed. Please rewrite your query and try again."
    return str(result)

# ------------------ Query Checker ------------------
query_check_system = """You are a SQL expert with a strong attention to detail.
Double check the SQLite query for common mistakes, including:
- Using NOT IN with NULL values
- Using UNION when UNION ALL should have been used
- Using BETWEEN for exclusive ranges
- Data type mismatch in predicates
- Properly quoting identifiers
- Using the correct number of arguments for functions
- Casting to the correct data type
- Using the proper columns for joins
If there are any of the above mistakes, rewrite the query. If not, return the original.
"""

query_check_prompt = ChatPromptTemplate.from_messages(
    [("system", query_check_system), ("placeholder", "{messages}")]
)
query_check = query_check_prompt | ChatOpenAI(model="gpt-4", temperature=0).bind_tools(
    [db_query_tool], tool_choice="required"
)

# -------------------- Error Handling Wrapper ----------------------
def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\nPlease fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }

def create_tool_node_with_fallback(tools: list) -> RunnableWithFallbacks:
    return ToolNode(tools).with_fallbacks([
        RunnableLambda(handle_tool_error)
    ], exception_key="error")

# -------------------- LangGraph State ----------------------
class SQLAgentState(TypedDict):
    messages: Annotated[list, lambda m: m]  # Used to thread through all tool messages

# -------------------- Graph Construction ----------------------
def build_graph():
    builder = StateGraph(SQLAgentState)

    builder.add_node("list_tables", create_tool_node_with_fallback([list_tables_tool]))
    builder.add_node("get_schema", create_tool_node_with_fallback([get_schema_tool]))
    builder.add_node("query_check", lambda s: {"messages": [query_check.invoke({"messages": s["messages"]})]})
    builder.add_node("execute_query", create_tool_node_with_fallback([db_query_tool]))

    def start_tool_call(state: SQLAgentState):
        return {
            "messages": [
                {
                    "tool_calls": [
                        {"name": "sql_db_list_tables", "args": {}, "id": "call1"}
                    ],
                    "content": "",
                }
            ]
        }

    def should_continue(state: SQLAgentState):
        last_msg = state["messages"][-1]
        if hasattr(last_msg, "tool_calls"):
            return END
        if "Error:" in str(last_msg):
            return "list_tables"
        return "query_check"

    builder.add_node("start", start_tool_call)
    builder.set_entry_point("start")
    builder.add_edge("start", "list_tables")
    builder.add_edge("list_tables", "get_schema")
    builder.add_edge("get_schema", "query_check")
    builder.add_edge("query_check", "execute_query")
    builder.add_edge("execute_query", "get_schema")  # Loop for retries
    builder.set_finish_point("execute_query")

    return builder.compile()

graph = build_graph()

# -------------------- Streamlit UI ----------------------
st.set_page_config(page_title="Supervisor SQL Agent", layout="wide")
st.title("🚦 Supervisor SQL Agent (LangGraph + Streamlit)")

query = st.text_input("Ask a database question:", placeholder="e.g. What is the SOC of all EVs?")

if st.button("Run") and query:
    with st.spinner("Running agent..."):
        init_state = {"messages": [
            {"content": query}
        ]}
        final_state = graph.invoke(init_state)
        st.subheader("🔍 Debug")
        st.json(final_state)

        last = final_state["messages"][-1]
        if hasattr(last, "content"):
            st.markdown(last.content)
        else:
            st.info("No final response.")
