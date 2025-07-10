# app.py — LangGraph-based Multi-Agent SQL Chatbot  ✅ 2025-07-10
# -------------------------------------------------------------------
# Requirements (see your requirements.txt / pyproject):
#   langchain>=0.3.26,<0.4.0
#   langchain-core>=0.3.66,<1.0.0
#   langchain-community>=0.0.28,<1.0.0
#   langchain-openai>=0.1.17,<1.0.0
#   langgraph>=0.5.2,<1.0.0
#   openai>=1.15.0,<2.0.0
#   streamlit>=1.35.0
#   sqlalchemy>=2.0.30, pandas>=2.2.0, psycopg2-binary>=2.9.9
# -------------------------------------------------------------------

import os, re, sys
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

# ────────────────────────── Config & LLM ────────────────────────────
DB_PATH = "vehicles.db"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("Missing OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=OPENAI_API_KEY)
db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")

# Dynamically fetch table names so we never get out of sync
VALID_TABLES: List[str] = [tbl.lower() for tbl in db.get_usable_table_names()]

# ───────────────────────────── Tools ────────────────────────────────
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()          # [0] list_tables, [1] sql_db_schema, …

@tool
def db_query_tool(query: str) -> str:
    """Run the supplied SQL query against the SQLite database and return its results."""
    result = db.run_no_throw(query)
    return str(result) if result else "Error: Query failed."

# ──────────────────────── Agent State Type ──────────────────────────
class AgentState(TypedDict):
    query: str
    sql_result: Optional[str]
    schema_help: Optional[str]
    final_response: Optional[str]

# ─────────────────────────── Helper Fns ─────────────────────────────
def extract_valid_table(query: str) -> Optional[str]:
    """Return the LAST token in the query that matches a real table name."""
    for word in reversed(re.findall(r"\b\w+\b", query.lower())):
        if word in VALID_TABLES:
            return word
    return None

# ───────────────────────────── Nodes ────────────────────────────────
def planner_node(state: AgentState) -> str:
    q = state["query"].lower()
    if any(tok in q for tok in ("table", "column", "schema", "structure", "describe", "list")):
        return "schema_agent"
    return "sql_agent"

# SQL-agent node (LangChain built-in SQL agent + fallback)
sql_agent_executor: AgentExecutor = create_sql_agent(llm=llm, toolkit=toolkit, verbose=True)

sql_agent_node = RunnableWithFallbacks(
    runnable=sql_agent_executor,
    # If the agent itself errors, emit a ToolMessage so the graph continues
    fallbacks=[RunnableLambda(lambda _: ToolMessage(tool_call_id="sql_query_tool",
                                                    content="Error: query execution failed."))]
)

def run_sql_agent(state: AgentState) -> AgentState:
    output = sql_agent_node.invoke({"input": state["query"]})
    return {**state, "sql_result": str(output)}

# Schema-helper node (lists tables OR describes a specific table)
def run_schema_agent(state: AgentState) -> AgentState:
    q_lower = state["query"].lower()
    if re.search(r"\blist\b|\bshow\b.*\btables?\b", q_lower):
        result = tools[0].invoke({})                 # sql_db_list_tables
    else:
        table_name = extract_valid_table(state["query"])
        if not table_name:
            return {**state, "schema_help": "❌ No valid table name found in query."}
        result = tools[1].invoke({"table_names": [table_name]})  # sql_db_schema
    return {**state, "schema_help": str(result)}

# Summariser node (final answer)
def summarizer_node(state: AgentState) -> AgentState:
    summary = state.get("sql_result") or state.get("schema_help") or "No output generated."
    return {**state, "final_response": summary}

# ───────────────────────── LangGraph ────────────────────────────────
graph = StateGraph(AgentState)

graph.add_node("planner", RunnableLambda(lambda s: s))  # identity pass-through
graph.add_node("sql_agent", run_sql_agent)
graph.add_node("schema_agent", run_schema_agent)
graph.add_node("summarizer", summarizer_node)

graph.set_entry_point("planner")
graph.add_conditional_edges("planner", planner_node)
graph.add_edge("sql_agent", "summarizer")
graph.add_edge("schema_agent", "summarizer")
graph.add_edge("summarizer", END)

app = graph.compile()

# ──────────────────────────── Streamlit UI ──────────────────────────
st.set_page_config(page_title="🧠 LangGraph SQL Chatbot", layout="centered")
st.title("🧠 LangGraph SQL Chatbot")

user_prompt = st.text_input("Ask a database question:", placeholder="e.g. show schema for trips")

if st.button("Run") and user_prompt:
    with st.spinner("Running agents …"):
        final_state = app.invoke({"query": user_prompt})
        st.success("Done!")
        st.subheader("Answer:")
        st.markdown(final_state["final_response"])
        st.divider()
        st.subheader("🪄 Debug Trace")
        st.json(final_state)

# ────────────────────────── CLI fallback ────────────────────────────
if __name__ == "__main__" and not st.runtime.exists():
    uq = sys.argv[1] if len(sys.argv) > 1 else "list tables"
    res = app.invoke({"query": uq})
    print("\n📤  Final Answer:\n", res["final_response"])
