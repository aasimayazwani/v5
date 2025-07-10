# app.py — LangGraph SQL Chatbot (with smarter routing & fallback)

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

# ─── Config ─────────────────────────────────────────────────────────
DB_PATH = "vehicles.db"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("Missing OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=OPENAI_API_KEY)
db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")
VALID_TABLES: List[str] = [tbl.lower() for tbl in db.get_usable_table_names()]

toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()               # 0=list_tables, 1=sql_db_schema, …

# ─── Custom helper tool (docstring required!) ──────────────────────
@tool
def db_query_tool(query: str) -> str:
    """Run a raw SQL query against the SQLite database."""
    result = db.run_no_throw(query)
    return str(result) if result else "Error: Query failed."

# ─── Agent state type ───────────────────────────────────────────────
class AgentState(TypedDict):
    query: str
    sql_result: Optional[str]
    schema_help: Optional[str]
    final_response: Optional[str]

# ─── Utility --------------------------------------------------------
def extract_valid_table(text: str) -> Optional[str]:
    for w in reversed(re.findall(r"\b\w+\b", text.lower())):
        if w in VALID_TABLES:
            return w
    return None

# ─── Nodes ----------------------------------------------------------
def planner_node(state: AgentState) -> str:
    q = state["query"].lower()
    # Broadened keyword check
    if any(tok in q for tok in ("table", "column", "schema", "structure",
                                "describe", "list", "database")):
        return "schema_agent"
    return "sql_agent"

sql_exec: AgentExecutor = create_sql_agent(llm=llm, toolkit=toolkit, verbose=True)

sql_node = RunnableWithFallbacks(
    runnable=sql_exec,
    fallbacks=[RunnableLambda(lambda _: ToolMessage(
        tool_call_id="sql_query_tool",
        content="Error: query execution failed."
    ))]
)

def run_sql_agent(state: AgentState) -> AgentState:
    res = sql_node.invoke({"input": state["query"]})
    return {**state, "sql_result": str(res)}

def run_schema_agent(state: AgentState) -> AgentState:
    q = state["query"].lower()
    if re.search(r"\blist\b|\bshow\b.*\btables?\b", q):
        res = tools[0].invoke({})
    else:
        tbl = extract_valid_table(state["query"])
        if not tbl:
            return {**state, "schema_help": "❌ No valid table name found."}
        res = tools[1].invoke({"table_names": [tbl]})
    return {**state, "schema_help": str(res)}

def summariser_node(state: AgentState) -> AgentState:
    output = state.get("sql_result") or state.get("schema_help") \
             or "No output generated."
    return {**state, "final_response": output}

# ─── Build LangGraph -----------------------------------------------
graph = StateGraph(AgentState)

graph.add_node("planner", RunnableLambda(lambda s: s))
graph.add_node("sql_agent", run_sql_agent)
graph.add_node("schema_agent", run_schema_agent)
graph.add_node("summariser", summariser_node)

graph.set_entry_point("planner")
graph.add_conditional_edges("planner", planner_node)

# **Dynamic fallback**: if SQL returns an error, jump to schema_agent
def sql_result_route(state: AgentState) -> str:
    return "schema_agent" if state.get("sql_result", "").startswith("Error") \
           else "summariser"

graph.add_conditional_edges("sql_agent", sql_result_route)

graph.add_edge("schema_agent", "summariser")
graph.add_edge("summariser", END)

app = graph.compile()

# ─── Streamlit UI ---------------------------------------------------
st.set_page_config(page_title="🧠 LangGraph SQL Chatbot", layout="centered")
st.title("🧠 LangGraph SQL Chatbot")

prompt = st.text_input("Ask a database question:",
                       placeholder="e.g. describe routes")
if st.button("Run") and prompt:
    with st.spinner("Thinking…"):
        state = app.invoke({"query": prompt})
    st.success("Done!")
    st.subheader("Answer:")
    st.markdown(state["final_response"])
    with st.expander("🔍 Debug trace"):
        st.json(state)

# ─── CLI fallback ---------------------------------------------------
if __name__ == "__main__" and not st.runtime.exists():
    q = sys.argv[1] if len(sys.argv) > 1 else "tell me about the database"
    res = app.invoke({"query": q})
    print("\nAnswer:", res["final_response"])
