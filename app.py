# app.py — Multi-Agent LangGraph SQL Chatbot

import os
from typing import TypedDict, Optional
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda, RunnableWithFallbacks
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import AgentExecutor, create_sql_agent
from langchain_core.prompts import ChatPromptTemplate

# ------------------ Config ---------------------
DB_PATH = "vehicles.db"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("Missing OPENAI_API_KEY env var")

llm = ChatOpenAI(model="gpt-4o", temperature=0)
db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()

# ------------------ Agent State ---------------------
class AgentState(TypedDict):
    query: str
    sql_result: Optional[str]
    schema_help: Optional[str]
    final_response: Optional[str]

# ------------------ Nodes ---------------------

# Planner Node
def planner_node(state: AgentState) -> str:
    q = state["query"].lower()
    if any(word in q for word in ["table", "column", "schema", "structure"]):
        return "schema_agent"
    return "sql_agent"

# SQL Agent Node
sql_agent_executor = create_sql_agent(llm=llm, toolkit=toolkit, verbose=True)
sql_agent_node = RunnableWithFallbacks(
    runnable=sql_agent_executor,
    fallbacks=[RunnableLambda(lambda s: ToolMessage(tool_call_id="sql_query_tool", tool_output="Query failed."))]
)

def run_sql_agent(state: AgentState) -> AgentState:
    output = sql_agent_node.invoke({"input": state["query"]})
    return {**state, "sql_result": str(output)}

# Schema Agent Node (uses list/describe table tools)
def run_schema_agent(state: AgentState) -> AgentState:
    if "list" in state["query"]:
        result = tools[0].invoke({})
    else:
        table_name = state["query"].split(" ")[-1]
        result = tools[1].invoke({"table_name": table_name})
    return {**state, "schema_help": str(result)}

# Final Answer Generator
def summarizer_node(state: AgentState) -> AgentState:
    summary = state.get("sql_result") or state.get("schema_help") or "No output generated."
    return {**state, "final_response": summary}

# ------------------ LangGraph ---------------------

graph = StateGraph(AgentState)

graph.add_node("planner", RunnableLambda(lambda s: s))  # just routes
graph.add_node("sql_agent", run_sql_agent)
graph.add_node("schema_agent", run_schema_agent)
graph.add_node("summarizer", summarizer_node)

# Entry point
graph.set_entry_point("planner")
graph.add_conditional_edges("planner", planner_node)
graph.add_edge("sql_agent", "summarizer")
graph.add_edge("schema_agent", "summarizer")
graph.add_edge("summarizer", END)

app = graph.compile()

# ------------------ Run it ---------------------
if __name__ == "__main__":
    import sys
    user_query = sys.argv[1] if len(sys.argv) > 1 else "List all tables in the database"
    result = app.invoke({"query": user_query})
    print("\n📤 Final Answer:\n", result["final_response"])
