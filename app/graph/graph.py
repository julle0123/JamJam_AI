# app/graph/graph.py
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from app.graph.state import AgentState
from app.graph import nodes
from app.graph.tools import classify_emotion_tool, rag_search_tool, summarize_tool

def build_agent_graph():
    g = StateGraph(AgentState)

    g.add_node("agent", nodes.call_agent)
    g.add_node("tools", ToolNode([classify_emotion_tool, rag_search_tool, summarize_tool]))
    g.add_node("tools_to_prompt", nodes.tools_to_prompt)
    g.add_node("finalize", nodes.finalize)

    g.set_entry_point("agent")
    g.add_conditional_edges("agent", nodes.should_call_tools, {
        "tools": "tools",
        "finalize": "finalize",
    })

    # Tool 실행 → 결과 요약 저장 → 다시 agent 호출
    g.add_edge("tools", "tools_to_prompt")
    g.add_edge("tools_to_prompt", "agent")

    g.add_edge("finalize", END)
    return g.compile()
