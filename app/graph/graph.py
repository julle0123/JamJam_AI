# app/graph/graph.py
# LangGraph 상태 머신 구성. 선요약/회상 선주입 → 에이전트 → (도구) → 최종응답.
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from app.graph.state import AgentState
from app.graph import nodes
from app.graph.tools import classify_emotion_tool, rag_search_tool, summarize_tool

def build_agent_graph(checkpointer=None):
    # 그래프 노드 구성
    g = StateGraph(AgentState)

    # 0) 맥락 선주입(요약/회상/감정)
    g.add_node("preload_context", nodes.preload_context)

    # 1) 에이전트(도구 자율 호출)
    g.add_node("agent", nodes.call_agent)

    # 2) 도구 실행/정리
    g.add_node("tools", ToolNode([classify_emotion_tool, rag_search_tool, summarize_tool]))
    g.add_node("tools_to_prompt", nodes.tools_to_prompt)

    # 3) 종료
    g.add_node("finalize", nodes.finalize)

    # 흐름 정의
    g.set_entry_point("preload_context")
    g.add_edge("preload_context", "agent")
    g.add_conditional_edges("agent", nodes.should_call_tools, {
        "tools": "tools",
        "finalize": "finalize",
    })
    g.add_edge("tools", "tools_to_prompt")
    g.add_edge("tools_to_prompt", "agent")
    g.add_edge("finalize", END)

    return g.compile(checkpointer=checkpointer) if checkpointer else g.compile()
