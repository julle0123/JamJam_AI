from langgraph.graph import StateGraph, END
from app.graph.state import ChatState
from app.graph import nodes

def build_graph():
    graph = StateGraph(ChatState)

    graph.add_node("decide_tools", nodes.decide_tools_node)
    graph.add_node("classify_emotion", nodes.classify_emotion_node)
    graph.add_node("rag_search", nodes.rag_node)
    graph.add_node("summarize", nodes.summary_node)
    graph.add_node("generate_response", nodes.generate_response_node)

    graph.set_entry_point("decide_tools")

    # 조건 분기: LLM 판단 결과에 따라 선택된 툴만 실행
    def tool_selector(state):
        selected = []
        if state["tool_flags"].get("use_emotion"): selected.append("classify_emotion")
        if state["tool_flags"].get("use_rag"): selected.append("rag_search")
        if state["tool_flags"].get("use_summary"): selected.append("summarize")
        selected.append("generate_response")  # 항상 실행
        return selected

    graph.add_conditional_edges("decide_tools", tool_selector)

    # 툴 실행 후 응답 노드로 합류
    graph.add_edge("classify_emotion", "generate_response")
    graph.add_edge("rag_search", "generate_response")
    graph.add_edge("summarize", "generate_response")
    graph.add_edge("generate_response", END)

    return graph.compile()
