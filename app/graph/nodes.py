import json
from app.graph.prompts import load_prompt_template
from app.core.client import llm
from app.graph.tools import classify_emotion, summarize
from app.services.memory import recall_or_general_context

# 1) 툴 선택 노드 (그대로)
def decide_tools_node(state):
    prompt = load_prompt_template("tool_decision").format(user_input=state["user_input"])
    decision = llm.invoke(prompt)
    try:
        state["tool_flags"] = json.loads(decision)
    except:
        state["tool_flags"] = {"use_emotion": False, "use_rag": False, "use_summary": False}
    return state

# 2) 감정분류
def classify_emotion_node(state):
    state["emotion"] = classify_emotion(state["user_input"])
    return state

# 3) RAG (회상 의도면 ‘특정 시점 기억’)
def rag_node(state):
    db = state.get("db")
    state["memory"] = recall_or_general_context(
        user_input=state["user_input"],
        member_id=state["member_id"],
        db=db,
        top_k=3,
        recall_window_min=30,
    )
    return state

# 4) 요약
def summary_node(state):
    state["summary"] = summarize(state["member_id"])
    return state

# 5) 응답 생성
def generate_response_node(state):
    history_text = "\n".join(
        [f"{m.type.upper()}: {m.content}" for m in state.get("history", [])]
    )

    prompt = load_prompt_template("response").format(
        user_input=state["user_input"],
        emotion=state.get("emotion", ""),
        memory=state.get("memory", ""),
        summary=state.get("summary", ""),
        history=history_text
    )
    state["response"] = llm.invoke(prompt)
    return state
